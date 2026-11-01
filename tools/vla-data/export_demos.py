"""Certify fresh capture folders; export accepted whole episodes and train-only stats."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import base64,io
from checkpoints import certify_folders
from dataset import build_index, training_normalization
from export import installed_lerobot_revision, LEROBOT_REVISION

def preserve_source_jpegs(selected, output):
    """Keep original camera bytes in native Image columns; never re-encode RGB."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    for path in sorted((output/'data').rglob('*.parquet')):
        table=pq.read_table(path)
        encoded={view:[] for view in ('head','hand')}
        current_index=None;episode=None
        for episode_index,frame_index in zip(table.column('episode_index').to_pylist(),table.column('frame_index').to_pylist()):
            if episode_index!=current_index:
                if type(episode_index)is not int or not 0<=episode_index<len(selected):
                    raise ValueError('Native episode index differs from certified sources')
                record,source=selected[episode_index];raw=Path(source).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=record['sourceSha256']:
                    raise ValueError('Packet changed before preserving source JPEGs')
                episode=json.loads(raw);current_index=episode_index
            if type(frame_index)is not int or not 0<=frame_index<len(episode['samples']):
                raise ValueError('Native frame index differs from certified sources')
            sample=episode['samples'][frame_index]
            for view in encoded:
                image=episode['images'][sample['images'][view]]
                data=base64.b64decode(image['dataUrl'].split(',')[1],validate=True)
                if not data.startswith(b'\xff\xd8\xff'):raise ValueError('Expected original camera JPEG')
                encoded[view].append({'bytes':data,'path':None})
        for view,values in encoded.items():
            name=f'observation.images.{view}';index=table.schema.get_field_index(name)
            field=table.schema.field(index)
            table=table.set_column(index,field,pa.array(values,type=field.type))
        temporary=path.with_suffix('.parquet.tmp')
        pq.write_table(table,temporary,compression='zstd');temporary.replace(path)

def write_partition(records, source_paths, output, normalization, partition):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    selected=[(item,path) for item,path in zip(records,source_paths)
              if item['acceptance']['status']=='accepted' and next(r for r in build_index([item]))['partition']==partition]
    if not selected: return None
    manifest=selected[0][0]['episode']['manifest']
    features={'observation.state':{'dtype':'float32','shape':(6,),'names':manifest['axisNames']},
              'action':{'dtype':'float32','shape':(6,),'names':manifest['axisNames']}}
    for view in ['head','hand']:features[f'observation.images.{view}']={'dtype':'image','shape':(3,240,320),'names':['channels','height','width']}
    dataset=LeRobotDataset.create(f'local/hsr-{partition}',fps=10,features=features,root=output,
                                 robot_type=manifest['robotModelId'],use_videos=False,video_backend='pyav')
    episode_map=[]
    try:
        for index,(record,path)in enumerate(selected):
            raw=Path(path).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=record['sourceSha256']:raise ValueError('Packet changed after certification')
            episode=json.loads(raw)
            for sample in episode['samples']:
                frame={'observation.state':np.array(sample['state'],dtype=np.float32),
                       'action':np.array(sample['action'],dtype=np.float32),'task':episode['instruction']}
                if not all(np.isfinite(frame[k]).all() for k in ['observation.state','action']):raise ValueError('Feature overflows float32')
                for view in ['head','hand']:
                    image=episode['images'][sample['images'][view]]
                    data=base64.b64decode(image['dataUrl'].split(',')[1],validate=True)
                    frame[f'observation.images.{view}']=np.asarray(Image.open(io.BytesIO(data))).copy()
                dataset.add_frame(frame)
            dataset.save_episode()
            episode_map.append({'episode_index':index,'episodeId':episode['collection']['episodeId'],
                                'sourceSha256':record['sourceSha256'],'seed':episode['seed']})
    finally:dataset.finalize()
    preserve_source_jpegs(selected,output)
    # Native image statistics remain train-only (identity visual normalization).
    # Numeric means/std below override native automatic aggregation explicitly.
    stats=json.loads((output/'meta/stats.json').read_text())
    for key in ['state','action']:
        native_key='observation.state' if key=='state'else'action'
        stats[native_key]={**stats[native_key],'mean':normalization[key]['mean'],'std':normalization[key]['std']}
    (output/'meta/stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    (output/'hsr-episode-map.json').write_text(json.dumps(episode_map,indent=2)+'\n')
    return stats

def export_folders(inputs,output):
    if output.exists():raise FileExistsError('Preserve previous export: '+str(output))
    if installed_lerobot_revision()!=LEROBOT_REVISION:raise ValueError('Pinned LeRobot revision required')
    records,source_paths,attempts=certify_folders(inputs)
    rows=build_index(records)
    identities=[a['episodeId']for a in attempts]
    if len(set(identities))!=len(identities):raise ValueError('Duplicate attempt across input folders')
    normalization=training_normalization(records)
    output.mkdir(parents=True)
    (output/'normalization.json').write_text(json.dumps(normalization,indent=2)+'\n')
    (output/'index.json').write_text(json.dumps({'schema':'hsr-demo-index-v1','attempts':attempts,'episodes':rows},indent=2)+'\n')
    try:
        train_stats=write_partition(records,source_paths,output/'train',normalization,'train')
        val_stats=write_partition(records,source_paths,output/'validation',normalization,'validation')
        if val_stats is not None:
            # Copy all training statistics; even unused visual/min/max metadata
            # must never silently introduce validation-derived normalization.
            (output/'validation/meta/stats.json').write_text(json.dumps(train_stats,indent=2)+'\n')
        counts={p:{instruction:sum(r['partition']==p and r['status']=='accepted' and r['instruction']==instruction for r in rows)
                   for instruction in ['Put the mug in the basket.','Put the tennis ball in the basket.']}
                for p in ['train','validation']}
        status={'schema':'hsr-demo-export-v1','status':'complete','lerobotRevision':LEROBOT_REVISION,'counts':counts,
                'cameraStorage':'original-jpeg-bytes',
                'accepted':sum(r['status']=='accepted'for r in rows),'rejected':sum(r['status']=='rejected'for r in rows),
                'attempts':len(attempts),'normalizationSha256':hashlib.sha256((output/'normalization.json').read_bytes()).hexdigest(),
                'policyFeatures':['observation.images.head','observation.images.hand','observation.state','action','task']}
    except Exception as e:
        (output/'export-status.json').write_text(json.dumps({'status':'failed','error':str(e)},indent=2)+'\n');raise
    (output/'export-status.json').write_text(json.dumps(status,indent=2)+'\n')
    return status

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(export_folders(args.input,args.output)))
