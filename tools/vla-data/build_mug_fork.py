"""Recollected mug dataset: the parent's ball episodes, newly certified gentle-mug episodes, and the parent's
normalization byte-for-byte. Certification needs the collection commit in Git, so it runs where the data was collected:
  build_mug_fork.py certify --input RAW... --out certified.json          (collection machine)
  build_mug_fork.py build --parent DATASET --certified certified.json --raw-root RAW_ROOT --source-commit SHA --output NEW   (training host)
With --append the parent is an earlier mug fork: nothing is removed and the new records (e.g. takeover segments) are added."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/vla-data'),str(ROOT/'tools/vla-train')]

MUG='Put the mug in the basket.'
TEACHER='hsr-mug-gentle-v1';DART='hsr-mug-gentle-dart-v1';SCENES=('hsr-stationary-v5','hsr-stationary-v6-recovery','hsr-stationary-v12-takeover')

def sha(path):return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def old_mug_episodes(episode_map,index_rows):
    """Native episode indices of one partition whose dataset-index instruction is the mug."""
    by_id={r['episodeId']:r for r in index_rows}
    return [e['episode_index']for e in episode_map if by_id[e['episodeId']]['instruction']==MUG]

GENERATIONS={'hsr-mug-gentle-v1':('hsr-mug-gentle-v1','hsr-mug-gentle-dart-v1'),'hsr-mug-gentle-dart-v1':('hsr-mug-gentle-v1','hsr-mug-gentle-dart-v1'),
             'hsr-mug-gentle-v2':('hsr-mug-gentle-v2','hsr-mug-gentle-dart-v2'),'hsr-mug-gentle-dart-v2':('hsr-mug-gentle-v2','hsr-mug-gentle-dart-v2'),
             'hsr-mug-gentle-v3':('hsr-mug-gentle-v3','hsr-mug-gentle-dart-v3'),'hsr-mug-gentle-dart-v3':('hsr-mug-gentle-v3','hsr-mug-gentle-dart-v3'),
             'hsr-mug-gentle-v4':('hsr-mug-gentle-v4','hsr-mug-gentle-dart-v4'),'hsr-mug-gentle-dart-v4':('hsr-mug-gentle-v4','hsr-mug-gentle-dart-v4')}
V2_SCENES=('hsr-stationary-v5','hsr-stationary-v6-recovery')

def teacher_generation(records):
    """(plain, DART) teacher versions shared by every new record; a fork never mixes teacher generations."""
    found={GENERATIONS.get(r['episode']['collection']['teacherVersion'])for r in records}
    if len(found)!=1 or None in found:raise ValueError('New mug records must share one gentle teacher generation')
    return found.pop()

def verify_append_generation(parent_fork,records):
    """Appending keeps the earlier fork's teacher generation (v1 data and v2 data never share a fork)."""
    if parent_fork.get('teacherVersion')!=teacher_generation(records)[0]:raise ValueError('Appended records must match the fork teacher generation')

def verify_new_records(records,parent_attempts):
    """Only gentle-teacher mug demonstrations, on the normal or mug recovery scene, on seeds never attempted for the mug.
    DART demonstrations (noise-injected execution, clean labels) are the exception: normal scene and train seeds only, and
    they may repeat a seed of an earlier mug demonstration, but each seed at most once among the DART records."""
    used={a['seed']for a in parent_attempts if a['instruction']==MUG};dart_seeds=set()
    if not records:return
    plain,dart_version=teacher_generation(records)
    for record in records:
        episode=record['episode'];collection=episode['collection'];dart=collection['teacherVersion']==dart_version
        scenes=('hsr-stationary-v5',)if dart else SCENES if plain==TEACHER else V2_SCENES
        if episode['instruction']!=MUG or collection['teacherVersion']not in(plain,dart_version)or collection['sceneVersion']not in scenes:
            raise ValueError('Only gentle-teacher mug demonstrations can enter the mug fork')
        if dart:
            if not 1000<=episode['seed']<=1999 or episode['seed']in dart_seeds:raise ValueError(f"DART seed {episode['seed']} must be a train seed used once")
            dart_seeds.add(episode['seed'])
        elif episode['seed']in used:raise ValueError(f"Mug seed {episode['seed']} was already used by the parent dataset")

BASE_MARGIN_M,BASE_MARGIN_RAD=0.006,0.012

def apply_base_margin(records):
    """Accepted episodes whose sampled base pose left 6 mm / 12 mrad (40 % of the base-drift limit) become rejected."""
    import copy,math
    out,count=[],0
    for record in records:
        episode=record['episode'];x0,z0,yaw0=episode['startBase'];poses=episode['audit']['basePoses']
        far=max(math.hypot(x-x0,z-z0)for x,z,_ in poses)>BASE_MARGIN_M or max(abs(yaw-yaw0)for _,_,yaw in poses)>BASE_MARGIN_RAD
        if far and record['acceptance']['status']=='accepted':
            record=copy.deepcopy(record);record['acceptance']['status']='rejected';record['acceptance']['reasons']=[*record['acceptance'].get('reasons',[]),'base-margin-exceeded'];count+=1
        out.append(record)
    return out,count

def lineage(parent_manifest,normalization):
    """(dataset f289 was trained on, its normalization, appended-to fork or None). Appending to a mug fork keeps the fork's
    original parent, whose normalization the fork must still carry byte-for-byte."""
    fork=parent_manifest.get('mugDataFork')
    if not fork:return identity(parent_manifest),identity(normalization),None
    if identity(normalization)!=fork['parentNormalizationSha256']:raise ValueError('Mug fork lost its inherited normalization')
    return fork['parentDatasetSha256'],fork['parentNormalizationSha256'],identity(parent_manifest)

def native_frames(root,skip=frozenset()):
    """(frame hash, task, episode, frame) per row in global order, renumbering episodes after skipped ones."""
    import pandas as pd
    from recovery_dataset import native_rows,frame_hash
    tasks=pd.read_parquet(root/'meta/tasks.parquet');names=dict(zip(tasks.task_index,tasks.index))
    rows=sorted(native_rows(root),key=lambda row:row['index']);kept=sorted({r['episode_index']for r in rows}-set(skip))
    renumber={old:new for new,old in enumerate(kept)}
    return [(frame_hash(r['observation.state'],r['action'],[r[f'observation.images.{v}']['bytes']for v in ['head','hand']]),names[r['task_index']],renumber[r['episode_index']],r['frame_index'])
            for r in rows if r['episode_index']in renumber]

def certify(inputs,root):
    """Certify collector folders; packet paths are stored relative to root so the bundle can move hosts."""
    from checkpoints import certify_folders
    from dataset import build_index
    records,paths,attempts=certify_folders(inputs);build_index(records)
    return {'schema':'hsr-mug-fork-certification-v1','records':records,'paths':[Path(p).resolve().relative_to(root.resolve()).as_posix()for p in paths],'attempts':attempts}

def build(parent,certified,raw_root,output,source_commit,append=False):
    from hsr_training import verify_dataset
    from dataset import build_index
    from export_demos import write_partition
    from recovery_dataset import verify_merged_frames
    from prepare_colab import dataset_files
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.aggregate import aggregate_datasets
    if output.exists():raise FileExistsError('Preserve previous fork output')
    parent_manifest,normalization=verify_dataset(parent,mug_data_fork=append)
    original_dataset,original_normalization,appended_to=lineage(parent_manifest,normalization)
    index=json.loads((parent/'index.json').read_text())
    if certified.get('schema')!='hsr-mug-fork-certification-v1':raise ValueError('Certified collection bundle required')
    records,margin_rejections=apply_base_margin(certified['records']);attempts=certified['attempts'];paths=[raw_root/p for p in certified['paths']];rows=build_index(records)
    for record,path in zip(records,paths):
        if sha(path)!=record['sourceSha256']:raise ValueError('Packet changed after certification')
    verify_new_records(records,index['attempts'])
    if append:verify_append_generation(parent_manifest['mugDataFork'],records)
    work,merged=output/'work',output/'dataset';work.mkdir(parents=True);merged.mkdir()
    partitions={}
    for side in ['train','validation']:
        episode_map=json.loads((parent/side/'hsr-episode-map.json').read_text())
        selected=[(r,p)for r,p,row in zip(records,paths,rows)if row['status']=='accepted'and row['partition']==side]
        if append:
            # The earlier fork's partition is kept whole (read in place); a side without new episodes is copied as is.
            kept,kept_map=parent/side,episode_map
            if not selected:
                shutil.copytree(parent/side,merged/side);partitions[side]={'deletedParentMugEpisodes':0,'keptParentEpisodes':len(kept_map),'addedEpisodes':0};continue
        else:
            drop=old_mug_episodes(episode_map,index['episodes']);kept=work/f'{side}-kept'
            delete_episodes(LeRobotDataset(f'local/parent-{side}',root=parent/side,video_backend='pyav'),drop,output_dir=kept,repo_id=f'local/kept-{side}')
            kept_map=[{**e,'episode_index':i}for i,e in enumerate(e for e in episode_map if e['episode_index']not in set(drop))]
            (kept/'hsr-episode-map.json').write_text(json.dumps(kept_map,indent=2)+'\n')
            if native_frames(parent/side,drop)!=native_frames(kept):raise ValueError('Deleting parent mug episodes changed the kept frames')
            if not selected:raise ValueError(f'No accepted gentle mug episodes for {side}')
        write_partition(records,paths,work/f'{side}-new',normalization,side)
        aggregate_datasets([f'local/kept-{side}',f'local/new-{side}'],f'local/fork-{side}',roots=[kept,work/f'{side}-new'],aggr_root=merged/side,concatenate_data=False,concatenate_videos=False)
        partitions[side]={'deletedParentMugEpisodes':0 if append else len(drop),'keptParentEpisodes':len(kept_map),'addedEpisodes':len(selected),**verify_merged_frames(kept,selected,merged/side)}
        new_map=json.loads((work/f'{side}-new'/'hsr-episode-map.json').read_text())
        (merged/side/'hsr-episode-map.json').write_text(json.dumps(kept_map+[{**e,'episode_index':len(kept_map)+e['episode_index']}for e in new_map],indent=2)+'\n')
    # The parent's train statistics (identical in both partitions) carry the inherited normalization.
    for side in ['train','validation']:shutil.copyfile(parent/'train/meta/stats.json',merged/side/'meta/stats.json')
    shutil.copyfile(parent/'normalization.json',merged/'normalization.json')
    keep=(lambda r:True)if append else(lambda r:r['instruction']!=MUG)
    (merged/'index.json').write_text(json.dumps({'schema':'hsr-demo-index-v1','attempts':[a for a in index['attempts']if keep(a)]+attempts,
                                                 'episodes':[r for r in index['episodes']if keep(r)]+rows},indent=2)+'\n')
    all_rows=json.loads((merged/'index.json').read_text())['episodes']
    if len({r['episodeId']for r in all_rows})!=len(all_rows):raise ValueError('An appended episode is already in the fork')
    fork={'schema':'hsr-mug-data-fork-v1','parentDatasetSha256':original_dataset,'parentNormalizationSha256':original_normalization,
          **({'appendedTo':appended_to}if append else{}),
          'teacherVersion':teacher_generation(records)[0],**({'dartTeacherVersion':teacher_generation(records)[1]}if any(r['episode']['collection']['teacherVersion']==teacher_generation(records)[1]for r in records)else{}),'sceneVersions':list(SCENES),'partitions':partitions,'certifiedAttempts':len(attempts),'baseMarginRejections':margin_rejections,'acceptedEpisodes':sum(r['status']=='accepted'for r in rows)}
    manifest={'schema':'hsr-training-bundle-v1','datasetReady':True,'sourceCommit':source_commit,'review':{'schema':'hsr-mug-data-fork-automatic-checks-v1','checks':['certification','kept parent frames','merged frame round trip','inherited normalization']},
              'roundTrip':partitions,'files':[{'path':p.relative_to(merged).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}for p in dataset_files(merged)],'mugDataFork':fork}
    (merged/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    verify_dataset(merged,mug_data_fork=True)
    shutil.rmtree(work)
    return {'datasetSha256':identity(manifest),'normalizationSha256':fork['parentNormalizationSha256'],'parentDatasetSha256':fork['parentDatasetSha256'],'partitions':partitions}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    c=commands.add_parser('certify');c.add_argument('--input',type=Path,nargs='+',required=True);c.add_argument('--root',type=Path,required=True);c.add_argument('--out',type=Path,required=True)
    b=commands.add_parser('build');b.add_argument('--parent',type=Path,required=True);b.add_argument('--certified',type=Path,required=True);b.add_argument('--raw-root',type=Path,required=True);b.add_argument('--output',type=Path,required=True);b.add_argument('--source-commit',required=True);b.add_argument('--append',action='store_true')
    args=parser.parse_args()
    if args.command=='certify':
        if args.out.exists():raise FileExistsError('Preserve previous certification')
        result=certify(args.input,args.root);args.out.write_text(json.dumps(result)+'\n')
        print(json.dumps({'attempts':len(result['attempts']),'accepted':sum(r['acceptance']['status']=='accepted'for r in result['records'])}))
    else:print(json.dumps(build(args.parent,json.loads(args.certified.read_text()),args.raw_root,args.output,args.source_commit,args.append),indent=1))
