"""Re-certify raw batches and compare every exported LeRobot frame to its packet."""
import argparse,base64,hashlib,io,json
from pathlib import Path
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from checkpoints import certify_folders
from dataset import build_index,training_normalization

def validate(inputs,output):
    records,paths,attempts=certify_folders(inputs);rows=build_index(records)
    index=json.loads((output/'index.json').read_text());normalization=json.loads((output/'normalization.json').read_text())
    if index['episodes']!=rows or index['attempts']!=attempts:raise ValueError('Export identity/status differs from raw certification')
    if normalization!=training_normalization(records):raise ValueError('Normalization differs from accepted training samples')
    if json.loads((output/'train/meta/stats.json').read_text())!=json.loads((output/'validation/meta/stats.json').read_text()):
        raise ValueError('Validation statistics differ from training')
    result={'schema':'hsr-real-demo-round-trip-v1','normalizationSha256':hashlib.sha256((output/'normalization.json').read_bytes()).hexdigest(),'partitions':{}}
    for side in ('train','validation'):
        chosen=[(row,path)for row,path in zip(rows,paths)if row['status']=='accepted'and row['partition']==side]
        dataset=LeRobotDataset(f'local/hsr-{side}',root=output/side,video_backend='pyav')
        if dataset.meta.total_episodes!=len(chosen)or len(dataset)!=sum(r['sampleCount']for r,_ in chosen):raise ValueError('LeRobot episode/frame count mismatch')
        allowed={'observation.images.head','observation.images.hand','observation.state','action','timestamp','frame_index','episode_index','index','task_index'}
        if set(dataset.features)!=allowed:raise ValueError('Oracle or unknown policy feature')
        offset=0
        for episode_index,(row,path)in enumerate(chosen):
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['sourceSha256']:raise ValueError('Raw packet changed during verification')
            episode=json.loads(raw)
            for i,sample in enumerate(episode['samples']):
                actual=dataset[offset+i]
                if actual['task']!=episode['instruction']or actual['episode_index'].item()!=episode_index or actual['frame_index'].item()!=i:
                    raise ValueError('Instruction or whole-episode boundary mismatch')
                if abs(actual['timestamp'].item()-i/10)>1e-5:raise ValueError('Native10Hz cadence mismatch')
                for key in ('state','action'):
                    np.testing.assert_allclose(actual['observation.state'if key=='state'else'action'].numpy(),sample[key],atol=1e-7)
                for view in ('head','hand'):
                    image=episode['images'][sample['images'][view]]
                    pixels=np.array(Image.open(io.BytesIO(base64.b64decode(image['dataUrl'].split(',')[1]))))
                    decoded=(actual[f'observation.images.{view}'].numpy().transpose(1,2,0)*255).round().astype('uint8')
                    np.testing.assert_array_equal(decoded,pixels)
            offset+=len(episode['samples'])
        result['partitions'][side]={'episodes':len(chosen),'pairs':offset,'allRgbStateActionTaskValues':'verified'}
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,nargs='+',required=True)
    parser.add_argument('--export',dest='output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(validate(args.input,args.output)))
