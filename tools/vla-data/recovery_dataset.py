"""Staged offline augmentation; never marks a dataset ready without fresh review."""
import argparse,base64,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

PREPARATION_FILES=(
 'src/robot/hsr-description.json','package.json','pnpm-lock.yaml',
 'tools/vla-data/recovery_dataset.py','tools/vla-data/certification.py','tools/vla-data/dataset.py',
 'tools/vla-data/checkpoints.py','tools/vla-data/catalog.py','tools/vla-data/export_demos.py','tools/vla-data/export.py',
 'tools/vla-data/collection-source-files.json','tools/vla-data/seed-registry.json',
 'tools/vla-train/hsr_training.py','tools/vla-train/recovery_training.py',
 'tools/vla-train/prepare_colab.py','tools/vla-train/colab_recording.py','tools/vla-train/requirements-colab.txt')

def preparation_hashes(source):
 validator_files=json.loads((source/'tools/vla-data/collection-source-files.json').read_text())
 return {name:sha(source/name)for name in sorted(set(PREPARATION_FILES)|set(validator_files))}

def verify_preparation_review(proof,gate,current):
 if not current or proof.get('preparationSourceHashes')!=current or gate.get('preparationSourceHashes')!=current:raise ValueError('Verified preparation source differs from reviewed/current source')

def sha(path):return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
def manifest_identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def verify_augmentation_profiles(records):
 for record in records:
  episode=record['episode'];collection=episode['collection']
  legacy=(collection['teacherVersion']=='hsr-stationary-teacher-v4'and collection['sceneVersion']=='hsr-stationary-v6-recovery'and episode['instruction']=='Put the mug in the basket.')
  ball=(collection['teacherVersion']=='hsr-ball-teacher-v1'and collection['sceneVersion']in('hsr-stationary-v5','hsr-stationary-v7-ball-regrasp','hsr-stationary-v8-physical-ball-loss','hsr-stationary-v9-empty-carry-loss')and episode['instruction']=='Put the tennis ball in the basket.')
  reseat=(collection['teacherVersion']=='hsr-ball-reseat-v1'and collection['sceneVersion']=='hsr-stationary-v10-shallow-reseat'and episode['instruction']=='Put the tennis ball in the basket.')
  if episode['kind']!='demonstration'or not(legacy or ball or reseat):raise ValueError('Declared recovery/ball augmentation profiles required')

def merge_index(original,rows,attempts):
 identities={r['episodeId']for r in original['attempts']}
 pairs={(r['seed'],r['instruction'])for r in original['attempts']}
 if len({(r['seed'],r['instruction'])for r in attempts})!=len(attempts):raise ValueError('Duplicate new seed-instruction pair')
 if identities&{r['episodeId']for r in attempts}or pairs&{(r['seed'],r['instruction'])for r in attempts}:raise ValueError('Previously used episode or seed-instruction pair')
 return {'schema':'hsr-demo-index-v1','attempts':original['attempts']+attempts,'episodes':original['episodes']+rows}

def verify_durable_cohort(catalog,rows,attempts):
 from catalog import catalog_rows
 import re
 actual=catalog_rows(catalog);saved_attempts=[r for b in catalog.get('batches',[])for r in b['attempts']]
 if (sorted(actual,key=lambda r:r['episodeId'])!=sorted(rows,key=lambda r:r['episodeId'])or
     sorted(saved_attempts,key=lambda r:r['episodeId'])!=sorted(attempts,key=lambda r:r['episodeId'])):raise ValueError('Durable recovery cohort differs from actual re-certification')
 artifacts=[b['artifact']for b in catalog['batches']]
 if not artifacts or any(a.get('storage')!='fresh-lfs-storage'or not re.fullmatch('[0-9a-f]{64}',a.get('sha256',''))or not re.fullmatch('[0-9a-f]{40}',a.get('commit',''))for a in artifacts):raise ValueError('Durable private retrieval required')
 return artifacts

def verify_final_review(gate,proof_sha,catalog_sha):
 if (gate.get('status')!='complete'or gate.get('unresolvedCritical',1)or gate.get('unresolvedImportant',1)or
     gate.get('augmentationProofSha256')!=proof_sha or gate.get('catalogSha256')!=catalog_sha):raise ValueError('Completed hash-bound fresh recovery review required')

def combine_normalization(original,added):
 # Combine population moments of two independently certified accepted-train sets.
 for key in ['schema','partition','status','registrySha256','axisNames','units','epsilon']:
  if original[key]!=added[key]:raise ValueError('Normalization contract changed')
 if set(original['episodeIds'])&set(added['episodeIds']):raise ValueError('Duplicate normalization episode')
 n,m=original['sampleCount'],added['sampleCount'];result={**original,'episodeIds':original['episodeIds']+added['episodeIds'],'sourceSha256':original['sourceSha256']+added['sourceSha256'],'sampleCount':n+m}
 for key in ['state','action']:
  a,b=np.asarray(original[key]['mean']),np.asarray(added[key]['mean']);av,bv=np.square(original[key]['rawStd']),np.square(added[key]['rawStd'])
  mean=(n*a+m*b)/(n+m);variance=(n*(av+np.square(a-mean))+m*(bv+np.square(b-mean)))/(n+m);raw=np.sqrt(variance)
  result[key]={'mean':mean.tolist(),'std':np.maximum(raw,result['epsilon']).tolist(),'rawStd':raw.tolist(),'zeroVarianceAxes':np.flatnonzero(raw<result['epsilon']).tolist()}
 return result

def native_rows(root):
 for path in sorted((root/'data').rglob('*.parquet')):yield from pq.read_table(path).to_pylist()

def frame_hash(state,action,images):
 h=hashlib.sha256()
 for value in [state,action]:h.update(np.asarray(value,dtype='<f4').tobytes())
 for raw in images:h.update(raw)
 return h.hexdigest()

def verify_native_chunks(ds,rows):
 if [r['index']for r in rows]!=list(range(len(rows))):raise ValueError('Native global indices are not contiguous')
 cursor=0
 for episode_index in range(ds.num_episodes):
  end=cursor
  while end<len(rows)and rows[end]['episode_index']==episode_index:end+=1
  if end==cursor or [r['frame_index']for r in rows[cursor:end]]!=list(range(end-cursor)):raise ValueError('Native episode frames are not contiguous')
  meta=ds.meta.episodes[episode_index]
  if meta['dataset_from_index']!=cursor or meta['dataset_to_index']!=end:raise ValueError('Native episode interval changed')
  for i in {cursor,*range(max(cursor,end-15),end)}:
   value=ds[i];expected=np.asarray([rows[min(i+k,end-1)]['action']for k in range(16)],dtype=np.float32)
   if not np.array_equal(value['action'].numpy(),expected)or not np.array_equal(value['action_is_pad'].numpy(),[i+k>=end for k in range(16)]):raise ValueError('Native action window or terminal padding changed')
  cursor=end
 if cursor!=len(rows):raise ValueError('Unmapped native frames')

def verify_merged_frames(original,new_selected,merged):
 import pandas as pd
 old_tasks=pd.read_parquet(original/'meta/tasks.parquet');old_names=dict(zip(old_tasks.task_index,old_tasks.index))
 old_map=json.loads((original/'hsr-episode-map.json').read_text());old=list(native_rows(original));old.sort(key=lambda row:row['index'])
 expected=[(frame_hash(row['observation.state'],row['action'],[row[f'observation.images.{view}']['bytes']for view in ['head','hand']]),old_names[row['task_index']],row['episode_index'],row['frame_index'])for row in old]
 del old
 for episode_index,(record,path)in enumerate(new_selected,start=len(old_map)):
  if sha(path)!=record['sourceSha256']:raise ValueError('New packet changed during round trip')
  episode=json.loads(path.read_text())
  for frame_index,sample in enumerate(episode['samples']):
   images=[base64.b64decode(episode['images'][sample['images'][view]]['dataUrl'].split(',')[1],validate=True)for view in ['head','hand']]
   expected.append((frame_hash(sample['state'],sample['action'],images),episode['instruction'],episode_index,frame_index))
 tasks=pd.read_parquet(merged/'meta/tasks.parquet');names=dict(zip(tasks.task_index,tasks.index));actual=list(native_rows(merged));actual.sort(key=lambda row:row['index'])
 found=[(frame_hash(row['observation.state'],row['action'],[row[f'observation.images.{view}']['bytes']for view in ['head','hand']]),names[row['task_index']],row['episode_index'],row['frame_index'])for row in actual]
 if found!=expected:raise ValueError('Merged JPEG/state/action/task/episode/frame bytes changed')
 if any(abs(row['timestamp']-row['frame_index']/10)>1e-5 for row in actual):raise ValueError('Merged10Hz cadence changed')
 from lerobot.datasets.lerobot_dataset import LeRobotDataset
 ds=LeRobotDataset('local/hsr-augmented',root=merged,video_backend='pyav',delta_timestamps={'action':[i/10 for i in range(16)]})
 verify_native_chunks(ds,actual)
 for i in [0,len(ds)-1]:
  if any(ds[i][f'observation.images.{v}'].shape!=(3,240,320)for v in ['head','hand']):raise ValueError('Native camera shape changed')
 return {'frames':len(actual),'episodes':ds.num_episodes,'originalJpegStateActionTaskEpisodeFrameBytes':'verified','nativeEpisodeBoundaryChunks':'verified'}

def build(original,inputs,output,source,catalog_path):
 if subprocess.check_output(['git','status','--porcelain'],cwd=source):raise ValueError('Save preparation source before certification')
 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip();code_hashes=preparation_hashes(source)
 sys.path[:0]=[str(source/'tools/vla-data'),str(source/'tools/vla-train')]
 from hsr_training import verify_dataset
 from checkpoints import certify_folders
 from dataset import build_index,training_normalization
 from export_demos import write_partition
 from export import installed_lerobot_revision,LEROBOT_REVISION
 from lerobot.datasets.aggregate import aggregate_datasets
 if installed_lerobot_revision()!=LEROBOT_REVISION:raise ValueError('Pinned LeRobot required')
 parent_manifest,parent_norm=verify_dataset(original);records,paths,attempts=certify_folders(inputs);rows=build_index(records)
 verify_augmentation_profiles(records)
 artifacts=verify_durable_cohort(json.loads(catalog_path.read_text()),rows,attempts)
 old_index=json.loads((original/'index.json').read_text());index=merge_index(old_index,rows,attempts)
 normalization=combine_normalization(parent_norm,training_normalization(records))
 if output.exists():raise FileExistsError('Preserve previous augmentation')
 output.mkdir(parents=True);recovery=output/'recovery-only';recovery.mkdir();merged=output/'dataset';merged.mkdir()
 proof={'schema':'hsr-recovery-augmentation-v1','preparationSourceCommit':source_commit,'preparationSourceHashes':code_hashes,'lerobotRevision':LEROBOT_REVISION,'parentManifestSha256':sha(original/'dataset-manifest.json'),'parentDatasetIdentity':manifest_identity(parent_manifest),'parentNormalizationIdentity':manifest_identity(parent_norm),'catalogSha256':sha(catalog_path),'artifacts':artifacts,'addedRows':rows,'attempts':attempts,'datasetReady':False,'partitions':{}}
 for side in ['train','validation']:
  selected=[(record,path)for record,path,row in zip(records,paths,rows)if row['status']=='accepted'and row['partition']==side]
  if selected:
   write_partition(records,paths,recovery/side,training_normalization(records),side)
   aggregate_datasets([f'local/original-{side}',f'local/recovery-{side}'],f'local/augmented-{side}',roots=[original/side,recovery/side],aggr_root=merged/side,concatenate_data=False,concatenate_videos=False)
  else:shutil.copytree(original/side,merged/side)
  proof['partitions'][side]=verify_merged_frames(original/side,selected,merged/side)
  episode_map=json.loads((original/side/'hsr-episode-map.json').read_text())
  episode_map.extend({'episode_index':i,'episodeId':r['episode']['collection']['episodeId'],'sourceSha256':r['sourceSha256'],'seed':r['episode']['seed']}for i,(r,_)in enumerate(selected,start=len(episode_map)))
  (merged/side/'hsr-episode-map.json').write_text(json.dumps(episode_map,indent=2)+'\n')
 stats=json.loads((merged/'train/meta/stats.json').read_text())
 for key,native_key in [('state','observation.state'),('action','action')]:stats[native_key].update(mean=normalization[key]['mean'],std=normalization[key]['std'])
 for side in ['train','validation']:(merged/side/'meta/stats.json').write_text(json.dumps(stats,indent=2)+'\n')
 (merged/'normalization.json').write_text(json.dumps(normalization,indent=2)+'\n');(merged/'index.json').write_text(json.dumps(index,indent=2)+'\n')
 verify_dataset(original)
 from prepare_colab import dataset_files
 proof['files']=[{'path':p.relative_to(merged).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}for p in dataset_files(merged)]
 if preparation_hashes(source)!=code_hashes or subprocess.check_output(['git','status','--porcelain'],cwd=source)or subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=source_commit:raise ValueError('Preparation source changed during certification')
 (output/'augmentation-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
 return proof

def finalize(output,source,catalog_path,review_path):
 sys.path[:0]=[str(source/'tools/vla-train')]
 from prepare_colab import reviewed_gate,dataset_files
 from hsr_training import verify_dataset
 if subprocess.check_output(['git','status','--porcelain'],cwd=source):raise ValueError('Save reviewed preparation source first')
 proof_path=output/'augmentation-proof.json';proof=json.loads(proof_path.read_text());root=output/'dataset'
 gate=reviewed_gate(review_path,source);verify_final_review(gate,sha(proof_path),sha(catalog_path))
 verify_preparation_review(proof,gate,preparation_hashes(source))
 if proof['catalogSha256']!=sha(catalog_path):raise ValueError('Recovery catalog changed')
 verify_durable_cohort(json.loads(catalog_path.read_text()),proof['addedRows'],proof['attempts'])
 current=[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}for p in dataset_files(root)]
 if current!=proof['files']:raise ValueError('Verified native augmentation changed')
 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
 manifest={'schema':'hsr-training-bundle-v1','datasetReady':True,'sourceCommit':source_commit,'review':gate,'roundTrip':proof['partitions'],'files':current,
           'recoveryFork':{'schema':'hsr-recovery-data-fork-v1','parentDatasetSha256':proof['parentDatasetIdentity'],'parentNormalizationSha256':proof['parentNormalizationIdentity'],'roundTripVerified':True,'augmentationProofSha256':sha(proof_path),'catalogSha256':sha(catalog_path),'artifacts':proof['artifacts']}}
 path=root/'dataset-manifest.json'
 if path.exists():raise FileExistsError('Preserve prior ready manifest')
 path.write_text(json.dumps(manifest,indent=2)+'\n');verify_dataset(root)
 return {'schema':'hsr-recovery-ready-v1','datasetReady':True,'datasetManifestSha256':sha(path),'datasetIdentity':manifest_identity(manifest),'normalizationIdentity':manifest_identity(json.loads((root/'normalization.json').read_text()))}

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--original',type=Path,required=True);parser.add_argument('--input',type=Path,nargs='+',required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--catalog',type=Path,required=True);parser.add_argument('--final-review',type=Path)
 args=parser.parse_args()
 if args.final_review:print(json.dumps(finalize(args.output,args.source.resolve(),args.catalog,args.final_review)))
 else:build(args.original,args.input,args.output,args.source.resolve(),args.catalog)
