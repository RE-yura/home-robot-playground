"""HSR-only SmolVLA adaptation. GPU execution is a separate user-operated gate."""
import argparse,hashlib,json,math,random,shutil
from pathlib import Path
import numpy as np
import torch

POLICY_ID='lerobot/smolvla_base'
POLICY_REVISION='5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6'
POLICY_SHA256='7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb'
PROCESSOR_ID='HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
PROCESSOR_REVISION='7b375e1b73b11138ff12fe22c8f2822d8fe03467'
LEROBOT_REVISION='ca69a2068462a37f7cdcb74180927a2f863d2bf7'
CAMERAS=('observation.images.head','observation.images.hand')
INPUTS=set(CAMERAS)|{'observation.state','task'}
TRAIN_INPUTS=INPUTS|{'action','action_is_pad'}
INSTRUCTIONS=('Put the mug in the basket.','Put the tennis ball in the basket.')
TRAINABLE_PREFIXES=('model.vlm_with_expert.lm_expert.','model.state_proj.','model.action_in_proj.',
                    'model.action_out_proj.','model.action_time_mlp_in.','model.action_time_mlp_out.')
# --train-vision (recipe v6 only): the SigLIP tower and its connector also train, at their own lower learning rate;
# the language model, its embeddings and head stay frozen.
VISION_PREFIXES=('model.vlm_with_expert.vlm.model.vision_model.','model.vlm_with_expert.vlm.model.connector.')
VISION_LEARNING_RATE=1e-5
ARM_ROLL_LOSS_WEIGHT=9
ACTION_OFFSET_STEPS=(0,2)
OFFSET_PARENT_MANIFEST='f289e2a8b06ca26216ca932700df4633272d1988d6e1253893ed5c5b9740468d'

def sha(path):
    with open(path,'rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def training_config(config,processor_path,device='cuda'):
    from lerobot.configs import FeatureType,PolicyFeature,NormalizationMode
    config.input_features={**{k:PolicyFeature(type=FeatureType.VISUAL,shape=(3,240,320))for k in CAMERAS},
                           'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(6,))}
    config.output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(6,))}
    config.chunk_size=16;config.n_action_steps=4;config.n_obs_steps=1
    config.max_state_dim=config.max_action_dim=32
    config.resize_imgs_with_padding=(512,512);config.tokenizer_max_length=48
    config.pad_language_to='max_length';config.num_steps=10
    config.empty_cameras=0;config.adapt_to_pi_aloha=False;config.use_delta_joint_actions_aloha=False
    config.freeze_vision_encoder=True;config.train_expert_only=True;config.train_state_proj=True
    config.normalization_mapping={'VISUAL':NormalizationMode.IDENTITY,'STATE':NormalizationMode.MEAN_STD,'ACTION':NormalizationMode.MEAN_STD}
    config.vlm_model_name=str(processor_path);config.load_vlm_weights=False;config.device=device
    config.validate_features()
    return config

def policy_batch(batch,*,training=True,relative_arm=False):
    if relative_arm and not training:raise ValueError('Relative encoding is for training targets only')
    required=TRAIN_INPUTS if training else INPUTS
    if not required<=batch.keys():raise ValueError('Missing HSR camera/state/task/action boundary')
    selected={k:batch[k]for k in required}
    if selected['observation.state'].shape[-1]!=6:raise ValueError('HSR requires six measured state channels')
    if training and selected['action'].shape[-2:]!=(16,6):raise ValueError('HSR requires16 next-goal actions with six channels')
    if not all(torch.isfinite(selected[k]).all()for k in ('observation.state','action')if k in selected):raise ValueError('Nonfinite HSR features')
    if any(task not in INSTRUCTIONS for task in selected['task']):raise ValueError('Unsupported HSR instruction')
    for key in CAMERAS:
        image=selected[key]
        if image.shape[-3:]!=(3,240,320)or not torch.isfinite(image).all()or image.min()<0 or image.max()>1:
            raise ValueError('Expected actual RGB tensor in [0,1]')
    if training and (selected['action_is_pad'].dtype!=torch.bool or selected['action_is_pad'].shape!=selected['action'].shape[:-1]):
        raise ValueError('Episode-end padding mask required')
    if relative_arm:
        from relative_arm_actions import encode_arm_actions
        selected['action']=encode_arm_actions(selected['action'],selected['observation.state'])
    return selected

def action_deltas(offset_steps=0):
    """Sixteen 100ms goal times; a nonzero offset starts the chunk after the observation."""
    if type(offset_steps)is not int or offset_steps not in ACTION_OFFSET_STEPS:raise ValueError('Action offset must be0 or2 steps')
    return [(offset_steps+i)/10 for i in range(16)]

def hsr_datasets(root,offset_steps=0):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    deltas={'action':action_deltas(offset_steps)}
    return tuple(LeRobotDataset(f'local/hsr-{name}',root=root/name,delta_timestamps=deltas,video_backend='pyav')for name in ('train','validation'))

def mug_release_mask(sample):
    """Offline TRAIN metadata: select each future mug goal during opening."""
    frames=int(sample['frame_index'])+torch.arange(16)
    return ((frames>=200)&(frames<230)&(sample['task']==INSTRUCTIONS[0])).unsqueeze(0)

def training_loss(policy,batch,*,release_mask=None,cartesian_normalization=None,arm_roll_precision=False):
    if arm_roll_precision and (release_mask is not None or cartesian_normalization is not None):
        raise ValueError('Cannot combine arm-roll precision with other loss experiments')
    if cartesian_normalization is not None:
        if release_mask is not None:raise ValueError('Cannot combine Cartesian and release-loss experiments')
        from cartesian_loss import cartesian_training_loss
        return cartesian_training_loss(policy,batch,cartesian_normalization)
    if release_mask is None and not arm_roll_precision:return policy.forward(batch)
    if release_mask is not None and (release_mask.dtype!=torch.bool or release_mask.shape!=batch['action_is_pad'].shape):
        raise ValueError('Boolean release mask must match the next-goal steps')
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS,OBS_LANGUAGE_ATTENTION_MASK
    images,image_masks=policy.prepare_images(batch)
    losses=policy.model.forward(images,image_masks,batch[OBS_LANGUAGE_TOKENS],batch[OBS_LANGUAGE_ATTENTION_MASK],
                                policy.prepare_state(batch),policy.prepare_action(batch),None,None)[...,:6]
    weights=torch.ones_like(losses)
    if arm_roll_precision:weights[:,:,2]=ARM_ROLL_LOSS_WEIGHT
    else:weights[:,:,5]+=release_mask.to(losses.device)
    valid=(~batch['action_is_pad']).unsqueeze(-1)
    loss=(losses*weights*valid).sum()/(valid.sum()*6).clamp_min(1)
    return loss,{'loss':loss.item()}

def freeze_policy(policy,train_vision=False):
    names=[]
    for name,p in policy.named_parameters():
        enabled=name.startswith(TRAINABLE_PREFIXES)or(train_vision and name.startswith(VISION_PREFIXES));p.requires_grad_(enabled)
        if enabled:names.append(name)
    if not any(n.startswith('model.state_proj.')for n in names)or not any(n.startswith(TRAINABLE_PREFIXES[0])for n in names):
        raise ValueError('HSR state projection and action expert must both train')
    policy.model.vlm_with_expert.vlm.eval()
    open_vlm=[n for n,p in policy.named_parameters()if p.requires_grad and n.startswith('model.vlm_with_expert.vlm.')]
    if any(not(train_vision and n.startswith(VISION_PREFIXES))for n in open_vlm):raise ValueError('VLM must remain frozen')
    if train_vision and not any(n.startswith(VISION_PREFIXES[0])for n in open_vlm):raise ValueError('Vision training found no vision tower')
    return names

def verify_dataset(root,*,mug_data_fork=False):
    manifest=json.loads((root/'dataset-manifest.json').read_text())
    if bool(manifest.get('mugDataFork'))!=mug_data_fork:
        raise ValueError('A recollected mug dataset requires --mug-data-fork' if manifest.get('mugDataFork') else '--mug-data-fork requires a dataset manifest with mugDataFork')
    if manifest.get('schema')!='hsr-training-bundle-v1'or not manifest.get('datasetReady'):
        raise ValueError('Certified50-per-instruction dataset and fresh final review required')
    names=[f['path']for f in manifest['files']]
    actual={p.relative_to(root).as_posix()for p in root.rglob('*')if p.is_file()and p.relative_to(root).as_posix()!='dataset-manifest.json'}
    if len(set(names))!=len(names)or set(names)!=actual:raise ValueError('Dataset file set mismatch')
    for item in manifest['files']:
        p=root/item['path']
        if p.resolve().is_relative_to(root.resolve())is False or p.stat().st_size!=item['bytes']or sha(p)!=item['sha256']:
            raise ValueError('Dataset file hash mismatch')
    index=json.loads((root/'index.json').read_text());normalization=json.loads((root/'normalization.json').read_text())
    episodes=index['episodes'];ids=[r['episodeId']for r in episodes]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate dataset episode identity')
    train=[r for r in episodes if r['status']=='accepted'and r['partition']=='train']
    validation=[r for r in episodes if r['status']=='accepted'and r['partition']=='validation']
    if any((r['partition']=='train'and not 1000<=r['seed']<=1999)or(r['partition']=='validation'and not 2000<=r['seed']<=2099)or r['partition']not in('train','validation')for r in episodes):raise ValueError('Calibration/evaluation leakage')
    if {r['seed']for r in train}&{r['seed']for r in validation}:raise ValueError('Seed split leakage')
    for instruction in INSTRUCTIONS:
        if len({r['seed']for r in train if r['instruction']==instruction})<50 or len({r['seed']for r in validation if r['instruction']==instruction})<10:
            raise ValueError('Insufficient distinct successful training/validation seeds')
    if mug_data_fork:
        from mug_data_fork import verify_inherited_normalization
        verify_inherited_normalization(manifest['mugDataFork'],normalization)
    elif (normalization.get('schema')!='hsr-training-normalization-v1'or normalization.get('partition')!='train'or
        normalization.get('status')!='accepted'or normalization['episodeIds']!=[r['episodeId']for r in train]or
        normalization['sourceSha256']!=[r['sourceSha256']for r in train]or normalization['sampleCount']!=sum(r['sampleCount']for r in train)):
        raise ValueError('Training-only normalization provenance mismatch')
    for side in ('state','action'):
        if len(normalization[side]['mean'])!=6 or len(normalization[side]['std'])!=6 or not np.isfinite(normalization[side]['mean']).all()or not np.isfinite(normalization[side]['std']).all()or min(normalization[side]['std'])<=0:
            raise ValueError('Six finite HSR normalization axes required')
    train_stats=json.loads((root/'train/meta/stats.json').read_text());val_stats=json.loads((root/'validation/meta/stats.json').read_text())
    if train_stats!=val_stats:raise ValueError('Validation introduced different statistics')
    for side,key in [('state','observation.state'),('action','action')]:
        if train_stats[key]['mean']!=normalization[side]['mean']or train_stats[key]['std']!=normalization[side]['std']:
            raise ValueError('Native statistics differ from accepted-training manifest')
    return manifest,normalization

def checkpoint_identity(dataset_manifest,normalization,source_commit,accumulate=4,batch_size=1):
    if type(accumulate)is not int or accumulate<1:raise ValueError('Positive gradient accumulation required')
    if type(batch_size)is not int or batch_size<1:raise ValueError('Positive batch size required')
    return {'schema':'hsr-trained-policy-v1','robotModelId':'hsr-household-6dof-v1','chunkLength':16,'executeSamples':4,'sampleHz':10,
            'policy':{'id':POLICY_ID,'revision':POLICY_REVISION,'baseSha256':POLICY_SHA256},
            'processor':{'id':PROCESSOR_ID,'revision':PROCESSOR_REVISION},'lerobotRevision':LEROBOT_REVISION,
            'datasetSha256':hashlib.sha256(json.dumps(dataset_manifest,sort_keys=True).encode()).hexdigest(),
            'normalizationSha256':hashlib.sha256(json.dumps(normalization,sort_keys=True).encode()).hexdigest(),
            'sourceCommit':source_commit,'trainingCodeHashes':{name:sha(Path(__file__).parent/name)for name in ('hsr_training.py','requirements-colab.txt','recovery_training.py')},'policyInputs':sorted(INPUTS),'normalization':normalization,
            'trainingSettings':{'batchSize':batch_size,'gradientAccumulation':accumulate},
            'trainedUpdates':0,'learnedRolloutGate':'unverified','adaptedOnnxParity':'unverified','publicationAllowed':False}

def check_resume(saved,current):
    keys=('robotModelId','chunkLength','executeSamples','sampleHz','policy','processor','lerobotRevision','datasetSha256','normalizationSha256','sourceCommit','trainingCodeHashes','policyInputs','trainingSettings')
    if any(saved.get(k)!=current.get(k)for k in keys):raise ValueError('Checkpoint resume contract changed')
    if type(saved.get('trainedUpdates'))is not int or saved['trainedUpdates']<=0:raise ValueError('Checkpoint has no verified optimizer updates')

# mug-recipe-v6: the delay-8 mug fork with batched forwards, first20x2 and four times v3's samples. Its parent is
# verified with the parent's own batch settings; the recipe's settings are recorded afterwards.
RECIPE_V6={'steps':16000,'save_every':2000,'accumulate':1,'batch_size':8,'early_weight':2,'index_seed':20261027}
def training_seed(seed):
    """The fixed 20261008 unless a mug-fork run names another seed (a second training run of the same data)."""
    return 20261008 if seed is None else seed

def recipe_parent_view(identity):
    return {**identity,'trainingSettings':{**identity['trainingSettings'],'batchSize':1,'gradientAccumulation':4}}

def recipe_indices(weights,*,batch,steps,seed):
    """All training indices up front (row = one optimizer step), drawn with replacement from a dedicated generator so a
    resume and the prefetching loader see the same sequence."""
    generator=torch.Generator().manual_seed(seed)
    return torch.multinomial(weights,batch*steps,replacement=True,generator=generator).view(steps,batch)

def prefetched_batches(rows,load,collate,*,workers=8,ahead=2):
    """collate([load(k) for k in row]) for each row in order; samples load in parallel threads and the next `ahead`
    batches are prepared while the current one trains (threads: the container's 64 MB shared memory cannot hold
    worker-process batches)."""
    from concurrent.futures import ThreadPoolExecutor
    rows=[list(r)for r in rows]
    with ThreadPoolExecutor(workers)as samples,ThreadPoolExecutor(1)as batches:
        build=lambda row:collate(list(samples.map(load,row)))
        pending=[batches.submit(build,row)for row in rows[:ahead]]
        for i in range(len(rows)):
            batch=pending.pop(0).result()
            if i+ahead<len(rows):pending.append(batches.submit(build,rows[i+ahead]))
            yield batch

def grasp_sampling_weights(frame_indices):
    """Training metadata only: retain every accepted frame, emphasize the first6s."""
    if not frame_indices or any(type(i)is not int or i<0 for i in frame_indices):
        raise ValueError('Nonnegative integer training frame indices required')
    return torch.tensor([4. if i<60 else 1. for i in frame_indices],dtype=torch.float64)

def initial_sampling_weights(frame_indices,*,early_weight=32):
    """A controlled initial-fit experiment; all accepted later frames stay eligible."""
    if not frame_indices or any(type(i)is not int or i<0 for i in frame_indices):
        raise ValueError('Nonnegative integer training frame indices required')
    if type(early_weight)is not int or early_weight not in (2,8,32):
        raise ValueError('Initial sampling weight must be2, 8 or32')
    return torch.tensor([float(early_weight)if i<20 else 1. for i in frame_indices],dtype=torch.float64)

def close_lift_sampling_weights(frame_indices):
    """Offline training frames only: preserve initial fit, emphasize close/lift."""
    weights=initial_sampling_weights(frame_indices,early_weight=8)
    for i,frame in enumerate(frame_indices):
        if 40<=frame<90:weights[i]=4.
    return weights

def balanced_grasp_sampling_weights(frame_indices,task_indices,*,ball_task_index):
    """Training-only task balance and ball close-frame exposure; keep every frame."""
    weights=initial_sampling_weights(frame_indices,early_weight=8)
    if len(task_indices)!=len(frame_indices)or any(type(i)is not int or i<0 for i in task_indices):
        raise ValueError('Matching nonnegative integer task indices required')
    if type(ball_task_index)is not int or len(set(task_indices))!=2 or ball_task_index not in task_indices:
        raise ValueError('Two accepted tasks and their exact ball task index required')
    for i,(frame,task)in enumerate(zip(frame_indices,task_indices)):
        if task==ball_task_index and 40<=frame<90:weights[i]=4.
    tasks=torch.tensor(task_indices)
    for task in set(task_indices):
        mask=tasks==task;weights[mask]/=weights[mask].sum()
    return weights

def empty_carry_sampling_weights(frame_indices,task_indices,episode_indices,*,ball_task_index,empty_carry_episode_indices):
    """Training metadata only:25% recovery, align/close x4, unchanged mug probability."""
    weights=initial_sampling_weights(frame_indices,early_weight=8)
    if (len(task_indices)!=len(frame_indices)or len(episode_indices)!=len(frame_indices)or
        any(type(i)is not int or i<0 for i in [*task_indices,*episode_indices])or
        type(ball_task_index)is not int or len(set(task_indices))!=2 or ball_task_index not in task_indices or
        not empty_carry_episode_indices or any(type(i)is not int or i<0 for i in empty_carry_episode_indices)or
        not set(empty_carry_episode_indices)<=set(episode_indices)):
        raise ValueError('Matching accepted training task/episode metadata required')
    empty=torch.tensor([i in empty_carry_episode_indices for i in episode_indices])
    ball=torch.tensor(task_indices)==ball_task_index
    if (empty&~ball).any()or not(ball&~empty).any():
        raise ValueError('Empty carry must be a proper subset of ball training episodes')
    total=weights.sum();empty_mass=total*.25;other_ball_mass=weights[ball].sum()-empty_mass
    if other_ball_mass<=0:raise ValueError('Ball exposure must exceed the25% recovery quota')
    grasp=empty&torch.tensor([80<=frame<130 for frame in frame_indices])
    weights[grasp]*=4
    weights[empty]*=empty_mass/weights[empty].sum()
    weights[ball&~empty]*=other_ball_mass/weights[ball&~empty].sum()
    return weights

def empty_carry_episode_indices(root):
    """Select only accepted/source-matched training episodes of the frozen scene."""
    episodes=json.loads((root/'index.json').read_text())['episodes']
    accepted={v['episodeId']:v for v in episodes if v['status']=='accepted'and v['partition']=='train'}
    mapping=json.loads((root/'train/hsr-episode-map.json').read_text())
    if len(mapping)!=len(accepted)or len({v['episodeId']for v in mapping})!=len(mapping)or len({v['episode_index']for v in mapping})!=len(mapping):
        raise ValueError('One-to-one accepted training episode map required')
    selected=set()
    for item in mapping:
        original=accepted.get(item['episodeId'])
        if (original is None or type(item['episode_index'])is not int or item['episode_index']<0 or
            any(item[k]!=original[k]for k in ['seed','sourceSha256'])):
            raise ValueError('Training episode source identity mismatch')
        if 1600<=item['seed']<=1639:
            if original['instruction']!=INSTRUCTIONS[1]:raise ValueError('Empty carry requires the ball task')
            selected.add(item['episode_index'])
    if len(selected)!=15:raise ValueError('Frozen dataset requires15 accepted empty-carry training episodes')
    return selected

def shallow_reseat_sampling_weights(frame_indices,task_indices,episode_indices,*,ball_task_index,shallow_episode_indices):
    """Offline25% new corrections; retain the original dataset's mug exposure."""
    weights=initial_sampling_weights(frame_indices,early_weight=8)
    if (len(task_indices)!=len(frame_indices)or len(episode_indices)!=len(frame_indices)or
        any(type(i)is not int or i<0 for i in [*task_indices,*episode_indices])or
        type(ball_task_index)is not int or len(set(task_indices))!=2 or ball_task_index not in task_indices or
        not shallow_episode_indices or any(type(i)is not int or i<0 for i in shallow_episode_indices)or not set(shallow_episode_indices)<=set(episode_indices)):
        raise ValueError('Matching accepted shallow training metadata required')
    shallow=torch.tensor([i in shallow_episode_indices for i in episode_indices]);ball=torch.tensor(task_indices)==ball_task_index
    if (shallow&~ball).any()or not(ball&~shallow).any():raise ValueError('Shallow examples must be a proper ball subset')
    mug=~ball;old_ball=ball&~shallow
    mug_mass=weights[mug].sum()/weights[~shallow].sum();ball_mass=1-mug_mass-.25
    if ball_mass<=0:raise ValueError('Parent ball exposure must exceed the shallow quota')
    for mask,mass in [(mug,mug_mass),(old_ball,ball_mass),(shallow,.25)]:weights[mask]*=mass/weights[mask].sum()
    return weights

def shallow_reseat_episode_indices(root):
    """The fixed203 cohort adds13 accepted training episodes after219 parents."""
    rows=json.loads((root/'index.json').read_text())['episodes']
    accepted={v['episodeId']:v for v in rows if v['status']=='accepted'and v['partition']=='train'}
    mapping=json.loads((root/'train/hsr-episode-map.json').read_text())
    if len(accepted)!=232 or len(mapping)!=232 or len({v['episodeId']for v in mapping})!=232 or [v['episode_index']for v in mapping]!=list(range(232)):
        raise ValueError('Exact232 accepted training episode map required')
    selected=set()
    for item in mapping:
        original=accepted.get(item['episodeId'])
        if original is None or any(item[k]!=original[k]for k in ['seed','sourceSha256']):raise ValueError('Shallow training source identity mismatch')
        if 1700<=item['seed']<=1779:
            if original['instruction']!=INSTRUCTIONS[1]:raise ValueError('Shallow correction requires the ball task')
            selected.add(item['episode_index'])
    if selected!=set(range(219,232)):raise ValueError('Fixed203 requires the13 new shallow training episodes')
    return selected

def transport_release_sampling_weights(frame_indices):
    """Offline frame proxy only: extend close/lift exposure through release."""
    weights=close_lift_sampling_weights(frame_indices)
    for i,frame in enumerate(frame_indices):
        if 90<=frame<230:weights[i]=4.
    return weights

def ordinary_approach_sampling_weights(frame_indices,task_indices,episode_indices,*,ball_task_index,shallow_episode_indices,ordinary_episode_indices):
    """Increase ordinary ball20..59 exposure without changing any family mass."""
    weights=shallow_reseat_sampling_weights(frame_indices,task_indices,episode_indices,ball_task_index=ball_task_index,shallow_episode_indices=shallow_episode_indices)
    if (not ordinary_episode_indices or any(type(i)is not int or i<0 for i in ordinary_episode_indices)or
        not set(ordinary_episode_indices)<=set(episode_indices)or set(ordinary_episode_indices)&set(shallow_episode_indices)):
        raise ValueError('Matching ordinary ball training metadata required')
    ordinary=torch.tensor([i in ordinary_episode_indices for i in episode_indices])
    if (ordinary&(torch.tensor(task_indices)!=ball_task_index)).any():raise ValueError('Ordinary examples must be ball episodes')
    window=ordinary&torch.tensor([20<=frame<60 for frame in frame_indices])
    if not window.any():raise ValueError('Ordinary approach window is empty')
    mass=weights[ordinary].sum()
    weights[window]*=8
    weights[ordinary]*=mass/weights[ordinary].sum()
    return weights

def ordinary_ball_episode_indices(root):
    """The50 original ordinary balls, source-matched within the fixed232 train map."""
    shallow_reseat_episode_indices(root)
    accepted={v['episodeId']:v for v in json.loads((root/'index.json').read_text())['episodes']if v['status']=='accepted'and v['partition']=='train'}
    mapping=json.loads((root/'train/hsr-episode-map.json').read_text())
    if any(type(v['episode_index'])is not int for v in mapping):raise ValueError('Integer training episode indices required')
    selected={v['episode_index']for v in mapping if v['episode_index']<100 and accepted[v['episodeId']]['instruction']==INSTRUCTIONS[1]}
    if len(selected)!=50:raise ValueError('Frozen original cohort requires50 ordinary ball training episodes')
    return selected

def sampling_learning_rate(*,low_rate=False,initial_sampling=False,close_lift_sampling=False,transport_release_sampling=False,sampling_from=None,resume=None,recovery_from=None):
    """One controlled refinement; ordinary training keeps its original rate."""
    if close_lift_sampling and (not low_rate or initial_sampling or not(sampling_from or resume)):
        raise ValueError('Close/lift sampling requires its low-rate fork or matching resume')
    if transport_release_sampling and (not low_rate or initial_sampling or close_lift_sampling or not(sampling_from or resume)):
        raise ValueError('Transport/release sampling requires its exclusive low-rate fork or matching resume')
    if low_rate and not((initial_sampling or close_lift_sampling or transport_release_sampling) and (sampling_from or resume or (initial_sampling and recovery_from))):
        raise ValueError('Low rate requires an explicit initial-sampling fork or matching resume')
    return 2e-5 if low_rate else 1e-4

def verify_initializer(root,manifest_sha256,current):
    """An explicitly hash-pinned experiment fork, never an ordinary resume."""
    from colab_recording import verify_checkpoint
    if sha(root/'FILES.json')!=manifest_sha256:raise ValueError('Initialization manifest hash mismatch')
    saved=json.loads((root/'hsr-checkpoint.json').read_text())
    verify_checkpoint(root,saved)
    keys=('robotModelId','chunkLength','executeSamples','sampleHz','policy','processor','lerobotRevision',
          'datasetSha256','normalizationSha256','policyInputs','trainingSettings')
    if any(saved.get(k)!=current.get(k)for k in keys):raise ValueError('Initialization data/policy/settings changed')
    return {'manifestSha256':manifest_sha256,'trainedUpdates':saved['trainedUpdates'],
            'sourceCommit':saved['sourceCommit'],'trainingCodeHashes':saved['trainingCodeHashes']}

def verify_ordinary_approach_initializer(root,manifest_sha256,current):
    """A sampling-only fork of the exact206 checkpoint; retain its normalization."""
    if manifest_sha256!='53b516a01ba0556b2cd1c0bafc80cf96ff7416b5c43f313a9aa17804d52c8d87':
        raise ValueError('Frozen206 final2000 parent manifest required')
    saved=json.loads((root/'hsr-checkpoint.json').read_text())
    sampling={'firstFrames':20,'earlyWeight':8,'laterWeight':1,'shallowProbability':.25,'mugProbability':'unchanged from parent first20x8','shallowTrainingSeeds':[1700,1779],'shallowTrainingEpisodes':13,'parentTrainingEpisodes':219}
    if saved.get('trainedUpdates')!=2000 or saved.get('trainingSettings',{}).get('sampling')!=sampling:
        raise ValueError('Frozen206 shallow parent history mismatch')
    expected={**current,'trainingSettings':{**current['trainingSettings'],'sampling':sampling}}
    return {**verify_initializer(root,manifest_sha256,expected),'parentSampling':sampling}

def verify_relative_arm_initializer(root,manifest_sha256,current,absolute_normalization):
    """Explicit action-normalization fork; retain physical state/finger statistics."""
    old_digest=hashlib.sha256(json.dumps(absolute_normalization,sort_keys=True).encode()).hexdigest()
    relative=current['normalization'];metadata=relative.get('relativeArmActions',{})
    if (metadata.get('mask')!=[True]*5+[False] or metadata.get('parentNormalizationSha256')!=old_digest or
        relative['state']!=absolute_normalization['state'] or
        any(relative['action'][key][5]!=absolute_normalization['action'][key][5]for key in ['mean','std']) or
        hashlib.sha256(json.dumps(relative,sort_keys=True).encode()).hexdigest()!=current['normalizationSha256']):
        raise ValueError('Explicit five-arm relative normalization with unchanged state/gripper required')
    previous={**current,'normalization':absolute_normalization,'normalizationSha256':old_digest}
    return {**verify_ordinary_approach_initializer(root,manifest_sha256,previous),
            'normalizationChange':{'from':old_digest,'to':current['normalizationSha256'],'relativeArmMask':[True]*5+[False]}}

def verify_relative_arm_continuation(root,current):
    """Only the closed223 final2000 may continue, preserving its full state."""
    from colab_recording import verify_checkpoint
    parent='1102010a36a20c46f89e04c0611788b19a31674e08956d900930e118e2e784f1'
    if sha(root/'FILES.json')!=parent:raise ValueError('Frozen223 final2000 continuation parent required')
    saved=json.loads((root/'hsr-checkpoint.json').read_text());verify_checkpoint(root,saved)
    if (saved.get('sourceCommit')!='28736833b6ddd7db9b4fa42d049530e5f5bcd1aa' or
        saved.get('trainingStep')!=2000 or saved.get('trainedUpdates')!=2000 or
        current['normalizationSha256']!='80002373ef17fade8a792d1795109ccb725819d526b8c2666d473fb248b07270' or
        current['sourceCommit']==saved['sourceCommit']):
        raise ValueError('Explicit223 relative continuation identity required')
    old,new=saved['trainingCodeHashes'],current['trainingCodeHashes']
    if set(old)!=set(new) or any(old[k]!=new[k]for k in old if k!='hsr_training.py'):
        raise ValueError('Continuation may change only the declared trainer entry contract')
    check_resume(saved,{**current,'sourceCommit':saved['sourceCommit'],'trainingCodeHashes':old})
    return {'schema':'hsr-relative-arm-continuation-v1','parentManifestSha256':parent,
            'parentSourceCommit':saved['sourceCommit'],'parentTrainingCodeHashes':old,
            'parentUpdates':2000,'targetUpdates':4000,'optimizerState':'restored with all RNG state',
            'headInitialization':'retain trained223 head; copy original initialization proof'}

def verify_empty_carry_initializer(root,manifest_sha256,current):
    """The exact159 parent is a recovery-data fork, without a samplingFork field."""
    if manifest_sha256!='e6ffb40378f4efc9317efb2b8b129c78bf78f8c81626152de100d5f9aa3bb9a4':
        raise ValueError('Frozen159 parent manifest required')
    saved=json.loads((root/'hsr-checkpoint.json').read_text())
    sampling={'firstFrames':20,'earlyWeight':8,'laterWeight':1}
    recovery=saved.get('recoveryFork',{})
    if saved.get('trainedUpdates')!=5000 or saved.get('trainingSettings',{}).get('sampling')!=sampling or recovery.get('schema')!='hsr-recovery-initialization-v1' or recovery.get('optimizerState')!='fresh':
        raise ValueError('Frozen159 recovery parent history mismatch')
    expected={**current,'trainingSettings':{**current['trainingSettings'],'sampling':sampling}}
    lineage=verify_initializer(root,manifest_sha256,expected)
    return {**lineage,'optimizerState':'fresh','parentRecoveryFork':recovery}

def verify_sampling_initializer(root,manifest_sha256,current):
    """Only an explicit sampling fork may change a recognized parent's sampler."""
    saved=json.loads((root/'hsr-checkpoint.json').read_text())
    sampling=saved.get('trainingSettings',{}).get('sampling')if isinstance(saved.get('trainingSettings'),dict)else None
    if sampling is None:return verify_initializer(root,manifest_sha256,current)
    parent=saved.get('samplingFork',{})
    profiles={'first20x32':{'firstFrames':20,'earlyWeight':32,'laterWeight':1},
              'first20x8':{'firstFrames':20,'earlyWeight':8,'laterWeight':1},
              'first20x8-close40to90x4':{'firstFrames':20,'earlyWeight':8,'closeLiftFrames':[40,90],'closeLiftWeight':4,'laterWeight':1}}
    if parent.get('profile')not in profiles or sampling!=profiles[parent['profile']]:
        raise ValueError('Unrecognized parent sampling profile')
    # All other source-independent data/policy/optimizer settings remain strict.
    expected={**current,'trainingSettings':{**current['trainingSettings'],'sampling':sampling}}
    lineage=verify_initializer(root,manifest_sha256,expected)
    lineage['parentSamplingFork']=parent
    if 'parentRecoveryFork'in parent:lineage['parentRecoveryFork']=parent['parentRecoveryFork']
    return lineage

def training_state(optimizer,scaler,generator,step):
    return {'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'sampler':generator.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'python':random.getstate(),'numpy':np.random.get_state(),'step':step}

def restore_training_state(state,optimizer,scaler,generator):
    optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);generator.set_state(state['sampler'])
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda']);random.setstate(state['python']);np.random.set_state(state['numpy'])
    return state['step']

def finish_optimizer_step(optimizer,scaler,parameters):
    parameters=list(parameters)
    scaler.unscale_(optimizer)
    finite=all(torch.isfinite(p.grad).all()for p in parameters if p.grad is not None)
    if finite:
        torch.nn.utils.clip_grad_norm_(parameters,10,error_if_nonfinite=True)
    elif not scaler.is_enabled():
        raise FloatingPointError('Nonfinite training gradient without loss scaling')
    # Enabled GradScaler has already recorded overflow during unscale_. Do not
    # clip those gradients: step skips the update and update lowers the scale.
    previous_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
    return scaler.get_scale()>=previous_scale

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True);parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--save-every',type=int,default=250);parser.add_argument('--accumulate',type=int,default=4)
    start=parser.add_mutually_exclusive_group()
    start.add_argument('--resume',type=Path);start.add_argument('--initialize-from',type=Path);start.add_argument('--recovery-from',type=Path);start.add_argument('--sampling-from',type=Path)
    parser.add_argument('--recovery-control',action='store_true')
    parser.add_argument('--initialize-manifest-sha256');parser.add_argument('--grasp-sampling',action='store_true')
    parser.add_argument('--initial-sampling',action='store_true')
    parser.add_argument('--close-lift-sampling',action='store_true')
    parser.add_argument('--transport-release-sampling',action='store_true')
    parser.add_argument('--initial-sampling-weight',type=int,choices=(2,8,32))
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--mug-recipe-v6',action='store_true')
    parser.add_argument('--train-vision',action='store_true')
    parser.add_argument('--prefix-noise',type=float,choices=(0.03,),default=0.)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--sampling-low-rate',action='store_true')
    parser.add_argument('--mug-release-loss',action='store_true')
    parser.add_argument('--cartesian-loss',action='store_true')
    parser.add_argument('--balanced-grasp-sampling',action='store_true')
    parser.add_argument('--balanced-gripper-loss',action='store_true')
    parser.add_argument('--empty-carry-sampling',action='store_true')
    parser.add_argument('--shallow-reseat-sampling',action='store_true')
    parser.add_argument('--ordinary-approach-sampling',action='store_true')
    parser.add_argument('--relative-arm-actions',action='store_true')
    parser.add_argument('--relative-arm-continuation',action='store_true')
    parser.add_argument('--arm-roll-precision-loss',action='store_true')
    parser.add_argument('--action-offset-steps',type=int,choices=ACTION_OFFSET_STEPS,default=0)
    parser.add_argument('--rtc-training-max-delay',type=int,choices=(0,8,10),default=0)
    parser.add_argument('--mug-data-fork',action='store_true')
    args=parser.parse_args()
    import re
    if not re.fullmatch('[0-9a-f]{40}',args.source_commit):raise ValueError('Saved training source commit required')
    if args.seed is not None and(not args.mug_data_fork or args.seed<0):
        raise ValueError('A training seed belongs to the mug fork and must be nonnegative')
    if args.prefix_noise and not(args.rtc_training_max_delay==8 and args.mug_data_fork):
        raise ValueError('Prefix noise belongs to the delay-8 mug fork')
    if (args.batch_size!=1 or args.initial_sampling_weight==2 or args.train_vision)and not args.mug_recipe_v6:
        raise ValueError('Batch size and initial weight 2 belong to the explicit mug recipe v6')
    if args.mug_recipe_v6:
        r=RECIPE_V6
        if not(args.rtc_training_max_delay==8 and args.mug_data_fork and args.sampling_from and args.initial_sampling and args.sampling_low_rate and
               args.initial_sampling_weight==r['early_weight'] and (args.steps,args.save_every,args.accumulate,args.batch_size)==(r['steps'],r['save_every'],r['accumulate'],r['batch_size'])):
            raise ValueError('Mug recipe v6 requires the f289 delay-8 mug fork, first20x2, 16000 steps/save2000/accumulate1/batch8')
    if args.action_offset_steps or args.rtc_training_max_delay:
        others=[args.action_offset_steps and args.rtc_training_max_delay,args.grasp_sampling,args.close_lift_sampling,args.transport_release_sampling,args.mug_release_loss,args.cartesian_loss,
                args.balanced_grasp_sampling,args.balanced_gripper_loss,args.empty_carry_sampling,args.shallow_reseat_sampling,
                args.ordinary_approach_sampling,args.relative_arm_actions,args.relative_arm_continuation,args.arm_roll_precision_loss,
                args.recovery_control,args.recovery_from,args.initialize_from]
        budget=(RECIPE_V6['steps'],RECIPE_V6['save_every'],RECIPE_V6['accumulate'])if args.mug_recipe_v6 else(8000,2000,4)if args.rtc_training_max_delay==8 else(2000,500,4)
        if (any(others)or not(args.initial_sampling and args.initial_sampling_weight==(RECIPE_V6['early_weight']if args.mug_recipe_v6 else 8)and args.sampling_low_rate)or
            not(args.sampling_from or args.resume)or (args.steps,args.save_every,args.accumulate)!=budget):
            # Delays 0-8 (the range inference can use with 16-goal chunks) get four times the updates of the first 0-10 run.
            raise ValueError('Action offset/prefix conditioning requires its exclusive first20x8 low-rate fork to2000/save500/accumulate4 (max delay 8: 8000/save2000/accumulate4) or matching resume')
        if args.sampling_from and args.initialize_manifest_sha256!=OFFSET_PARENT_MANIFEST:
            raise ValueError('Action offset/prefix conditioning must start from the frozen f289 parent')
    if args.initial_sampling_weight is not None and not args.initial_sampling:
        raise ValueError('Initial sampling weight requires its explicit profile')
    early_weight=args.initial_sampling_weight or 32
    if args.initial_sampling and (args.grasp_sampling or not(args.sampling_from or args.resume or args.recovery_from)):
        raise ValueError('Initial sampling requires its explicit sampling fork or matching resume')
    if args.close_lift_sampling and (args.grasp_sampling or args.initial_sampling or args.initial_sampling_weight is not None):
        raise ValueError('Close/lift sampling cannot mix sampling profiles')
    if args.transport_release_sampling and (args.grasp_sampling or args.initial_sampling or args.close_lift_sampling or args.initial_sampling_weight is not None):
        raise ValueError('Transport/release sampling cannot mix sampling profiles')
    if args.mug_release_loss and not(args.close_lift_sampling and args.sampling_low_rate and (args.sampling_from or args.resume)):
        raise ValueError('Mug release loss requires an explicit low-rate close/lift fork or matching resume')
    if args.cartesian_loss and not(args.initial_sampling and args.initial_sampling_weight==8 and args.sampling_low_rate and (args.sampling_from or args.resume) and not args.mug_release_loss):
        raise ValueError('Cartesian loss requires its first20x8 low-rate fork or matching resume')
    if args.cartesian_loss and args.sampling_from and args.initialize_manifest_sha256!='f289e2a8b06ca26216ca932700df4633272d1988d6e1253893ed5c5b9740468d':
        raise ValueError('Cartesian loss must start from the frozen f289 parent')
    if args.cartesian_loss and (args.steps>500 or args.save_every!=25 or (args.sampling_from and args.steps!=50)):
        raise ValueError('Cartesian budget requires a50-attempt pilot/save25 or matching resume up to500')
    if args.balanced_grasp_sampling:
        if not(args.initial_sampling and args.initial_sampling_weight==8 and args.sampling_low_rate and (args.sampling_from or args.resume))or args.cartesian_loss or args.mug_release_loss:
            raise ValueError('Balanced grasp sampling requires its exclusive first20x8 low-rate fork or matching resume')
        if args.sampling_from and args.initialize_manifest_sha256!='f289e2a8b06ca26216ca932700df4633272d1988d6e1253893ed5c5b9740468d':
            raise ValueError('Balanced grasp sampling must start from the frozen f289 parent')
        if args.steps>500 or args.save_every!=25 or (args.sampling_from and args.steps!=50):
            raise ValueError('Balanced grasp budget requires a50-attempt pilot/save25 or matching resume up to500')
    if args.balanced_gripper_loss and not args.balanced_grasp_sampling:
        raise ValueError('Balanced gripper loss requires its explicit balanced grasp profile')
    if args.arm_roll_precision_loss:
        if not args.ordinary_approach_sampling or args.relative_arm_actions or args.relative_arm_continuation or args.steps!=1000 or args.save_every!=500 or args.accumulate!=4:
            raise ValueError('Arm-roll precision requires absolute ordinary206 fork to1000/save500/accumulate4')
    if args.relative_arm_actions and not args.ordinary_approach_sampling:
        raise ValueError('Relative arm actions require the explicit ordinary-approach sampling fork')
    if args.relative_arm_continuation:
        if not(args.relative_arm_actions and args.ordinary_approach_sampling and args.resume)or any([args.sampling_from,args.recovery_from,args.initialize_from])or args.steps!=4000 or args.save_every!=1000 or args.accumulate!=4:
            raise ValueError('Relative continuation requires exact223 resume to4000/save1000/accumulate4')
        if args.output.exists():raise FileExistsError('Use a fresh relative continuation output')
    if args.ordinary_approach_sampling:
        if not(args.initial_sampling and args.initial_sampling_weight==8 and args.sampling_low_rate and (args.sampling_from or args.resume))or any([args.grasp_sampling,args.close_lift_sampling,args.transport_release_sampling,args.balanced_grasp_sampling,args.balanced_gripper_loss,args.empty_carry_sampling,args.shallow_reseat_sampling,args.cartesian_loss,args.mug_release_loss,args.recovery_control]):
            raise ValueError('Ordinary approach requires its exclusive shallow-data low-rate sampling fork or matching resume')
        if args.sampling_from and args.initialize_manifest_sha256!='53b516a01ba0556b2cd1c0bafc80cf96ff7416b5c43f313a9aa17804d52c8d87':
            raise ValueError('Ordinary approach must start from frozen206 final2000')
        target,save_every=(4000,1000)if args.relative_arm_continuation else(2000,1000)if args.relative_arm_actions else(1000,500)
        if args.steps>target or args.save_every!=save_every or (args.sampling_from and args.steps!=target):
            raise ValueError(f'Ordinary approach requires fixed{target}updates/save{save_every} or matching resume')
    if args.shallow_reseat_sampling:
        if not(args.initial_sampling and args.initial_sampling_weight==8 and args.sampling_low_rate and (args.recovery_from or args.resume))or any([args.grasp_sampling,args.close_lift_sampling,args.transport_release_sampling,args.balanced_grasp_sampling,args.balanced_gripper_loss,args.empty_carry_sampling,args.cartesian_loss,args.mug_release_loss,args.recovery_control]):
            raise ValueError('Shallow sampling requires its exclusive first20x8 low-rate recovery fork or matching resume')
        if args.recovery_from and args.initialize_manifest_sha256!='e6ffb40378f4efc9317efb2b8b129c78bf78f8c81626152de100d5f9aa3bb9a4':
            raise ValueError('Shallow sampling must start from the frozen159 final5000 parent')
        if args.steps>2000 or args.save_every!=500 or (args.recovery_from and args.steps!=2000):
            raise ValueError('Shallow sampling requires fixed2000updates/save500 or matching resume')
    if args.empty_carry_sampling:
        if not(args.initial_sampling and args.initial_sampling_weight==8 and args.sampling_low_rate and (args.sampling_from or args.resume))or any([args.balanced_grasp_sampling,args.balanced_gripper_loss,args.cartesian_loss,args.mug_release_loss]):
            raise ValueError('Empty-carry sampling requires its exclusive first20x8 low-rate fork or matching resume')
        if args.sampling_from and args.initialize_manifest_sha256!='e6ffb40378f4efc9317efb2b8b129c78bf78f8c81626152de100d5f9aa3bb9a4':
            raise ValueError('Empty-carry sampling must start from the frozen159 final5000 parent')
        if args.steps>500 or args.save_every!=250 or (args.sampling_from and args.steps!=500):
            raise ValueError('Empty-carry sampling requires a500-attempt pilot/save250 or matching resume up to500')
    learning_rate=sampling_learning_rate(low_rate=args.sampling_low_rate,initial_sampling=args.initial_sampling,
                                        close_lift_sampling=args.close_lift_sampling,
                                        transport_release_sampling=args.transport_release_sampling,
                                        sampling_from=args.sampling_from,resume=args.resume,recovery_from=args.recovery_from)
    if args.mug_data_fork and (args.rtc_training_max_delay!=8 or not args.sampling_from):
        raise ValueError('Mug data fork requires the delay-8 prefix fork from f289')
    if args.recovery_control and not args.recovery_from:raise ValueError('Recovery control requires a weights-only recovery parent')
    if args.sampling_from:
        if args.grasp_sampling or not re.fullmatch('[0-9a-f]{64}',args.initialize_manifest_sha256 or ''):
            raise ValueError('Sampling fork requires exact parent hash and its uniform/initial profile')
        if args.output.exists():raise FileExistsError('Use a fresh sampling experiment output')
    elif args.recovery_from:
        if args.grasp_sampling:raise ValueError('Recovery comparison cannot change sampling settings')
        if not re.fullmatch('[0-9a-f]{64}',args.initialize_manifest_sha256 or ''):raise ValueError('Recovery requires exact parent manifest hash')
        if args.output.exists():raise FileExistsError('Use a fresh recovery experiment output')
    elif args.initialize_from:
        if not args.grasp_sampling or not re.fullmatch('[0-9a-f]{64}',args.initialize_manifest_sha256 or ''):
            raise ValueError('Experiment initialization requires grasp sampling and exact parent manifest hash')
        if args.output.exists():raise FileExistsError('Use a fresh experiment output; preserve previous work')
    elif args.initialize_manifest_sha256:raise ValueError('Parent hash requires explicit initialization')
    if not torch.cuda.is_available():raise RuntimeError('Select the user-operated Colab GPU runtime before training')
    if args.steps<1 or args.save_every<1 or args.accumulate<1:raise ValueError('Positive training budget required')
    from importlib.metadata import distribution
    direct=json.loads(distribution('lerobot').read_text('direct_url.json'))
    if direct['vcs_info']['commit_id']!=LEROBOT_REVISION:raise ValueError('Pinned LeRobot required')
    manifest,normalization=verify_dataset(args.dataset,mug_data_fork=args.mug_data_fork)
    absolute_normalization=normalization
    if args.empty_carry_sampling and sha(args.dataset/'dataset-manifest.json')!='47d55f0d44e7ebf669086f76dc72246af40b224fa24dc94b32b7cb8f44f42ad2':
        raise ValueError('Empty-carry sampling requires the frozen251-episode dataset')
    if args.ordinary_approach_sampling and sha(args.dataset/'dataset-manifest.json')!='d3f9125d55d231aaf492d171efcc7683216101a084e8fba0c3ac958b3315fb67':
        raise ValueError('Ordinary approach requires the frozen270-episode dataset')
    if args.relative_arm_actions:
        from relative_arm_actions import relative_arm_normalization
        if hashlib.sha256(json.dumps(normalization,sort_keys=True).encode()).hexdigest()!='fa5d7d2b54e4b9e097d5f72d47586c017e6a5a1970512e1fd639cf9290d1180a':
            raise ValueError('Frozen absolute normalization required before relative fork')
        normalization=relative_arm_normalization(args.dataset,normalization)
        if hashlib.sha256(json.dumps(normalization,sort_keys=True).encode()).hexdigest()!='80002373ef17fade8a792d1795109ccb725819d526b8c2666d473fb248b07270':
            raise ValueError('Frozen train-only relative normalization mismatch')
    identity=checkpoint_identity(manifest,normalization,args.source_commit,args.accumulate,args.batch_size)
    # This experiment retains the verified low-rate parent's optimizer settings.
    if args.close_lift_sampling or args.transport_release_sampling or args.cartesian_loss:identity['trainingSettings']['learningRate']=learning_rate
    if args.sampling_low_rate and args.sampling_from:
        parent_settings=json.loads((args.sampling_from/'hsr-checkpoint.json').read_text()).get('trainingSettings',{})
        if 'learningRate'in parent_settings:identity['trainingSettings']['learningRate']=learning_rate
    if args.sampling_from:
        initializer=verify_ordinary_approach_initializer if args.ordinary_approach_sampling else verify_empty_carry_initializer if args.empty_carry_sampling else verify_sampling_initializer
        # A mug data fork compares the parent with the dataset it was trained on; the recollected one is recorded separately.
        parent_view={**identity,'datasetSha256':manifest['mugDataFork']['parentDatasetSha256']}if args.mug_data_fork else identity
        if args.mug_recipe_v6:parent_view=recipe_parent_view(parent_view)
        identity['samplingFork']=verify_relative_arm_initializer(args.sampling_from,args.initialize_manifest_sha256,identity,absolute_normalization)if args.relative_arm_actions else initializer(args.sampling_from,args.initialize_manifest_sha256,parent_view)
        if args.mug_data_fork:
            identity['mugDataFork']=manifest['mugDataFork'];identity['trainingCodeHashes']['mug_data_fork.py']=sha(Path(__file__).parent/'mug_data_fork.py')
        parent=json.loads((args.sampling_from/'hsr-checkpoint.json').read_text())
        if 'recoveryFork'in parent:identity['samplingFork']['parentRecoveryFork']=parent['recoveryFork']
        identity['samplingFork']['profile']='first20x8-close40to90x4-release90to230x4'if args.transport_release_sampling else 'first20x8-close40to90x4'if args.close_lift_sampling else f'first20x{early_weight}'if args.initial_sampling else 'uniform'
        if args.balanced_grasp_sampling:identity['samplingFork']['profile']='first20x8-balanced-ball-close40to90x4'
        if args.empty_carry_sampling:identity['samplingFork']['profile']='first20x8-empty-carry-quarter-grasp80to130x4'
        if args.ordinary_approach_sampling:identity['samplingFork']['profile']='shallow-quarter-ordinary20to60x8'
    if args.initialize_from:identity['fineTune']=verify_initializer(args.initialize_from,args.initialize_manifest_sha256,identity)
    if args.grasp_sampling:identity['trainingSettings']['sampling']={'firstFrames':60,'earlyWeight':4,'laterWeight':1}
    if args.initial_sampling:identity['trainingSettings']['sampling']={'firstFrames':20,'earlyWeight':early_weight,'laterWeight':1}
    if args.balanced_grasp_sampling:
        identity['trainingSettings']['sampling']={'firstFrames':20,'earlyWeight':8,'ballCloseFrames':[40,90],'ballCloseWeight':4,'laterWeight':1,'taskProbability':.5,'ballInstruction':INSTRUCTIONS[1]}
    if args.balanced_gripper_loss:
        identity['trainingSettings']['balancedGripperLoss']={'gripperWeight':2,'otherWeight':1,'futureSteps':'all nonpad','denominator':'valid steps times six channels'}
    if args.arm_roll_precision_loss:
        identity['trainingSettings']['armRollPrecisionLoss']={'axisWeights':[1,1,ARM_ROLL_LOSS_WEIGHT,1,1,1],'futureSteps':'all nonpad, both instructions','denominator':'valid steps times six channels','validationLoss':'unchanged native unweighted loss'}
    if args.empty_carry_sampling:
        identity['trainingSettings']['sampling']={'firstFrames':20,'earlyWeight':8,'laterWeight':1,'emptyCarryProbability':.25,'mugProbability':'unchanged from first20x8','emptyCarryTrainingSeeds':[1600,1639],'emptyCarryTrainingEpisodes':15,'emptyCarryGraspFrames':[80,130],'emptyCarryGraspWeight':4}
    if args.close_lift_sampling:identity['trainingSettings']['sampling']={'firstFrames':20,'earlyWeight':8,'closeLiftFrames':[40,90],'closeLiftWeight':4,'laterWeight':1}
    if args.transport_release_sampling:identity['trainingSettings']['sampling']={'firstFrames':20,'earlyWeight':8,'closeLiftFrames':[40,90],'closeLiftWeight':4,'transportReleaseFrames':[90,230],'transportReleaseWeight':4,'laterWeight':1}
    if args.sampling_low_rate:identity['trainingSettings']['learningRate']=learning_rate
    # Recorded after the unchanged parent settings were verified; a resume must match it.
    if args.action_offset_steps:identity['trainingSettings']['actionOffsetSteps']=args.action_offset_steps
    if args.rtc_training_max_delay:identity['trainingSettings']['rtcTrainingMaxDelay']=args.rtc_training_max_delay
    if args.seed is not None:identity['trainingSettings']['seed']=args.seed
    if args.prefix_noise:identity['trainingSettings']['prefixNoise']={'sigma':args.prefix_noise,'kind':'one Gaussian offset per sample and real axis over the committed goals, normalized units'}
    if args.mug_recipe_v6:identity['trainingSettings']['recipe']={'name':'mug-recipe-v6','indexSeed':RECIPE_V6['index_seed'],'prefetch':'8 loader threads, 2 batches ahead, fixed index order',
        **({'trainVision':{'prefixes':list(VISION_PREFIXES),'learningRate':VISION_LEARNING_RATE}}if args.train_vision else{})}
    if args.recovery_from:
        from recovery_training import verify_recovery_initializer
        identity['recoveryFork']=verify_recovery_initializer(args.recovery_from,args.initialize_manifest_sha256,identity,manifest,control=args.recovery_control)
    if args.shallow_reseat_sampling:
        # Verify unchanged parent settings above, then record this explicit
        # sampling change. Changed train-only normalization is already a fork.
        previous=identity['trainingSettings']['sampling']
        identity['trainingSettings']['sampling']={'firstFrames':20,'earlyWeight':8,'laterWeight':1,'shallowProbability':.25,'mugProbability':'unchanged from parent first20x8','shallowTrainingSeeds':[1700,1779],'shallowTrainingEpisodes':13,'parentTrainingEpisodes':219}
        if args.recovery_from:identity['recoveryFork']['samplingChange']={'from':previous,'to':identity['trainingSettings']['sampling']}
    if args.ordinary_approach_sampling:
        identity['trainingSettings']['sampling']={'firstFrames':20,'earlyWeight':8,'laterWeight':1,'shallowProbability':.25,'mugProbability':'unchanged from parent first20x8','shallowTrainingSeeds':[1700,1779],'shallowTrainingEpisodes':13,'parentTrainingEpisodes':219,'ordinaryBallApproachFrames':[20,60],'ordinaryBallApproachWeight':8,'ordinaryBallTrainingEpisodes':50,'ordinaryBallMass':'unchanged from shallow parent','allOtherFrameWeights':'unchanged from shallow parent'}
    if args.relative_arm_actions:
        identity['trainingCodeHashes']['relative_arm_actions.py']=sha(Path(__file__).parent/'relative_arm_actions.py')
        identity['trainingSettings']['relativeArmActions']={'mask':[True]*5+[False],'reference':'one physical measured state per16-goal chunk','headInitialization':'zero first5 action_out_proj rows once on fresh fork; retain gripper/padded rows'}
    if args.relative_arm_continuation:
        identity['relativeContinuation']=verify_relative_arm_continuation(args.resume,identity)
    if args.mug_release_loss:
        identity['trainingSettings']['mugReleaseLoss']={'instruction':INSTRUCTIONS[0],'futureFrames':[200,230],
                                                       'gripperWeight':2,'otherWeight':1,'denominator':'valid steps times six channels'}
    if args.cartesian_loss:
        from cartesian_loss import POSITION_WEIGHT,POSITION_SCALE_METRES
        identity['trainingSettings']['cartesianLoss']={'weight':POSITION_WEIGHT,'positionScaleMetres':POSITION_SCALE_METRES,
            'target':'FK of recorded teacher joint goals','cleanAction':'x_t - t*v_t, NOISE_AT_ONE','padding':'nonpad steps only'}
        for name in ('cartesian_loss.py','colab_recording.py'):
            identity['trainingCodeHashes'][name]=sha(Path(__file__).parent/name)
    args.output.mkdir(parents=True,exist_ok=True)
    from huggingface_hub import snapshot_download
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from torch.utils.data import default_collate
    from safetensors import safe_open
    processor=snapshot_download(PROCESSOR_ID,revision=PROCESSOR_REVISION,allow_patterns=['*.json','*.txt','*.jinja'])
    base=snapshot_download(POLICY_ID,revision=POLICY_REVISION,allow_patterns=['config.json','model.safetensors'])
    if sha(Path(base)/'model.safetensors')!=POLICY_SHA256:raise ValueError('Pinned canonical base weight hash mismatch')
    checkpoint=args.resume or args.initialize_from or args.recovery_from or args.sampling_from
    load=checkpoint/'policy'if checkpoint else Path(base)
    config=training_config(PreTrainedConfig.from_pretrained(load),processor)
    policy=SmolVLAPolicy.from_pretrained(load,config=config)
    with safe_open(str(load/'model.safetensors'),framework='pt')as weights:
        for name,p in policy.state_dict().items():
            if name not in weights.keys()or weights.get_tensor(name).shape!=p.shape:raise ValueError('Missing learned core weight: '+name)
            if args.arm_roll_precision_loss and not torch.equal(p,weights.get_tensor(name).to(device=p.device,dtype=p.dtype)):
                raise ValueError('Loaded parent weight changed: '+name)
    if args.arm_roll_precision_loss:
        if len(policy.state_dict())!=500:raise ValueError('All500 loaded parent tensors required')
        print('ARM_ROLL_PARENT_LOADED '+json.dumps({'manifestSha256':sha(checkpoint/'FILES.json'),'modelSha256':sha(load/'model.safetensors'),'matchedTensors':500,'comparison':'bit-exact after native loaded dtype conversion','headReset':False}),flush=True)
    if args.relative_arm_actions:
        from relative_arm_actions import reset_relative_arm_head
        relative_initialization=args.output/'relative-arm-initialization.json'
        if args.resume:proof=(args.resume/'relative-arm-initialization.json').read_bytes()
        else:
            proof=reset_relative_arm_head(policy)
            if not(proof['armRowsZero']and proof['retainedRowsUnchanged']):raise ValueError('Relative head initialization failed')
            proof['parentManifestSha256']=args.initialize_manifest_sha256
            proof=(json.dumps(proof,indent=2)+'\n').encode()
        if relative_initialization.exists():
            if relative_initialization.read_bytes()!=proof:raise ValueError('Preserve original relative initialization record')
        else:relative_initialization.write_bytes(proof)
    train_names=freeze_policy(policy,train_vision=args.train_vision)
    train,validation=hsr_datasets(args.dataset,args.action_offset_steps)
    allowed=set(CAMERAS)|{'observation.state','action','timestamp','frame_index','episode_index','index','task_index'}
    if set(train.features)!=allowed or set(validation.features)!=allowed or train.fps!=10 or validation.fps!=10:raise ValueError('Native dataset feature/cadence contract changed')
    sampling=transport_release_sampling_weights if args.transport_release_sampling else close_lift_sampling_weights if args.close_lift_sampling else (lambda frames:initial_sampling_weights(frames,early_weight=early_weight))if args.initial_sampling else grasp_sampling_weights if args.grasp_sampling else None
    weights=sampling([int(i)for i in train.hf_dataset.select_columns(['frame_index'])['frame_index']])if sampling else None
    if args.balanced_grasp_sampling:
        columns=train.hf_dataset.select_columns(['frame_index','task_index'])
        weights=balanced_grasp_sampling_weights([int(i)for i in columns['frame_index']],[int(i)for i in columns['task_index']],ball_task_index=train.meta.get_task_index(INSTRUCTIONS[1]))
    if args.empty_carry_sampling:
        columns=train.hf_dataset.select_columns(['frame_index','task_index','episode_index'])
        weights=empty_carry_sampling_weights([int(i)for i in columns['frame_index']],[int(i)for i in columns['task_index']],[int(i)for i in columns['episode_index']],ball_task_index=train.meta.get_task_index(INSTRUCTIONS[1]),empty_carry_episode_indices=empty_carry_episode_indices(args.dataset))
    if args.shallow_reseat_sampling or args.ordinary_approach_sampling:
        columns=train.hf_dataset.select_columns(['frame_index','task_index','episode_index'])
        values=[[int(i)for i in columns[key]]for key in ['frame_index','task_index','episode_index']]
        sampling_args={'ball_task_index':train.meta.get_task_index(INSTRUCTIONS[1]),'shallow_episode_indices':shallow_reseat_episode_indices(args.dataset)}
        weights=ordinary_approach_sampling_weights(*values,**sampling_args,ordinary_episode_indices=ordinary_ball_episode_indices(args.dataset))if args.ordinary_approach_sampling else shallow_reseat_sampling_weights(*values,**sampling_args)
    stats=train.meta.stats
    if args.relative_arm_actions:
        import copy
        stats=copy.deepcopy(stats);stats['action']={key:normalization['action'][key]for key in ['mean','std']}
    preprocess,postprocess=make_smolvla_pre_post_processors(config,stats)
    vision=[p for n,p in policy.named_parameters()if p.requires_grad and n.startswith(VISION_PREFIXES)]
    groups=[{'params':[p for n,p in policy.named_parameters()if p.requires_grad and not n.startswith(VISION_PREFIXES)]}]+([{'params':vision,'lr':VISION_LEARNING_RATE}]if vision else[])
    optimizer=torch.optim.AdamW(groups,lr=learning_rate,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    bf16=torch.cuda.is_bf16_supported();dtype=torch.bfloat16 if bf16 else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=not bf16)
    seed=training_seed(args.seed)
    generator=torch.Generator().manual_seed(seed);torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    step=0;updates=0
    if checkpoint:
        saved=json.loads((checkpoint/'hsr-checkpoint.json').read_text())
        if args.resume:
            if args.relative_arm_continuation:
                if verify_relative_arm_continuation(checkpoint,identity)!=identity['relativeContinuation']:
                    raise ValueError('Relative continuation parent changed before restore')
            else:check_resume(saved,identity)
            if 'fineTune'in saved:identity['fineTune']=saved['fineTune']
            if 'recoveryFork'in saved:identity['recoveryFork']=saved['recoveryFork']
            if 'samplingFork'in saved:identity['samplingFork']=saved['samplingFork']
        files=json.loads((checkpoint/'FILES.json').read_text())
        for f in files:
            if sha(checkpoint/f['path'])!=f['sha256']:raise ValueError('Resume file hash mismatch')
        if not(args.recovery_from or args.sampling_from):
            state=torch.load(checkpoint/'training.pt',map_location='cpu',weights_only=False)
            step=restore_training_state(state,optimizer,scaler,generator);updates=saved['trainedUpdates']
            if step>=args.steps:raise ValueError('Resume already reached this total step budget; choose a larger --steps')
    metrics=args.output/'training-metrics.jsonl'
    if args.relative_arm_continuation:
        # Preserve the original2000 rows so checkpoint history remains contiguous.
        with metrics.open('xb')as stream:stream.write((checkpoint/'training-metrics.jsonl').read_bytes())
    def save():
        if updates<=0:return
        destination=args.output/f'step-{step:06d}';temporary=args.output/f'.step-{step:06d}-writing'
        if destination.exists()or temporary.exists():raise FileExistsError('Preserve previous checkpoint')
        temporary.mkdir();policy.save_pretrained(temporary/'policy');preprocess.save_pretrained(temporary/'policy');postprocess.save_pretrained(temporary/'policy')
        shutil.copytree(processor,temporary/'processor')
        torch.save(training_state(optimizer,scaler,generator,step),temporary/'training.pt')
        (temporary/'hsr-checkpoint.json').write_text(json.dumps(dict(identity,trainedUpdates=updates,trainingStep=step,trainableParameters=train_names),indent=2)+'\n')
        if metrics.exists():shutil.copyfile(metrics,temporary/'training-metrics.jsonl')
        if args.relative_arm_actions:shutil.copyfile(relative_initialization,temporary/'relative-arm-initialization.json')
        files=[{'path':p.relative_to(temporary).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}for p in temporary.rglob('*')if p.is_file()]
        (temporary/'FILES.json').write_text(json.dumps(files,indent=2)+'\n');temporary.rename(destination)
        (args.output/'latest.json').write_text(json.dumps({'checkpoint':destination.name,'trainedUpdates':updates})+'\n')
    recipe_batches=None
    if args.mug_recipe_v6:
        # Fixed index order from a dedicated generator; workers only decode, so a resume continues the same sequence.
        rows=recipe_indices(weights,batch=args.batch_size,steps=args.steps,seed=RECIPE_V6['index_seed'])[step:].tolist()
        recipe_batches=prefetched_batches(rows,lambda k:train[int(k)],default_collate,workers=8,ahead=2)
    try:
        while step<args.steps:
            policy.train();policy.model.vlm_with_expert.vlm.eval();optimizer.zero_grad(set_to_none=True);loss_sum=0;flow_sum=0;cartesian_sum=0
            if recipe_batches is not None:
                import prefix_conditioning
                batch=preprocess(policy_batch(next(recipe_batches)))
                prefix_mask=prefix_conditioning.sample_prefix_mask(args.batch_size,batch['action_is_pad'].shape[1],args.rtc_training_max_delay,generator)
                with torch.autocast('cuda',dtype=dtype):loss,loss_info=prefix_conditioning.prefix_training_loss(policy,batch,prefix_mask,prefix_noise=args.prefix_noise,generator=generator)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss).backward();loss_sum=loss.item()
            for _ in range(0 if recipe_batches is not None else args.accumulate):
                index=int(torch.multinomial(weights,1,generator=generator))if weights is not None else int(torch.randint(len(train),(1,),generator=generator))
                sample=train[index]
                release_mask=mug_release_mask(sample)if args.mug_release_loss else None
                batch=preprocess(policy_batch(default_collate([sample]),relative_arm=args.relative_arm_actions))
                if args.balanced_gripper_loss:release_mask=torch.ones_like(batch['action_is_pad'])
                if args.rtc_training_max_delay:
                    import prefix_conditioning
                    prefix_mask=prefix_conditioning.sample_prefix_mask(1,batch['action_is_pad'].shape[1],args.rtc_training_max_delay,generator)
                    with torch.autocast('cuda',dtype=dtype):loss,loss_info=prefix_conditioning.prefix_training_loss(policy,batch,prefix_mask,prefix_noise=args.prefix_noise,generator=generator)
                else:
                    with torch.autocast('cuda',dtype=dtype):loss,loss_info=training_loss(policy,batch,release_mask=release_mask,cartesian_normalization=normalization['action']if args.cartesian_loss else None,arm_roll_precision=args.arm_roll_precision_loss)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss/args.accumulate).backward();loss_sum+=loss.item()/args.accumulate
                if args.cartesian_loss:
                    flow_sum+=loss_info['flowLoss']/args.accumulate;cartesian_sum+=loss_info['cartesianLoss']/args.accumulate
            updated=finish_optimizer_step(optimizer,scaler,[p for p in policy.parameters()if p.requires_grad]);step+=1
            if updated:updates+=1
            row={'step':step,'trainedUpdates':updates,'trainingLoss':loss_sum,'optimizerUpdate':updated,'lossScale':scaler.get_scale()}
            if args.cartesian_loss:row.update(flowLoss=flow_sum,cartesianLoss=cartesian_sum)
            if step%args.save_every==0 or step==args.steps:
                policy.eval();values=[]
                with torch.no_grad(),torch.autocast('cuda',dtype=dtype):
                    for i in np.linspace(0,len(validation)-1,min(16,len(validation)),dtype=int):
                        loss,_=policy.forward(preprocess(policy_batch(default_collate([validation[int(i)]]),relative_arm=args.relative_arm_actions)))
                        values.append(loss.item())
                if not all(math.isfinite(v)for v in values):raise FloatingPointError('Nonfinite validation loss')
                row['validationLoss']=sum(values)/len(values)
            with open(metrics,'a')as stream:stream.write(json.dumps(row)+'\n')
            if step%10==0:print(json.dumps(row),flush=True)
            if step%args.save_every==0 or step==args.steps:save()
    except BaseException as error:
        (args.output/'failure.json').write_text(json.dumps({'type':type(error).__name__,'error':str(error),'step':step,'trainedUpdates':updates,'publicationAllowed':False})+'\n')
        raise

if __name__=='__main__':main()
