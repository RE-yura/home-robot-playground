"""Current-state-relative HSR arm goals; absolute gripper, one anchor per chunk."""
import copy,hashlib,json
import numpy as np
import torch

def _reference(actions,state):
    if actions.ndim!=3 or actions.shape[1:]!=(16,6) or state.shape!=(actions.shape[0],6):
        raise ValueError('Aligned Bx16x6 goals and Bx6 physical measured state required')
    return state.to(device=actions.device,dtype=actions.dtype)[:,None,:5]

def encode_arm_actions(actions,state):
    return torch.cat((actions[:,:,:5]-_reference(actions,state),actions[:,:,5:]),dim=-1)

def decode_arm_actions(actions,state):
    return torch.cat((actions[:,:,:5]+_reference(actions,state),actions[:,:,5:]),dim=-1)

def relative_arm_normalization(root,absolute):
    """All valid future targets, within episodes even when Parquet packs them."""
    import pyarrow.parquet as pq
    episodes={};count=0
    for path in sorted((root/'train/data').rglob('*.parquet')):
        table=pq.read_table(path,columns=['episode_index','frame_index','observation.state','action'])
        for row in table.to_pylist():
            episodes.setdefault(row['episode_index'],[]).append((row['frame_index'],row['observation.state'],row['action']));count+=1
    if count!=absolute['sampleCount'] or set(episodes)!=set(range(len(absolute['episodeIds']))):
        raise ValueError('Accepted training episode/frame identities required')
    parts=[]
    for _,records in sorted(episodes.items()):
        ordered=sorted(records)
        if [x[0]for x in ordered]!=list(range(len(ordered))):raise ValueError('Contiguous unique episode frames required')
        state=np.asarray([x[1]for x in ordered],dtype=np.float32);actions=np.asarray([x[2]for x in ordered],dtype=np.float32)
        if state.shape!=actions.shape or state.shape!=(len(ordered),6) or not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError('Finite six-channel measured states/goals required')
        for offset in range(min(16,len(ordered))):
            goals=actions[offset:].copy();goals[:,:5]-=state[:len(goals),:5];parts.append(goals.astype(np.float64))
    values=np.concatenate(parts);mean=values.mean(axis=0);std=np.maximum(values.std(axis=0),1e-6)
    result=copy.deepcopy(absolute);result['action']={'mean':mean.tolist(),'std':std.tolist()}
    for name in ['mean','std']:result['action'][name][5]=absolute['action'][name][5]
    result['relativeArmActions']={'mask':[True]*5+[False],'chunkLength':16,
        'reference':'same physical measured state at prediction start for all16goals',
        'validTargetPairs':len(values),'excludeEpisodeEndPadding':True,
        'statsArithmetic':'Float32 relative targets, Float64 mean/std',
        'gripperNormalization':'exact original accepted-train absolute statistics',
        'parentNormalizationSha256':hashlib.sha256(json.dumps(absolute,sort_keys=True).encode()).hexdigest()}
    return result

def reset_relative_arm_head(policy):
    """Start the five newly relative output rows fresh; retain finger/padded rows."""
    layer=policy.model.action_out_proj
    def digest(weight,bias):
        raw=b''.join(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()for x in [weight,bias])
        return hashlib.sha256(raw).hexdigest()
    arm_before=digest(layer.weight[:5],layer.bias[:5]);retained=digest(layer.weight[5:],layer.bias[5:])
    with torch.no_grad():layer.weight[:5].zero_();layer.bias[:5].zero_()
    return {'schema':'hsr-relative-arm-head-initialization-v1','resetRows':list(range(5)),
        'armBeforeSha256':arm_before,'armAfterSha256':digest(layer.weight[:5],layer.bias[:5]),
        'retainedRowsSha256':retained,'retainedRowsUnchanged':retained==digest(layer.weight[5:],layer.bias[5:]),
        'armRowsZero':bool(torch.count_nonzero(layer.weight[:5])==0 and torch.count_nonzero(layer.bias[:5])==0)}
