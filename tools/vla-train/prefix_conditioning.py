"""Training-time action-prefix conditioning for the HSR SmolVLA (Black et al. 2025, arXiv 2512.05964).

A random delay d keeps the first d goals of the chunk clean (ground-truth values at model time 0)
and leaves them out of the loss, so the policy learns to continue goals the executor has already
committed. Ported from LeRobot's pi05 `rtc_training_max_delay`; d = 0 is the unchanged objective.
"""
import torch
import torch.nn.functional as F

RTC_TRAINING_MAX_DELAY=10


def sample_prefix_mask(batch,horizon,max_delay,generator):
    """Boolean (batch, horizon) masks whose first d entries are True, d uniform in 0..max_delay."""
    if type(max_delay)is not int or not 0<=max_delay<horizon:raise ValueError('Prefix delay must be an integer in [0, horizon)')
    delays=torch.randint(0,max_delay+1,(batch,),generator=generator)
    return torch.arange(horizon).unsqueeze(0)<delays.unsqueeze(1)


def _time_embedding(model,time,device,dtype):
    from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding
    return create_sinusoidal_pos_embedding(time,model.vlm_with_expert.expert_hidden_size,model.config.min_period,
                                           model.config.max_period,device=device).type(dtype=dtype)


def _fuse_suffix(model,action_emb,time_emb):
    bsize=action_emb.shape[0]
    hidden=model.action_time_mlp_out(F.silu(model.action_time_mlp_in(torch.cat([action_emb,time_emb],dim=2))))
    pad_masks=torch.ones(bsize,hidden.shape[1],dtype=torch.bool,device=action_emb.device)
    att_masks=torch.tensor([1]*model.config.chunk_size,dtype=hidden.dtype,device=hidden.device)[None,:].expand(bsize,-1)
    return hidden,pad_masks,att_masks


def embed_suffix_tokens(model,noisy_actions,token_time):
    """SmolVLA's embed_suffix with one flow-matching time per action token."""
    action_emb=model.action_in_proj(noisy_actions)
    if token_time.ndim==1:token_time=token_time[:,None].expand(action_emb.shape[:2])
    return _fuse_suffix(model,action_emb,_time_embedding(model,token_time,action_emb.device,action_emb.dtype))


def embed_suffix_masked(model,noisy_actions,time,prefix_mask):
    """Same values as embed_suffix_tokens with time 0 on the prefix, written as a select between the embeddings of
    the step time and of 0 so that both stay constants in an exported graph with fixed Euler steps."""
    action_emb=model.action_in_proj(noisy_actions)
    step=_time_embedding(model,time,action_emb.device,action_emb.dtype)[:,None,:].expand_as(action_emb)
    clean=_time_embedding(model,torch.zeros_like(time),action_emb.device,action_emb.dtype)[:,None,:].expand_as(action_emb)
    return _fuse_suffix(model,action_emb,torch.where(prefix_mask[...,None],clean,step))


def prefix_cache(model,images,img_masks,lang_tokens,lang_masks,state):
    """Observation key/value cache, as computed once per chunk by VLAFlowMatching.sample_actions."""
    from lerobot.policies.common.vla_utils import make_att_2d_masks
    embs,pad_masks,att_masks=model.embed_prefix(images,img_masks,lang_tokens,lang_masks,state=state)
    _,past=model.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pad_masks,att_masks),position_ids=torch.cumsum(pad_masks,dim=1)-1,
                                         past_key_values=None,inputs_embeds=[embs,None],use_cache=model.config.use_cache)
    return pad_masks,past


def _denoise(model,cache,suffix):
    from lerobot.policies.common.vla_utils import make_att_2d_masks
    prefix_pad_masks,past=cache;suffix_embs,suffix_pad_masks,suffix_att_masks=suffix
    batch,suffix_len,prefix_len=prefix_pad_masks.shape[0],suffix_pad_masks.shape[1],prefix_pad_masks.shape[1]
    full=torch.cat([prefix_pad_masks[:,None,:].expand(batch,suffix_len,prefix_len),make_att_2d_masks(suffix_pad_masks,suffix_att_masks)],dim=2)
    position_ids=torch.sum(prefix_pad_masks,dim=-1)[:,None]+torch.cumsum(suffix_pad_masks,dim=1)-1
    outputs,_=model.vlm_with_expert.forward(attention_mask=full,position_ids=position_ids,past_key_values=past,
                                            inputs_embeds=[None,suffix_embs],use_cache=model.config.use_cache)
    if past is not None:past.crop(prefix_len)
    return model.action_out_proj(outputs[1][:,-model.config.chunk_size:].to(dtype=torch.float32))


def denoise_tokens(model,cache,x_t,token_time):
    """VLAFlowMatching.denoise_step with per-token time."""
    return _denoise(model,cache,embed_suffix_tokens(model,x_t,token_time))


def sample_with_prefix(model,images,img_masks,lang_tokens,lang_masks,state,noise,prefix,prefix_mask):
    """Euler sampling (noise at t=1) with the masked goals clamped to `prefix` (normalized, padded) at time 0."""
    cache=prefix_cache(model,images,img_masks,lang_tokens,lang_masks,state);keep=prefix_mask[...,None]
    x_t,dt=noise,-1.0/model.config.num_steps
    for step in range(model.config.num_steps):
        time=torch.full((noise.shape[0],),1.0+step*dt,dtype=torch.float32,device=noise.device)
        x_t=torch.where(keep,prefix,x_t)
        x_t=torch.where(keep,prefix,x_t+dt*_denoise(model,cache,embed_suffix_masked(model,x_t,time,prefix_mask)))
    return x_t


def model_inputs(policy,batch):
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS,OBS_LANGUAGE_ATTENTION_MASK
    images,image_masks=policy.prepare_images(batch)
    return images,image_masks,batch[OBS_LANGUAGE_TOKENS],batch[OBS_LANGUAGE_ATTENTION_MASK],policy.prepare_state(batch)


def perturb_prefix(x,prefix_mask,*,sigma,real_dims,generator):
    """Prefix noise (training only): one Gaussian offset per sample and real action axis (normalized units), added to
    every committed goal of that sample. At run time the committed goals are the policy's own earlier predictions,
    whose errors are smooth over a chunk, so the model should not trust them exactly. Postfix goals are untouched."""
    if not sigma:return x
    offset=torch.randn((x.shape[0],1,real_dims),generator=generator).to(device=x.device,dtype=x.dtype)*sigma
    out=x.clone();out[...,:real_dims]=out[...,:real_dims]+prefix_mask.to(x.device).unsqueeze(-1).to(x.dtype)*offset
    return out


def prefix_flow_losses(model,images,img_masks,lang_tokens,lang_masks,state,actions,noise,time,prefix_mask,prefix_noise=0.,generator=None,real_dims=6):
    """VLAFlowMatching.forward with a clean, time-0 action prefix (optionally perturbed, see perturb_prefix)."""
    from lerobot.policies.common.flow_matching import FlowConvention,make_flow_matching_inputs
    from lerobot.policies.common.vla_utils import make_att_2d_masks
    if noise is None:noise=model.sample_noise(actions.shape,actions.device)
    if time is None:time=model.sample_time(actions.shape[0],actions.device)
    x_t,u_t,token_time=make_flow_matching_inputs(actions,noise,time,convention=FlowConvention.NOISE_AT_ONE,prefix_mask=prefix_mask)
    x_t=perturb_prefix(x_t,prefix_mask,sigma=prefix_noise,real_dims=real_dims,generator=generator)
    prefix_embs,prefix_pad_masks,prefix_att_masks=model.embed_prefix(images,img_masks,lang_tokens,lang_masks,state=state)
    suffix_embs,suffix_pad_masks,suffix_att_masks=embed_suffix_tokens(model,x_t,token_time)
    pad_masks=torch.cat([prefix_pad_masks,suffix_pad_masks],dim=1);att_masks=torch.cat([prefix_att_masks,suffix_att_masks],dim=1)
    (_,suffix_out),_=model.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pad_masks,att_masks),
                                                   position_ids=torch.cumsum(pad_masks,dim=1)-1,past_key_values=None,
                                                   inputs_embeds=[prefix_embs,suffix_embs],use_cache=False)
    v_t=model.action_out_proj(suffix_out[:,-model.config.chunk_size:].to(dtype=torch.float32))
    return F.mse_loss(u_t,v_t,reduction='none')


def prefix_training_loss(policy,batch,prefix_mask,noise=None,time=None,prefix_noise=0.,generator=None):
    """SmolVLAPolicy.forward's mean loss with the clean prefix removed from numerator and denominator."""
    actions=policy.prepare_action(batch)
    if prefix_mask.shape!=actions.shape[:2]or prefix_mask.dtype!=torch.bool:raise ValueError('Prefix mask must be boolean (batch, horizon)')
    prefix_mask=prefix_mask.to(actions.device)
    losses=prefix_flow_losses(policy.model,*model_inputs(policy,batch),actions,noise,time,prefix_mask,
                              prefix_noise,generator,policy.config.action_feature.shape[0])
    losses=losses[:,:,:policy.config.action_feature.shape[0]]
    valid=~(batch['action_is_pad']|prefix_mask)
    loss=(losses*valid.unsqueeze(-1)).sum()/(valid.sum()*losses.shape[-1]).clamp_min(1)
    return loss,{'loss':loss.item(),'prefixGoals':int(prefix_mask.sum())}
