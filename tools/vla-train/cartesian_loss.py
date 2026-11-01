"""HSR position supervision from recorded joint goals; training only."""
import torch

POSITION_WEIGHT = .1
POSITION_SCALE_METRES = .01

def hsr_tcp(q):
    """Exact position equivalent of src/robot/hsr-ik.ts forwardHSR."""
    lift, arm, roll, wrist, hand = q.unbind(-1)
    x, y, z = .012*torch.cos(hand), .012*torch.sin(hand), torch.full_like(hand,.2205)
    x, z = torch.cos(wrist)*x-torch.sin(wrist)*z, torch.sin(wrist)*x+torch.cos(wrist)*z
    x, y = torch.cos(roll)*x-torch.sin(roll)*y, torch.sin(roll)*x+torch.cos(roll)*y
    x, z = torch.cos(arm)*x-torch.sin(arm)*z, torch.sin(arm)*x+torch.cos(arm)*z
    ros_x = .141+.005*torch.cos(arm)-.345*torch.sin(arm)+x
    ros_y = .078+y
    ros_z = .34+lift+.005*torch.sin(arm)+.345*torch.cos(arm)+z
    return torch.stack((-ros_y,ros_z,-ros_x),dim=-1)

def cartesian_training_loss(policy,batch,normalization):
    """Keep the pinned policy's flow loss, adding masked physical TCP error."""
    actions = policy.prepare_action(batch)
    noise = policy.model.sample_noise(actions.shape,actions.device)
    time = policy.model.sample_time(actions.shape[0],actions.device)
    velocities = []
    handle = policy.model.action_out_proj.register_forward_hook(lambda _module,_args,value:velocities.append(value))
    try:
        flow_loss,info = policy.forward(batch,noise=noise,time=time)
    finally:
        handle.remove()
    if len(velocities)!=1:
        raise RuntimeError('Expected one pinned action projection during training')
    # NOISE_AT_ONE: x_t=t*noise+(1-t)*action, target velocity=noise-action.
    # Geometry uses the estimated clean action; it never adds policy features.
    with torch.autocast(device_type=actions.device.type,enabled=False):
        t = time.float().reshape(-1,1,1)
        target = actions.float()[...,:6]
        estimated = t*noise.float()[...,:6]+(1-t)*target-t*velocities[0].float()[...,:6]
        mean = torch.as_tensor(normalization['mean'],dtype=torch.float32,device=actions.device)
        std = torch.as_tensor(normalization['std'],dtype=torch.float32,device=actions.device)
        position_error = hsr_tcp((estimated*std+mean)[...,:5])-hsr_tcp((target*std+mean)[...,:5])
        squared = position_error.square().sum(-1)/(3*POSITION_SCALE_METRES**2)
        valid = ~batch['action_is_pad']
        geometry_loss = (squared*valid).sum()/valid.sum().clamp_min(1)
    loss = flow_loss+POSITION_WEIGHT*geometry_loss
    return loss,{**info,'flowLoss':flow_loss.item(),'cartesianLoss':geometry_loss.item(),'loss':loss.item()}
