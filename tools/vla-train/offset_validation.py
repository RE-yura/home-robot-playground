"""Compare checkpoints on validation frames with delayed or undelayed action labels.

Evaluation only (no optimizer). Every checkpoint sees the same frames, flow-matching
noise/time draws and sampling noise, so differences come from the weights alone.
Reports the native flow-matching loss and the sampled chunk's normalized absolute
error against the labels, each over non-padded goals. With --prefix-delays, instead
reports the sampled error on the goals after a clamped ground-truth prefix of d goals
(training-time action conditioning), d = 0 being plain sampling.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--checkpoint',action='append',required=True,help='name=path to a step-NNNNNN directory')
    parser.add_argument('--frames',type=int,default=96);parser.add_argument('--draws',type=int,default=4)
    parser.add_argument('--prefix-delays',type=lambda v:[int(x)for x in v.split(',')])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh evaluation output required')
    from torch.utils.data import default_collate
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    import hsr_training as h
    stats=json.loads((args.dataset/'train'/'meta'/'stats.json').read_text())
    datasets={offset:h.hsr_datasets(args.dataset,offset)[1]for offset in h.ACTION_OFFSET_STEPS}
    frames=[int(i)for i in np.linspace(0,len(datasets[0])-1,args.frames)]
    report={'schema':'hsr-offset-validation-v1','frames':frames,'draws':args.draws,'checkpoints':{}}
    for entry in args.checkpoint:
        name,path=entry.split('=',1);checkpoint=Path(path)
        config=h.training_config(PreTrainedConfig.from_pretrained(checkpoint/'policy'),checkpoint/'processor')
        policy=SmolVLAPolicy.from_pretrained(checkpoint/'policy',config=config).float().eval()
        preprocess,_=make_smolvla_pre_post_processors(config,stats)
        rows={}
        if args.prefix_delays:
            import prefix_conditioning as pc
            for delay in args.prefix_delays:
                errors=[]
                for frame in frames:
                    batch=preprocess(h.policy_batch(default_collate([datasets[0][frame]])))
                    actions=policy.prepare_action(batch);mask=torch.arange(16,device=actions.device)[None]<delay
                    valid=~(batch['action_is_pad']|mask)
                    for draw in range(args.draws):
                        noise=torch.randn((1,16,32),generator=torch.Generator().manual_seed(frame*1000+draw)).to(actions.device)
                        with torch.no_grad():sampled=pc.sample_with_prefix(policy.model,*pc.model_inputs(policy,dict(batch)),noise,actions,mask)
                        if valid.any():errors.append(((sampled[...,:6]-actions[...,:6]).abs()[valid]).mean().item())
                if not all(math.isfinite(v)for v in errors):raise FloatingPointError('Nonfinite evaluation value')
                rows[f'prefix{delay}']={'sampledPostfixNormalizedMae':sum(errors)/len(errors),'samples':len(errors)}
                print(name,'prefix',delay,json.dumps(rows[f'prefix{delay}']),flush=True)
        for offset,dataset in ({}if args.prefix_delays else datasets).items():
            losses,errors=[],[]
            for frame in frames:
                batch=preprocess(h.policy_batch(default_collate([dataset[frame]])))
                valid=~batch['action_is_pad'];target=batch['action'][...,:6]
                for draw in range(args.draws):
                    generator=torch.Generator().manual_seed(frame*1000+draw)
                    noise=torch.randn((1,16,32),generator=generator).to(target.device)
                    # Beta(1.5,1) by inverse CDF, as in SmolVLA's flow-matching time sampler.
                    time=(torch.rand((1,),generator=generator)**(2/3)*.999+.001).to(target.device)
                    with torch.no_grad():
                        loss,_=policy.forward(dict(batch),noise=noise.clone(),time=time)
                        policy.reset();sampled=policy.predict_action_chunk(dict(batch),noise=noise.clone())[...,:6]
                    losses.append(loss.item())
                    if valid.any():errors.append(((sampled-target).abs()[valid]).mean().item())
            rows[f'offset{offset}']={'flowLoss':sum(losses)/len(losses),'sampledNormalizedMae':sum(errors)/len(errors),'samples':len(losses)}
            if not all(math.isfinite(v)for v in losses+errors):raise FloatingPointError('Nonfinite evaluation value')
            print(name,offset,json.dumps(rows[f'offset{offset}']),flush=True)
        report['checkpoints'][name]={'path':str(checkpoint),'manifestSha256':h.sha(checkpoint/'FILES.json'),**rows}
        del policy;torch.cuda.empty_cache()
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
