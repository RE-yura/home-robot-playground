"""Export the verified learned HSR policy and compare all actions on real inputs.

CPU numerical parity only; no browser, physical success or publication claim.
The immutable trainer, source weights and normalization are never overwritten.
"""
import argparse
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def parity_report(expected, actual):
    expected,actual=np.asarray(expected),np.asarray(actual)
    if expected.shape!=(1,16,6) or actual.shape!=expected.shape:
        raise ValueError('Full 16-by-six HSR action sequence required')
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError('Nonfinite action sequence')
    difference=np.abs(expected-actual)
    count=int(np.sum(difference > .001+.001*np.abs(expected)))
    return {'passed':count==0,'comparedValues':int(expected.size),
            'violatingValues':count,'maxAbsoluteDifference':float(difference.max()),
            'meanAbsoluteDifference':float(difference.mean()),'atol':.001,'rtol':.001}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


@contextmanager
def tracing_sdpa_lengths(torch_module, masking):
    """Handle scalar trace lengths without treating them as cache positions.

    Transformers 5.5.4's compatibility branch assumes every Tensor length is
    a one-dimensional legacy cache_position. JIT represents a fixed sequence
    length as a scalar Tensor. This export-only shim leaves native inference
    and legacy position vectors unchanged and restores the registry on exit.
    Upstream report: huggingface/transformers#45735.
    """
    interface=masking.ALL_MASK_ATTENTION_FUNCTIONS
    original=interface['sdpa']
    @wraps(original)
    def compatible(*args,**kwargs):
        args=list(args)
        length=kwargs.get('q_length',args[1] if len(args)>1 else None)
        if isinstance(length,torch_module.Tensor) and length.ndim==0:
            length=int(length.item())
            if 'q_length' in kwargs:kwargs['q_length']=length
            else:args[1]=length
        return original(*args,**kwargs)
    interface['sdpa']=compatible
    try:yield
    finally:interface['sdpa']=original


@contextmanager
def tracing_integer_cumsum(torch_module):
    """Make PyTorch's bool-to-int64 cumulative sum promotion explicit in ONNX."""
    original=torch_module.cumsum
    @wraps(original)
    def compatible(value,dim,*,dtype=None,out=None):
        if dtype is None and value.dtype==torch_module.bool:dtype=torch_module.int64
        return original(value,dim,dtype=dtype,out=out)
    torch_module.cumsum=compatible
    try:yield
    finally:torch_module.cumsum=original


@contextmanager
def tracing_integer_full(torch_module):
    """Preserve inferred integer/bool torch.full dtype in the legacy exporter."""
    original=torch_module.full
    @wraps(original)
    def compatible(*args,**kwargs):
        fill=kwargs.get('fill_value',args[1] if len(args)>1 else None)
        if kwargs.get('dtype') is None:
            if isinstance(fill,bool):kwargs['dtype']=torch_module.bool
            elif isinstance(fill,int):kwargs['dtype']=torch_module.int64
        return original(*args,**kwargs)
    torch_module.full=compatible
    try:yield
    finally:torch_module.full=original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    runroot=Path(config['runRecord']).parent
    if not json.loads((runroot/'completion-verification.json').read_text())['passed']:
        raise ValueError('Fully verified learned checkpoint required')
    if args.output.exists():raise FileExistsError('Fresh ONNX output required')
    args.output.mkdir();started=time.monotonic()
    import torch,onnx,onnxruntime as ort
    from torch.utils.data import default_collate
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS,OBS_LANGUAGE_ATTENTION_MASK
    sys.path.insert(0,str(Path(config['inputRoot'])/'code'))
    import hsr_training as h
    checkpoint=Path(config['trainingRoot'])/'step-002000'
    from colab_recording import verify_checkpoint
    original=verify_checkpoint(checkpoint,config['expectedIdentity'])
    n=config['expectedIdentity']['normalization']
    c=h.training_config(PreTrainedConfig.from_pretrained(checkpoint/'policy'),checkpoint/'processor',device='cpu')
    policy=SmolVLAPolicy.from_pretrained(checkpoint/'policy',config=c).float().eval()
    torch.set_num_threads(4)
    stats=json.loads((Path(config['inputRoot'])/'dataset'/'train'/'meta'/'stats.json').read_text())
    preprocess,postprocess=make_smolvla_pre_post_processors(c,stats)
    validation=LeRobotDataset('local/hsr-validation',root=Path(config['inputRoot'])/'dataset'/'validation',
                              delta_timestamps={'action':[i/10 for i in range(16)]},video_backend='pyav')

    class Adapted(torch.nn.Module):
        def __init__(self):
            super().__init__();self.policy=policy
            for side in ('state','action'):
                for key in ('mean','std'):
                    self.register_buffer(side+'_'+key,torch.tensor(n[side][key],dtype=torch.float32))
        def forward(self,head,hand,tokens,language_mask,state,noise):
            images,masks=self.policy.prepare_images({h.CAMERAS[0]:head,h.CAMERAS[1]:hand})
            normalized=(state-self.state_mean)/(self.state_std+1e-8)
            padded=self.policy.prepare_state({'observation.state':normalized})
            result=self.policy.model.sample_actions(images,masks,tokens,language_mask,padded,noise=noise)
            return result[:,:,:6]*self.action_std+self.action_mean

    model=Adapted().eval();cases=[]
    indices=[0,len(validation)//2,len(validation)-1]
    for number,index in enumerate(indices):
        raw=h.policy_batch(default_collate([validation[index]]),training=False)
        native=preprocess(dict(raw))
        noise=torch.randn((1,16,32),generator=torch.Generator().manual_seed(42+number),dtype=torch.float32)
        inputs=(raw[h.CAMERAS[0]],raw[h.CAMERAS[1]],native[OBS_LANGUAGE_TOKENS],
                native[OBS_LANGUAGE_ATTENTION_MASK].bool(),raw['observation.state'],noise)
        with torch.inference_mode():
            policy.reset()
            canonical=postprocess(policy.predict_action_chunk(native,noise=noise)).cpu().numpy()
            wrapped=model(*inputs).cpu().numpy()
        proof=parity_report(canonical,wrapped)
        if not proof['passed']:raise ValueError('Export input/normalization wrapper differs from native policy')
        cases.append((inputs,canonical,{'validationIndex':index,'instruction':raw['task'][0],
                                        'nativeWrapperParity':proof}))
        print('ONNX_NATIVE_REFERENCE_READY',number,index,flush=True)
    names=['head_rgb','hand_rgb','language_tokens','language_mask','hsr_state','initial_noise']
    graph=args.output/'hsr-smolvla.onnx'
    print('ONNX_EXPORT_START',flush=True)
    import transformers.masking_utils as masking
    with (torch.no_grad(),tracing_sdpa_lengths(torch,masking),
          tracing_integer_cumsum(torch),tracing_integer_full(torch)):
        torch.onnx.export(model,cases[0][0],str(graph),input_names=names,
                          output_names=['hsr_action_chunk'],opset_version=18,dynamo=False,
                          do_constant_folding=True,external_data=True)
    onnx.checker.check_model(str(graph))
    print('ONNX_GRAPH_CHECKED',graph.stat().st_size,flush=True)
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(graph),sess_options=options,providers=['CPUExecutionProvider'])
    if session.get_providers()!=['CPUExecutionProvider']:
        raise ValueError('Require explicitly identified CPU inference')
    reports=[]
    for number,(inputs,canonical,metadata) in enumerate(cases):
        feed={name:value.cpu().numpy() for name,value in zip(names,inputs)}
        if {i.name for i in session.get_inputs()}!=set(names):
            raise ValueError('An authorized policy input was omitted from the graph')
        begin=time.monotonic();actual=session.run(['hsr_action_chunk'],feed)[0]
        report=dict(metadata,**parity_report(canonical,actual),inferenceSeconds=time.monotonic()-begin)
        # Preserve the existing normalized-action tolerance as well as physical goals.
        mean=np.asarray(n['action']['mean'],dtype=np.float32)
        std=np.asarray(n['action']['std'],dtype=np.float32)
        normalized=parity_report((canonical-mean)/std,(actual-mean)/std)
        report['normalizedActionParity']=normalized
        report['passed']=report['passed'] and normalized['passed']
        np.savez(args.output/f'case-{number}.npz',**feed,expected=canonical,actual=actual)
        reports.append(report);print('ONNX_CASE_PARITY',json.dumps(report),flush=True)
    if verify_checkpoint(checkpoint,config['expectedIdentity'])!=original:
        raise ValueError('Original learned checkpoint changed')
    proof={'schema':'hsr-adapted-onnx-cpu-parity-v1','passed':all(r['passed'] for r in reports),
           'inputNames':names,'outputShape':[1,16,6],'cases':reports,'comparedValues':sum(r['comparedValues'] for r in reports),
           'sourceCheckpointManifestSha256':original[0],'provider':'CPUExecutionProvider',
           'torch':torch.__version__,'onnx':onnx.__version__,'onnxruntime':ort.__version__,
           'elapsedSeconds':time.monotonic()-started,'originalCheckpointPreserved':True,
           'browserParity':'unverified','learnedRolloutGate':'unverified','publicationAllowed':False,
           'exportCompatibility':'scalar SDPA lengths, bool cumsum promotion, integer full dtype; restored',
           'files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':digest(p)}
                    for p in sorted(args.output.iterdir()) if p.is_file()]}
    (args.output/'onnx-parity.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('ADAPTED_ONNX_CPU_PARITY_RESULT',json.dumps(proof),flush=True)
    if not proof['passed']:raise ValueError('Adapted ONNX parity failed at unchanged tolerance')


if __name__=='__main__':main()
