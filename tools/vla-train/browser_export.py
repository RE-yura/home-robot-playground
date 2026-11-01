"""Export one verified HSR checkpoint to the browser model format on CPU.

Parametrized port of the export that produced the public f289 model: native
wrapper parity on three validation frames, legacy ONNX export with CPU parity,
browser fixtures, and unchanged-tensor external weights. The checkpoint is
bound by its FILES.json hash; nothing is published.
"""
import argparse,gc,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
NAMES=['head_rgb','hand_rgb','language_tokens','language_mask','hsr_state','initial_noise']
PREFIX_NAMES=['action_prefix','prefix_mask']
PREFIX_CASE_DELAY=3
WEIGHTS_NAME='hsr-smolvla.weights'


def input_names(prefix_conditioned):
    return NAMES+(PREFIX_NAMES if prefix_conditioned else [])


def adapted_wrapper(policy,normalization,prefix_conditioned):
    """Browser graph: RGB in [0,1], physical state, explicit noise (and, when prefix-conditioned, the executor's
    committed physical goals with a 0/1 mask); physical goals out."""
    import torch
    import hsr_training as h
    class Adapted(torch.nn.Module):
        def __init__(self):
            super().__init__();self.policy=policy
            for side in ('state','action'):
                for key in ('mean','std'):
                    self.register_buffer(side+'_'+key,torch.tensor(normalization[side][key],dtype=torch.float32))
        def forward(self,head,hand,tokens,language_mask,state,noise,action_prefix=None,prefix_mask=None):
            images,masks=self.policy.prepare_images({h.CAMERAS[0]:head,h.CAMERAS[1]:hand})
            normalized=(state-self.state_mean)/(self.state_std+1e-8)
            padded=self.policy.prepare_state({'observation.state':normalized})
            if prefix_conditioned:
                import prefix_conditioning as pc
                goals=torch.nn.functional.pad((action_prefix-self.action_mean)/(self.action_std+1e-8),(0,noise.shape[-1]-6))
                result=pc.sample_with_prefix(self.policy.model,images,masks,tokens,language_mask,padded,noise,goals,prefix_mask>.5)
            else:result=self.policy.model.sample_actions(images,masks,tokens,language_mask,padded,noise=noise)
            return result[:,:,:6]*self.action_std+self.action_mean
    return Adapted()


def digest(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def external_data_helpers():
    spec=importlib.util.spec_from_file_location('external_data',HERE.parent/'vla-learned'/'external-data.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def verified_checkpoint(checkpoint,manifest_sha256,dataset):
    """Bind the checkpoint by its manifest hash and the dataset it was trained on."""
    import hsr_training as h
    from colab_recording import verify_checkpoint
    if digest(checkpoint/'FILES.json')!=manifest_sha256:raise ValueError('Checkpoint manifest hash mismatch')
    saved=json.loads((checkpoint/'hsr-checkpoint.json').read_text())
    identity=verify_checkpoint(checkpoint,saved)
    if identity[0]!=manifest_sha256:raise ValueError('Checkpoint manifest identity mismatch')
    manifest,normalization=h.verify_dataset(dataset,mug_data_fork=bool(saved.get('mugDataFork')))
    if (json_sha(manifest)!=saved['datasetSha256']or json_sha(normalization)!=saved['normalizationSha256']or
        normalization!=saved['normalization']):
        raise ValueError('Dataset/normalization differs from the checkpoint identity')
    return saved,identity


def load_policy(checkpoint,num_steps=10):
    """CPU FP32 policy whose sample loop (unrolled into the graph) runs `num_steps` Euler steps."""
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    import hsr_training as h
    c=h.training_config(PreTrainedConfig.from_pretrained(checkpoint/'policy'),checkpoint/'processor',device='cpu')
    c.num_steps=num_steps
    return SmolVLAPolicy.from_pretrained(checkpoint/'policy',config=c).float().eval()


def export_fp32(checkpoint,saved,identity,dataset,output,prefix_conditioned=False,num_steps=10):
    import torch,onnx,onnxruntime as ort
    from torch.utils.data import default_collate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS,OBS_LANGUAGE_ATTENTION_MASK
    import hsr_training as h
    from colab_recording import verify_checkpoint
    from onnx_export import parity_report,tracing_sdpa_lengths,tracing_integer_cumsum,tracing_integer_full
    output.mkdir(parents=True);started=time.monotonic()
    n=saved['normalization']
    policy=load_policy(checkpoint,num_steps);c=policy.config
    torch.set_num_threads(4)
    stats=json.loads((dataset/'train'/'meta'/'stats.json').read_text())
    preprocess,postprocess=make_smolvla_pre_post_processors(c,stats)
    # Only the observations of these frames are used; the prefix case also takes its first ground-truth goals.
    validation=LeRobotDataset('local/hsr-validation',root=dataset/'validation',
                              delta_timestamps={'action':[i/10 for i in range(16)]},video_backend='pyav')

    names=input_names(prefix_conditioned);model=adapted_wrapper(policy,n,prefix_conditioned).eval();cases=[]
    mean_t,std_t=(torch.tensor(n['action'][key],dtype=torch.float32)for key in ('mean','std'))
    mug_index=next(i for i,row in enumerate(validation.hf_dataset.select_columns(['task_index']).to_list())if int(row['task_index'])==1)
    if validation[mug_index]['task']!=h.INSTRUCTIONS[0]:raise ValueError('Mug validation frame required')
    for number,index in enumerate([0,mug_index,len(validation)-1]):
        raw=h.policy_batch(default_collate([validation[index]]),training=False)
        native=preprocess(dict(raw))
        noise=torch.randn((1,16,32),generator=torch.Generator().manual_seed(42+number),dtype=torch.float32)
        inputs=(raw[h.CAMERAS[0]],raw[h.CAMERAS[1]],native[OBS_LANGUAGE_TOKENS],
                native[OBS_LANGUAGE_ATTENTION_MASK].bool(),raw['observation.state'],noise)
        delay=PREFIX_CASE_DELAY if prefix_conditioned and number==1 else 0
        if prefix_conditioned:
            prefix=validation[index]['action'][None].float()if delay else torch.zeros((1,16,6))
            inputs+=(prefix,(torch.arange(16)<delay).float()[None])
        with torch.inference_mode():
            policy.reset()
            if delay:
                import prefix_conditioning as pc
                goals=torch.nn.functional.pad((prefix-mean_t)/(std_t+1e-8),(0,26))
                sampled=pc.sample_with_prefix(policy.model,*pc.model_inputs(policy,native),noise,goals,inputs[-1]>.5)
                canonical=postprocess(sampled[:,:,:6]).cpu().numpy()
            else:canonical=postprocess(policy.predict_action_chunk(native,noise=noise)).cpu().numpy()
            wrapped=model(*inputs).cpu().numpy()
        proof=parity_report(canonical,wrapped)
        if not proof['passed']:raise ValueError('Export input/normalization wrapper differs from native policy')
        cases.append((inputs,canonical,{'validationIndex':index,'instruction':raw['task'][0],'prefixDelay':delay,'nativeWrapperParity':proof}))
        print('ONNX_NATIVE_REFERENCE_READY',number,index,flush=True)
    graph=output/'hsr-smolvla.onnx'
    import transformers.masking_utils as masking
    with (torch.no_grad(),tracing_sdpa_lengths(torch,masking),tracing_integer_cumsum(torch),tracing_integer_full(torch)):
        torch.onnx.export(model,cases[0][0],str(graph),input_names=names,output_names=['hsr_action_chunk'],
                          opset_version=18,dynamo=False,do_constant_folding=True,external_data=True)
    onnx.checker.check_model(str(graph))
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(graph),sess_options=options,providers=['CPUExecutionProvider'])
    if session.get_providers()!=['CPUExecutionProvider']:raise ValueError('Require explicitly identified CPU inference')
    if {i.name for i in session.get_inputs()}!=set(names):raise ValueError('An authorized policy input was omitted from the graph')
    mean=np.asarray(n['action']['mean'],dtype=np.float32);std=np.asarray(n['action']['std'],dtype=np.float32)
    reports=[]
    for number,(inputs,canonical,metadata) in enumerate(cases):
        feed={name:value.cpu().numpy()for name,value in zip(names,inputs)}
        begin=time.monotonic();actual=session.run(['hsr_action_chunk'],feed)[0]
        report=dict(metadata,**parity_report(canonical,actual),inferenceSeconds=time.monotonic()-begin)
        report['normalizedActionParity']=parity_report((canonical-mean)/std,(actual-mean)/std)
        report['passed']=report['passed']and report['normalizedActionParity']['passed']
        np.savez(output/f'case-{number}.npz',**feed,expected=canonical,actual=actual)
        reports.append(report);print('ONNX_CASE_PARITY',json.dumps(report),flush=True)
    if verify_checkpoint(checkpoint,saved)!=identity:raise ValueError('Original learned checkpoint changed')
    proof={'schema':'hsr-adapted-onnx-cpu-parity-v1','passed':all(r['passed']for r in reports),
           'inputNames':names,'prefixConditioned':prefix_conditioned,'numSteps':num_steps,'outputShape':[1,16,6],'cases':reports,'comparedValues':sum(r['comparedValues']for r in reports),
           'sourceCheckpointManifestSha256':identity[0],'provider':'CPUExecutionProvider',
           'torch':torch.__version__,'onnx':onnx.__version__,'onnxruntime':ort.__version__,
           'elapsedSeconds':time.monotonic()-started,'originalCheckpointPreserved':True,
           'browserParity':'unverified','learnedRolloutGate':'unverified','publicationAllowed':False,
           'files':[{'name':p.name,'bytes':p.stat().st_size,'sha256':digest(p)}for p in sorted(output.iterdir())if p.is_file()]}
    (output/'onnx-parity.json').write_text(json.dumps(proof,indent=2)+'\n')
    if not proof['passed']:raise ValueError('Adapted ONNX parity failed at unchanged tolerance')
    return proof


def prepare_browser_assets(checkpoint,saved,identity,dataset,export,output):
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS,OBS_LANGUAGE_ATTENTION_MASK
    import hsr_training as h
    proof=json.loads((export/'onnx-parity.json').read_text())
    if not proof['passed']or proof['sourceCheckpointManifestSha256']!=identity[0]or len(proof['cases'])!=3:
        raise ValueError('Verified checkpoint and three native cases required')
    files={item['name']:item for item in proof['files']}
    graph=files['hsr-smolvla.onnx']
    if digest(export/'hsr-smolvla.onnx')!=graph['sha256']:raise ValueError('Exported graph identity changed')
    references=[]
    for index in range(3):
        path=export/f'case-{index}.npz'
        if path.stat().st_size!=files[path.name]['bytes']or digest(path)!=files[path.name]['sha256']:
            raise ValueError('Native fixture identity changed')
        with np.load(path,allow_pickle=False)as case:
            references.append({name:case[name].reshape(-1).tolist()for name in proof['inputNames']+['expected']})
    settings=h.training_config(PreTrainedConfig.from_pretrained(checkpoint/'policy'),checkpoint/'processor',device='cpu')
    preprocess,_=make_smolvla_pre_post_processors(settings,json.loads((dataset/'train/meta/stats.json').read_text()))
    languages={};first=references[0]
    for instruction in h.INSTRUCTIONS:
        raw={h.CAMERAS[0]:torch.tensor(first['head_rgb'],dtype=torch.float32).reshape(1,3,240,320),
             h.CAMERAS[1]:torch.tensor(first['hand_rgb'],dtype=torch.float32).reshape(1,3,240,320),
             'observation.state':torch.tensor(first['hsr_state'],dtype=torch.float32).reshape(1,6),'task':[instruction]}
        native=preprocess(h.policy_batch(raw,training=False))
        languages[instruction]={'tokens':native[OBS_LANGUAGE_TOKENS].reshape(-1).tolist(),
                                'mask':native[OBS_LANGUAGE_ATTENTION_MASK].bool().reshape(-1).tolist()}
    for reference,metadata in zip(references,proof['cases']):
        language=languages[metadata['instruction']]
        if reference['language_tokens']!=language['tokens']or reference['language_mask']!=language['mask']:
            raise ValueError('Pinned native processor disagrees with exported tokens')
    output.mkdir(parents=True)
    manifest={'schema':'hsr-learned-browser-v1','graph':{'bytes':graph['bytes'],'sha256':graph['sha256']},
              'instructions':languages,'actionNormalization':saved['normalization']['action'],
              'sourceCheckpointManifestSha256':identity[0],'processor':saved['processor'],
              'browserParity':'unverified','learnedRolloutGate':'unverified','publicationAllowed':False}
    (output/'browser-manifest.json').write_text(json.dumps(manifest,separators=(',',':'),allow_nan=False)+'\n')
    for index,reference in enumerate(references):
        (output/f'browser-case-{index}.json').write_text(json.dumps(reference,separators=(',',':'),allow_nan=False)+'\n')
    return manifest


def convert_external_data(export,output,normalization,checkpoint_sha256):
    import onnx,onnxruntime as ort
    from onnx_export import parity_report
    helpers=external_data_helpers()
    native=json.loads((export/'onnx-parity.json').read_text())
    if not native['passed']or native['sourceCheckpointManifestSha256']!=checkpoint_sha256 or len(native['cases'])!=3:
        raise ValueError('Verified learned export required')
    files={item['name']:item for item in native['files']};original=export/'hsr-smolvla.onnx'
    source=files['hsr-smolvla.onnx']
    if digest(original)!=source['sha256']:raise ValueError('Original FP32 graph identity changed')
    output.mkdir(parents=True);target=output/'hsr-smolvla.onnx'
    model=onnx.load(original);before_tensors=helpers.tensor_identity(model)
    onnx.save_model(model,target,save_as_external_data=True,all_tensors_to_one_file=True,
                    location=WEIGHTS_NAME,size_threshold=1024,convert_attribute=False)
    before_graph=helpers.graph_identity_without_tensor_storage(model);del model;gc.collect()
    model=onnx.load(target,load_external_data=True)
    after_tensors=helpers.tensor_identity(model);after_graph=helpers.graph_identity_without_tensor_storage(model)
    del model;gc.collect()
    if before_tensors!=after_tensors or before_graph!=after_graph:
        raise ValueError('External serialization changed graph operations or tensor bytes')
    onnx.checker.check_model(str(target),full_check=True)
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(target),sess_options=options,providers=['CPUExecutionProvider'])
    mean,std=(np.asarray(normalization['action'][key],dtype=np.float32)for key in ('mean','std'))
    cases=[]
    for index in range(3):
        with np.load(export/f'case-{index}.npz',allow_pickle=False)as reference:
            actual=session.run(['hsr_action_chunk'],{item.name:reference[item.name]for item in session.get_inputs()})[0]
            physical=parity_report(reference['expected'],actual)
            normalized=parity_report((reference['expected']-mean)/std,(actual-mean)/std)
            cases.append({'case':index,'passed':physical['passed']and normalized['passed'],'physical':physical,'normalized':normalized})
    del session
    proof={'schema':'hsr-learned-external-data-v1','passed':all(c['passed']for c in cases),
           'sourceGraph':{'bytes':source['bytes'],'sha256':source['sha256']},
           'outputGraph':{'bytes':target.stat().st_size,'sha256':digest(target),'format':'onnx-external-data'},
           'externalData':[{'path':WEIGHTS_NAME,'bytes':(output/WEIGHTS_NAME).stat().st_size,'sha256':digest(output/WEIGHTS_NAME)}],
           'precision':'unchanged-FP32','unchangedTensorIdentity':before_tensors,'unchangedGraphStructureSha256':before_graph,
           'cases':cases,'originalGraphPreserved':digest(original)==source['sha256'],'publicationAllowed':False}
    (output/'conversion-receipt.json').write_text(json.dumps(proof,indent=2)+'\n')
    if not proof['passed']or not proof['originalGraphPreserved']:raise ValueError('All unchanged physical/normalized tolerances must pass')
    return proof


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--checkpoint-manifest-sha256',required=True)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prefix-conditioned',action='store_true',help='add action_prefix/prefix_mask inputs (training-time RTC checkpoints)')
    parser.add_argument('--num-steps',type=int,choices=(10,5),default=10,help='Euler steps unrolled into the graph (the trained default is 10)')
    args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError('Fresh export output required')
    sys.path.insert(0,str(HERE))
    saved,identity=verified_checkpoint(args.checkpoint,args.checkpoint_manifest_sha256,args.dataset)
    if args.prefix_conditioned and not saved['trainingSettings'].get('rtcTrainingMaxDelay'):raise ValueError('Prefix inputs need a prefix-conditioned checkpoint')
    export=export_fp32(args.checkpoint,saved,identity,args.dataset,args.output/'onnx-fp32',args.prefix_conditioned,num_steps=args.num_steps)
    manifest=prepare_browser_assets(args.checkpoint,saved,identity,args.dataset,args.output/'onnx-fp32',args.output/'browser-assets')
    conversion=convert_external_data(args.output/'onnx-fp32',args.output/'external-data',saved['normalization'],identity[0])
    summary={'schema':'hsr-browser-export-v1','sourceCheckpointManifestSha256':identity[0],'nativeParity':export['passed'],'inputNames':export['inputNames'],'numSteps':args.num_steps,
             'fp32Graph':manifest['graph'],'externalGraph':conversion['outputGraph'],'weights':conversion['externalData'][0],
             'browserFiles':[{'name':p.name,'bytes':p.stat().st_size,'sha256':digest(p)}for p in sorted((args.output/'browser-assets').iterdir())],
             'browserParity':'unverified','publicationAllowed':False}
    (args.output/'export-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('BROWSER_EXPORT_RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':main()
