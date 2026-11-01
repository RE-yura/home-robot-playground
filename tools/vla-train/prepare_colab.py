"""Create concrete hash-addressed Colab uploads only after collection and review."""
import argparse,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
from hsr_training import verify_dataset,sha

ROOT=Path(__file__).resolve().parents[2]
CODE_FILES=('hsr_training.py','recovery_training.py','colab_recording.py','prepare_colab.py','requirements-colab.txt')

def reviewed_gate(path,repo):
    gate=json.loads(path.read_text())
    if gate.get('schema')!='hsr-dataset-review-gate-v1'or gate.get('status')!='complete'or gate.get('unresolvedCritical',1)or gate.get('unresolvedImportant',1):
        raise ValueError('A completed fresh dataset review with material findings resolved is required')
    relative=path.resolve().relative_to(repo)
    saved=subprocess.check_output(['git','show',f'HEAD:{relative.as_posix()}'],cwd=repo)
    if saved!=path.read_bytes():raise ValueError('Save the exact completed review gate in Git')
    return gate

def dataset_files(root):
    allowed=[]
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('Symlinks cannot enter training data')
        if not path.is_file():continue
        name=path.relative_to(root).as_posix();parts=Path(name).parts
        if name in ('index.json','normalization.json','export-status.json'):allowed.append(path);continue
        if parts[0]in('train','validation'):
            rest=parts[1:]
            if rest==('hsr-episode-map.json',)or(rest[0]=='data'and path.suffix=='.parquet')or(rest[0]=='images'and path.suffix=='.png')or(rest[0]=='meta'and path.suffix in('.json','.parquet')):
                allowed.append(path);continue
        raise ValueError('Unknown file cannot enter training bundle: '+name)
    return sorted(allowed)

def archive(output,files,manifest=None):
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
        for name,path in files:z.write(path,name)
        if manifest is not None:z.writestr('dataset-manifest.json',json.dumps(manifest,indent=2)+'\n')
    return {'name':output.name,'bytes':output.stat().st_size,'sha256':sha(output)}

def prepare(inputs,export_root,output,review_path):
    repo=ROOT.resolve()
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo):raise ValueError('Save training source/status before making a handoff')
    if output.exists():raise FileExistsError('Preserve prior handoff output')
    gate=reviewed_gate(review_path,repo)
    sys.path.insert(0,str(repo/'tools/vla-data'))
    from checkpoints import summarize
    from validate_export import validate
    index=json.loads((export_root/'index.json').read_text())
    counts=summarize(index['episodes'],index['attempts'])
    if not counts['collectionCountsComplete']:raise ValueError('Need50 distinct accepted training seeds plus10 validation seeds per instruction')
    round_trip=validate(inputs,export_root)
    files=dataset_files(export_root)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    manifest={'schema':'hsr-training-bundle-v1','datasetReady':True,'sourceCommit':source,'review':gate,'roundTrip':round_trip,
              'files':[{'path':p.relative_to(export_root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}for p in files]}
    # Verify the exact consumer contract without mutating the original export.
    consumer_manifest=export_root/'dataset-manifest.json'
    if consumer_manifest.exists():raise FileExistsError('Preserve prior training manifest')
    consumer_manifest.write_text(json.dumps(manifest,indent=2)+'\n')
    try:verify_dataset(export_root)
    finally:consumer_manifest.unlink()
    output.mkdir(parents=True)
    data=archive(output/'hsr-training-dataset.zip',[(p.relative_to(export_root).as_posix(),p)for p in files],manifest)
    code=archive(output/'hsr-training-code.zip',[(name,Path(__file__).parent/name)for name in CODE_FILES])
    handoff={'schema':'hsr-colab-handoff-v1','datasetReady':True,'trainingSourceCommit':source,'artifacts':[dict(code,kind='code'),dict(data,kind='dataset')],
             'gpuTraining':'unverified','adaptedOnnxParity':'unverified','learnedRolloutGate':'unverified','publicationAllowed':False}
    (output/'colab-handoff.json').write_text(json.dumps(handoff,indent=2)+'\n')
    return handoff

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,nargs='+',required=True);parser.add_argument('--export',dest='export_root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--review-gate',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.input,args.export_root,args.output,args.review_gate)))
