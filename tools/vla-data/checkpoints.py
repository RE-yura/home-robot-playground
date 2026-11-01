"""Certify and package bounded capture batches, including entirely failed batches."""
import argparse,hashlib,json,subprocess,zipfile
from pathlib import Path
from certification import certify_attempt
from dataset import build_index

INSTRUCTIONS=('Put the mug in the basket.','Put the tennis ball in the basket.')
ROOT=Path(__file__).resolve().parents[2]
CERTIFIER_FILES=('tools/vla-data/certification.py','tools/vla-data/dataset.py','tools/vla-data/checkpoints.py')

def saved_certifier():
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT):raise ValueError('Save certification source before packaging real data')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    return {'sourceCommit':head,'sourceHashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in CERTIFIER_FILES}}

def certify_folders(inputs):
    records,paths,attempts,seen=[],[],[],set()
    for folder in inputs:
        report=json.loads((folder/'collection-report.json').read_text())
        for run in report['runs']:
            identity=run['episodeId']
            if identity in seen:raise ValueError('Duplicate attempt across input folders')
            seen.add(identity)
            receipt=json.loads((folder/f'{identity}-receipt.json').read_text())
            if receipt!=run:raise ValueError('Attempt receipt differs from collection report')
            attempts.append({'episodeId':identity,'seed':run['seed'],'instruction':run['instruction'],
                             'packetCreated':run['packetCreated'],'runnerFailure':run.get('runnerFailure')})
            if not run['packetCreated']:continue
            path=folder/f'{identity}.json'
            item,decoded=certify_attempt(path,folder/f'{identity}-audit.json',run,report)
            del decoded
            item['episode'].pop('images')
            records.append(item);paths.append(path)
    build_index(records)
    return records,paths,attempts

def summarize(rows,attempts):
    counts={side:{instruction:{'episodes':sum(r['status']=='accepted'and r['partition']==side and r['instruction']==instruction for r in rows),
                              'uniqueSeeds':len({r['seed']for r in rows if r['status']=='accepted'and r['partition']==side and r['instruction']==instruction})}
                  for instruction in INSTRUCTIONS}for side in ('train','validation')}
    ready=all(counts['train'][i]['uniqueSeeds']>=50 and counts['validation'][i]['uniqueSeeds']>=10 for i in INSTRUCTIONS)
    return {'attempts':len(attempts),'accepted':sum(r['status']=='accepted'for r in rows),
            'rejected':sum(r['status']=='rejected'for r in rows),'packetlessFailures':sum(not a['packetCreated']for a in attempts),
            'counts':counts,'collectionCountsComplete':ready,'datasetReady':False,'finalReview':'pending'}

def package_checkpoint(inputs,output,certification=None):
    if output.exists():raise FileExistsError('Preserve prior checkpoint: '+str(output))
    records,paths,attempts=certify_folders(inputs);rows=build_index(records)
    summary=summarize(rows,attempts)
    # Allow-list raw episode/audit/receipt/report files only. Never glob logs,
    # credentials, environment files or caches from an output directory.
    files={}
    for number,folder in enumerate(inputs):
        report=json.loads((folder/'collection-report.json').read_text())
        names=['collection-report.json']
        for run in report['runs']:
            identity=run['episodeId'];names.append(f'{identity}-receipt.json')
            if run['packetCreated']:names.extend([f'{identity}.json',f'{identity}-audit.json'])
            if run.get('preparationSha256'):
                name=f'{identity}-preparation.json';raw=(folder/name).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=run['preparationSha256']:raise ValueError('Preparation failure bytes changed')
                names.append(name)
        for name in names:files[f'raw/{number}/{name}']=folder/name
    manifest={'schema':'hsr-demo-checkpoint-v1','kind':'demonstration','summary':summary,'certification':certification,
              'attempts':attempts,'episodes':rows,
              'decisions':[{'episodeId':r['episode']['collection']['episodeId'],'acceptance':r['acceptance']}for r in records],
              'files':[{'path':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for name,p in files.items()]}
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
        for name,p in files.items():z.write(p,name)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    proof={'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    verify_checkpoint(output,proof)
    return dict(proof,**summary)

def verify_checkpoint(archive,proof):
    raw=archive.read_bytes()
    if len(raw)!=proof['bytes']or hashlib.sha256(raw).hexdigest()!=proof['sha256']:raise ValueError('Checkpoint archive hash mismatch')
    with zipfile.ZipFile(archive)as z:
        manifest=json.loads(z.read('MANIFEST.json'))
        if manifest.get('schema')!='hsr-demo-checkpoint-v1':raise ValueError('Unsupported checkpoint manifest')
        names=[r['path']for r in manifest['files']]
        if len(set(names))!=len(names)or set(z.namelist())!=set(names)|{'MANIFEST.json'}:raise ValueError('Checkpoint file set mismatch')
        for item in manifest['files']:
            data=z.read(item['path'])
            if len(data)!=item['bytes']or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Checkpoint inner file hash mismatch')
        if summarize(manifest['episodes'],manifest['attempts'])!=manifest['summary']:raise ValueError('Checkpoint counts mismatch')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(package_checkpoint(args.input,args.output,saved_certifier())))
