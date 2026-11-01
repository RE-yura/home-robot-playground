"""Register remotely retrieved, re-certified bounded artifacts in a Git-sized ledger."""
import argparse,hashlib,json,re,subprocess,tempfile,zipfile
from pathlib import Path,PurePosixPath
from checkpoints import certify_folders,summarize,CERTIFIER_FILES,ROOT
from dataset import build_index,REGISTRY_SHA256

def catalog_rows(catalog):
    """Effective current quality decisions; immutable artifact indexes stay intact."""
    rows=[dict(r)for b in catalog.get('batches',[])for r in b['episodes']]
    by_id={r['episodeId']:r for r in rows};seen=set()
    for decision in catalog.get('qualityRejections',[]):
        identity=decision.get('episodeId');row=by_id.get(identity)
        certification=decision.get('certification',{})
        if (identity in seen or row is None or row['status']!='accepted' or
            row.get('sourceSha256')!=decision.get('sourceSha256') or
            not re.fullmatch('[0-9a-f]{64}',decision.get('auditSha256','')) or
            not re.fullmatch('[0-9a-f]{40}',certification.get('sourceCommit','')) or
            not decision.get('reasons') or not all(isinstance(r,str)and r for r in decision['reasons'])):
            raise ValueError('Quality rejection must bind one previously accepted immutable packet and audit')
        seen.add(identity);row['status']='rejected'
    return rows

def requalify_catalog(catalog,records,attempts,certification):
    """Only lower acceptance after complete current certification, never relabel raw."""
    original=[r for b in catalog['batches']for r in b['episodes']]
    original_attempts=[r for b in catalog['batches']for r in b['attempts']]
    current=build_index(records)
    old={r['episodeId']:r for r in original};new={r['episodeId']:r for r in current}
    if set(old)!=set(new)or sorted(attempts,key=lambda r:r['episodeId'])!=sorted(original_attempts,key=lambda r:r['episodeId']):
        raise ValueError('Requalify the exact complete durable packet/attempt cohort')
    effective={r['episodeId']:r for r in catalog_rows(catalog)}
    for identity,row in new.items():
        if {k:v for k,v in row.items()if k!='status'}!={k:v for k,v in old[identity].items()if k!='status'}:
            raise ValueError('Requalification cannot change packet bytes or metadata')
        if effective[identity]['status']=='rejected'and row['status']!='rejected':
            raise ValueError('Quality requalification cannot upgrade rejected attempts')
    decisions={r['episodeId']:r for r in catalog.get('qualityRejections',[])}
    for item in records:
        acceptance=item['acceptance'];identity=item['episode']['collection']['episodeId']
        if old[identity]['status']=='accepted'and new[identity]['status']=='rejected':
            decisions[identity]={'episodeId':identity,'sourceSha256':item['sourceSha256'],
                'auditSha256':acceptance['auditSha256'],'reasons':acceptance['reasons'],'certification':certification}
    updated={**catalog,'qualityRejections':list(decisions.values())}
    updated['summary']=summarize(catalog_rows(updated),original_attempts)
    return updated

def merge_catalog(existing,batch):
    catalog=existing or {'schema':'hsr-durable-demo-catalog-v1','seedRegistrySha256':REGISTRY_SHA256,'batches':[]}
    if catalog['schema']!='hsr-durable-demo-catalog-v1'or catalog['seedRegistrySha256']!=REGISTRY_SHA256:
        raise ValueError('Unsupported durable catalog or seed registry')
    for old in catalog['batches']:
        if old['artifact']['sha256']==batch['artifact']['sha256']:
            if old['episodes']!=batch['episodes']or old['attempts']!=batch['attempts']:raise ValueError('Changed artifact identity')
            return catalog
    seen={r['episodeId']for b in catalog['batches']for r in b['attempts']}
    incoming=[r['episodeId']for r in batch['attempts']]
    if len(set(incoming))!=len(incoming)or seen.intersection(incoming):raise ValueError('Duplicate or changed episode identity')
    batches=catalog['batches']+[batch]
    updated={**catalog,'batches':batches}
    updated['summary']=summarize(catalog_rows(updated),[r for b in batches for r in b['attempts']])
    return updated

def register(archive,proof,catalog_path):
    raw=archive.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if (proof.get('sha256')!=sha or proof.get('bytes')!=len(raw)or proof.get('storage')!='fresh-lfs-storage' or
        not re.fullmatch('[0-9a-f]{40}',proof.get('commit',''))or proof.get('file')!=f'artifacts/{sha}.zip'or
        proof.get('url')!=f"https://github.com/RE-yura/home-robot-playground/blob/{proof['commit']}/artifacts/{sha}.zip"):
        raise ValueError('Verified private artifact retrieval proof required')
    with zipfile.ZipFile(archive)as z,tempfile.TemporaryDirectory()as directory:
        manifest=json.loads(z.read('MANIFEST.json'))
        if manifest.get('schema')not in ('hsr-demo-checkpoint-v1','hsr-demo-pilot-artifact-v1'):raise ValueError('Unsupported demonstration archive')
        if manifest['schema']=='hsr-demo-checkpoint-v1':
            certification=manifest.get('certification')or{}
            if not re.fullmatch('[0-9a-f]{40}',certification.get('sourceCommit',''))or set(certification.get('sourceHashes',{}))!=set(CERTIFIER_FILES):
                raise ValueError('Saved certification source required')
            for path,expected in certification['sourceHashes'].items():
                saved=subprocess.check_output(['git','show',f"{certification['sourceCommit']}:{path}"],cwd=ROOT)
                if hashlib.sha256(saved).hexdigest()!=expected:raise ValueError('Certification source hash mismatch')
        names=[item['path']for item in manifest['files']]
        if len(set(names))!=len(names)or set(z.namelist())!=set(names)|{'MANIFEST.json'}:raise ValueError('Archive file set mismatch')
        root=Path(directory)
        for item in manifest['files']:
            name=PurePosixPath(item['path'])
            if name.is_absolute()or '..'in name.parts:raise ValueError('Unsafe archive path')
            data=z.read(item['path'])
            if len(data)!=item['bytes']or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Inner artifact hash mismatch')
            if name.parts[0]=='raw':
                target=root.joinpath(*name.parts);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        folders=sorted({p.parent for p in (root/'raw').rglob('collection-report.json')})
        if not folders:raise ValueError('Raw capture batches missing')
        records,_,attempts=certify_folders(folders);rows=build_index(records)
        expected=manifest if manifest['schema']=='hsr-demo-checkpoint-v1'else json.loads(z.read('export/index.json'))
        if rows!=expected['episodes']or attempts!=expected['attempts']:raise ValueError('Re-certified archive index mismatch')
    old=json.loads(catalog_path.read_text())if catalog_path.exists()else None
    current=merge_catalog(old,{'artifact':proof,'episodes':rows,'attempts':attempts})
    catalog_path.parent.mkdir(parents=True,exist_ok=True)
    temporary=catalog_path.with_suffix(catalog_path.suffix+'.tmp');temporary.write_text(json.dumps(current,indent=2)+'\n');temporary.replace(catalog_path)
    return current['summary']

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--proof',type=Path,required=True)
    parser.add_argument('--catalog',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(register(args.archive,json.loads(args.proof.read_text()),args.catalog)))
