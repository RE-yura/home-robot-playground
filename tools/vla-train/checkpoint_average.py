"""Uniform mean of the policy weights of several completed checkpoints of ONE training run (lecture 7 p.28: average
the last checkpoints instead of trusting a single one). The output is a checkpoint directory named after the latest
source; every other file is that source's, unchanged, and hsr-checkpoint.json records the averaged sources.
Usage: checkpoint_average.py --checkpoint DIR --checkpoint DIR [...] --output ROOT"""
import argparse, hashlib, json, shutil
from pathlib import Path

from colab_recording import IDENTITY_KEYS, verify_checkpoint


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def average_checkpoints(sources, output_root):
    import torch
    from safetensors.torch import load_file, save_file
    if len(sources) < 2: raise ValueError('Averaging needs at least two checkpoints')
    loaded = []
    for source in map(Path, sources):
        saved = json.loads((source/'hsr-checkpoint.json').read_text())
        verify_checkpoint(source, saved)
        loaded.append((saved['trainingStep'], source, saved))
    loaded.sort(key=lambda item: item[0])
    if len({step for step, *_ in loaded}) != len(loaded): raise ValueError('Checkpoints must be distinct steps')
    if any({k: s.get(k) for k in IDENTITY_KEYS} != {k: loaded[0][2].get(k) for k in IDENTITY_KEYS} for *_, s in loaded):
        raise ValueError('Checkpoints must come from the same run (identical identity)')
    tensors = [load_file(str(source/'policy/model.safetensors')) for _, source, _ in loaded]
    first = tensors[0]
    for other in tensors[1:]:
        if set(other) != set(first) or any(other[k].shape != first[k].shape or other[k].dtype != first[k].dtype for k in first):
            raise ValueError('Checkpoint tensor names, shapes or dtypes differ')
    mean = {k: (sum(t[k].double() for t in tensors) / len(tensors)).to(first[k].dtype).contiguous() for k in first}
    step, last, saved = loaded[-1]
    destination = Path(output_root)/last.name; temporary = Path(output_root)/f'.{last.name}-writing'
    if destination.exists() or temporary.exists(): raise FileExistsError('Preserve an existing averaged checkpoint')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(last, temporary); (temporary/'FILES.json').unlink()
    save_file(mean, str(temporary/'policy/model.safetensors'), metadata={'format': 'pt'})
    saved = dict(saved, checkpointAverage={'method': 'uniform-mean', 'sources': [
        {'trainingStep': s, 'path': str(p), 'manifestSha256': sha(p/'FILES.json'), 'modelSha256': sha(p/'policy/model.safetensors')}
        for s, p, _ in loaded]})
    (temporary/'hsr-checkpoint.json').write_text(json.dumps(saved, indent=2)+'\n')
    files = [{'path': p.relative_to(temporary).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(temporary.rglob('*')) if p.is_file()]
    (temporary/'FILES.json').write_text(json.dumps(files, indent=2)+'\n'); temporary.rename(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='append', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = average_checkpoints(args.checkpoint, args.output)
    print(json.dumps({'checkpoint': str(out), 'manifestSha256': sha(out/'FILES.json'), 'modelSha256': sha(out/'policy/model.safetensors')}))
