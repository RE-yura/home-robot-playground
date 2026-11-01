"""Prepare private browser fixtures from the immutable, verified learned export.

Run in the trainer environment. Never copies weights into Git.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

GRAPH_HASH = '7ac9785cdafb5e763761769484c4b037f8d397d173475d048b9d7dcf85ede1ff'
GRAPH_BYTES = 1592644520
INSTRUCTIONS = ('Put the mug in the basket.', 'Put the tennis ball in the basket.')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(config_path, export, output):
    import numpy as np
    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    config = json.loads(config_path.read_text())
    sys.path.insert(0, str(Path(config['inputRoot']) / 'code'))
    import hsr_training as h
    from colab_recording import verify_checkpoint
    checkpoint = Path(config['trainingRoot']) / 'step-002000'
    original = verify_checkpoint(checkpoint, config['expectedIdentity'])
    proof = json.loads((export / 'onnx-parity.json').read_text())
    if not proof['passed'] or proof['sourceCheckpointManifestSha256'] != original[0] or len(proof['cases']) != 3:
        raise ValueError('Verified original checkpoint and three native cases required')
    graph = export / 'hsr-smolvla.onnx'
    if graph.stat().st_size != GRAPH_BYTES or digest(graph) != GRAPH_HASH:
        raise ValueError('Original learned graph identity changed')
    files = {item['name']: item for item in proof['files']}
    references = []
    for index in range(3):
        path = export / f'case-{index}.npz'
        if path.stat().st_size != files[path.name]['bytes'] or digest(path) != files[path.name]['sha256']:
            raise ValueError('Original native fixture identity changed')
        with np.load(path, allow_pickle=False) as case:
            references.append({name: case[name].reshape(-1).tolist() for name in (
                'head_rgb', 'hand_rgb', 'language_tokens', 'language_mask', 'hsr_state', 'initial_noise', 'expected')})
    settings = h.training_config(PreTrainedConfig.from_pretrained(checkpoint / 'policy'), checkpoint / 'processor', device='cpu')
    stats = json.loads((Path(config['inputRoot']) / 'dataset/train/meta/stats.json').read_text())
    preprocess, _ = make_smolvla_pre_post_processors(settings, stats)
    languages = {}
    first = references[0]
    for instruction in INSTRUCTIONS:
        raw = {h.CAMERAS[0]: torch.tensor(first['head_rgb'], dtype=torch.float32).reshape(1, 3, 240, 320),
               h.CAMERAS[1]: torch.tensor(first['hand_rgb'], dtype=torch.float32).reshape(1, 3, 240, 320),
               'observation.state': torch.tensor(first['hsr_state'], dtype=torch.float32).reshape(1, 6),
               'task': [instruction]}
        native = preprocess(h.policy_batch(raw, training=False))
        languages[instruction] = {'tokens': native[OBS_LANGUAGE_TOKENS].reshape(-1).tolist(),
                                 'mask': native[OBS_LANGUAGE_ATTENTION_MASK].bool().reshape(-1).tolist()}
    for reference, metadata in zip(references, proof['cases']):
        language = languages[metadata['instruction']]
        if reference['language_tokens'] != language['tokens'] or reference['language_mask'] != language['mask']:
            raise ValueError('Pinned native processor disagrees with preserved export tokens')
    if output.exists():
        raise FileExistsError('Use a fresh destination; preserve prior browser evidence')
    output.mkdir(parents=True)
    normalization = config['expectedIdentity']['normalization']['action']
    manifest = {'schema': 'hsr-learned-browser-v1', 'graph': {'bytes': GRAPH_BYTES, 'sha256': GRAPH_HASH},
                'instructions': languages, 'actionNormalization': normalization,
                'sourceCheckpointManifestSha256': original[0], 'processor': config['expectedIdentity']['processor'],
                'browserParity': 'unverified', 'learnedRolloutGate': 'unverified', 'publicationAllowed': False}
    (output / 'browser-manifest.json').write_text(json.dumps(manifest, separators=(',', ':'), allow_nan=False) + '\n')
    for index, reference in enumerate(references):
        (output / f'browser-case-{index}.json').write_text(json.dumps(reference, separators=(',', ':'), allow_nan=False) + '\n')
    if verify_checkpoint(checkpoint, config['expectedIdentity']) != original:
        raise ValueError('Original checkpoint changed during browser preparation')
    receipt = {'schema': 'hsr-learned-browser-assets-v1', 'graph': manifest['graph'],
               'files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(output.iterdir())],
               'originalCheckpointPreserved': True, 'browserParity': 'unverified', 'publicationAllowed': False}
    (output / 'preparation-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.config, args.export, args.output)
