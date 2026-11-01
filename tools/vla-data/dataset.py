"""Whole-episode metadata and train-only statistics; certification is separate."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

REGISTRY_PATH = Path(__file__).with_name('seed-registry.json')
REGISTRY = json.loads(REGISTRY_PATH.read_text())
REGISTRY_SHA256 = hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()
AXES = ['arm_lift_joint', 'arm_flex_joint', 'arm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint', 'gripper_opening']
UNITS = ['m', 'rad', 'rad', 'rad', 'rad', 'ratio']


def partition(seed, *, collection=False):
    if type(seed) is not int or seed < 0 or seed > 0xffffffff or seed in REGISTRY['calibrationSeeds']:
        raise ValueError('Invalid or calibration seed')
    for name, (first, last) in REGISTRY['partitions'].items():
        if first <= seed <= last:
            if collection and name == 'learned-rollout':
                raise ValueError('Held-out rollout seeds cannot become teacher data')
            return name
    raise ValueError('Seed was not reserved before collection')


def build_index(records):
    rows, seen = [], set()
    for item in records:
        episode, acceptance = item['episode'], item['acceptance']
        if episode.get('schema') != 'hsr-demonstration-v1' or episode.get('kind') != 'demonstration':
            raise ValueError('Calibration cannot enter a demonstration index')
        identity = episode['collection']['episodeId']
        if not re.fullmatch(r'[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}', identity):
            raise ValueError('Invalid capture identity')
        if identity in seen:
            raise ValueError('Duplicate episode identity')
        seen.add(identity)
        if acceptance.get('schema') != 'hsr-demo-acceptance-v1' or acceptance.get('status') not in ('accepted', 'rejected'):
            raise ValueError('Certification decision required')
        source_hash = item['sourceSha256']
        if not re.fullmatch(r'[0-9a-f]{64}', source_hash):
            raise ValueError('Source packet hash required')
        instruction = episode['instruction']
        if instruction not in ('Put the mug in the basket.', 'Put the tennis ball in the basket.'):
            raise ValueError('Unsupported instruction')
        rows.append({'episodeId': identity, 'seed': episode['seed'], 'partition': partition(episode['seed'], collection=True),
                     'instruction': instruction, 'status': acceptance['status'], 'sourceSha256': source_hash,
                     'sampleCount': len(episode['samples'])})
    return rows


def training_normalization(records):
    rows = build_index(records)
    chosen = [(item, row) for item, row in zip(records, rows) if row['partition'] == 'train' and row['status'] == 'accepted']
    if not chosen:
        raise ValueError('No accepted training episodes')
    values = {key: [] for key in ('state', 'action')}
    for item, _ in chosen:
        samples = item['episode']['samples']
        if not samples:
            raise ValueError('Empty accepted episode')
        for sample in samples:
            for key in values:
                v = np.asarray(sample[key], dtype=np.float64)
                if v.shape != (6,) or not np.isfinite(v).all():
                    raise ValueError('Expected six finite channels')
                values[key].append(v)
    epsilon = REGISTRY['normalization']['epsilon']
    result = {'schema': 'hsr-training-normalization-v1', 'partition': 'train', 'status': 'accepted',
              'registrySha256': REGISTRY_SHA256, 'axisNames': AXES, 'units': UNITS, 'epsilon': epsilon,
              'episodeIds': [row['episodeId'] for _, row in chosen],
              'sourceSha256': [row['sourceSha256'] for _, row in chosen], 'sampleCount': len(values['state'])}
    for key, vectors in values.items():
        data = np.stack(vectors)
        raw_std = data.std(axis=0, ddof=0)
        result[key] = {'mean': data.mean(axis=0).tolist(), 'std': np.maximum(raw_std, epsilon).tolist(),
                       'rawStd': raw_std.tolist(), 'zeroVarianceAxes': np.flatnonzero(raw_std < epsilon).tolist()}
    return result


def normalize(values, statistics):
    data = np.asarray(values, dtype=np.float64)
    if data.shape[-1:] != (6,) or not np.isfinite(data).all():
        raise ValueError('Expected six finite channels')
    return (data - np.asarray(statistics['mean'])) / np.asarray(statistics['std'])


def denormalize(values, statistics):
    data = np.asarray(values, dtype=np.float64)
    if data.shape[-1:] != (6,) or not np.isfinite(data).all():
        raise ValueError('Expected six finite channels')
    return data * np.asarray(statistics['std']) + np.asarray(statistics['mean'])
