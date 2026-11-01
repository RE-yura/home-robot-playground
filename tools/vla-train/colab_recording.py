"""Auxiliary log/checkpoint recording; never imports or changes trainer RNG.

File integrity here is not tensor, ONNX, rollout or remote-persistence proof.
Use a unique, independently verified durable directory for each training run.
"""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import uuid

IDENTITY_KEYS = (
    'robotModelId', 'chunkLength', 'executeSamples', 'sampleHz', 'policy',
    'processor', 'lerobotRevision', 'datasetSha256', 'normalizationSha256',
    'sourceCommit', 'trainingCodeHashes', 'policyInputs', 'trainingSettings')
METRIC_KEYS = {'step', 'trainedUpdates', 'trainingLoss', 'optimizerUpdate',
               'lossScale', 'validationLoss', 'flowLoss', 'cartesianLoss'}


def read_metrics(path, start_step=0, start_updates=0):
    """Read complete JSONL rows only; fail on invalid committed rows."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    end = data.rfind(b'\n') + 1
    rows = []
    for line in data[:end].splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or not set(row) <= METRIC_KEYS:
            raise ValueError('Unexpected metric fields')
        if not METRIC_KEYS - {'validationLoss','flowLoss','cartesianLoss'} <= set(row):
            raise ValueError('Missing metric fields')
        if type(row['step']) is not int or row['step'] != start_step + 1:
            raise ValueError('Nonconsecutive training attempts')
        if type(row['optimizerUpdate']) is not bool:
            raise ValueError('Optimizer update must be boolean')
        start_updates += int(row['optimizerUpdate'])
        if type(row['trainedUpdates']) is not int or row['trainedUpdates'] != start_updates:
            raise ValueError('Actual optimizer update count changed')
        for name in ('trainingLoss', 'lossScale', 'validationLoss', 'flowLoss', 'cartesianLoss'):
            if name in row and (type(row[name]) not in (int, float) or
                                not math.isfinite(row[name]) or row[name] < 0):
                raise ValueError('Invalid numeric metric: ' + name)
        if row['lossScale'] <= 0:
            raise ValueError('Loss scale must be positive')
        start_step = row['step']
        rows.append(row)
    return rows


def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_checkpoint(root, expected):
    """Check final directory, exact files/hashes and unchanged input identity."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or not re.fullmatch(r'step-[0-9]{6}', root.name):
        raise ValueError('Only completed checkpoint directories are accepted')
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Checkpoint symlinks are not supported')
    files = json.loads((root/'FILES.json').read_text())
    if not isinstance(files, list) or not files:
        raise ValueError('Missing checkpoint manifest')
    names = set()
    for item in files:
        name = item['path']
        relative = PurePosixPath(name)
        if (not isinstance(name, str) or relative.is_absolute() or
            relative.as_posix() != name or any(p in ('..', '.') for p in relative.parts) or
            name in names or name == 'FILES.json'):
            raise ValueError('Invalid or duplicate manifest path')
        names.add(name)
        path = root/name
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Missing manifested file')
        if (type(item['bytes']) is not int or item['bytes'] < 0 or
            path.stat().st_size != item['bytes'] or _sha(path) != item['sha256']):
            raise ValueError('Checkpoint file size/hash mismatch: ' + name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != names | {'FILES.json'}:
        raise ValueError('Checkpoint has unlisted files')
    if not {'hsr-checkpoint.json','training.pt','policy/model.safetensors','training-metrics.jsonl'} <= names:
        raise ValueError('Required checkpoint files missing')
    saved = json.loads((root/'hsr-checkpoint.json').read_text())
    if any(key not in expected or saved.get(key) != expected[key] for key in IDENTITY_KEYS):
        raise ValueError('Reviewed checkpoint identity changed')
    if (saved.get('schema') != 'hsr-trained-policy-v1' or
        saved.get('publicationAllowed') is not False or
        saved.get('learnedRolloutGate') != 'unverified' or
        saved.get('adaptedOnnxParity') != 'unverified'):
        raise ValueError('Unexpected checkpoint gate claims')
    if type(saved.get('trainingStep')) is not int or saved['trainingStep'] != int(root.name[5:]):
        raise ValueError('Checkpoint step/name mismatch')
    metrics_path = root/'training-metrics.jsonl'
    data = metrics_path.read_bytes()
    if not data.endswith(b'\n'):
        raise ValueError('Checkpoint metrics are incomplete')
    first = json.loads(data.splitlines()[0])
    rows = read_metrics(metrics_path, first['step']-1,
                        first['trainedUpdates']-int(first['optimizerUpdate']))
    if (not rows or rows[-1]['step'] != saved['trainingStep'] or
        rows[-1]['trainedUpdates'] != saved.get('trainedUpdates') or
        type(saved.get('trainedUpdates')) is not int or saved['trainedUpdates'] <= 0):
        raise ValueError('Checkpoint metrics/update count mismatch')
    return _sha(root/'FILES.json'), sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def backup_checkpoint(source, backup_root, expected, reserve_bytes=256*1024**2):
    """Copy and rehash before atomic final rename; never prune/replace backups.

    Free-space reporting and readback alone do not prove cloud durability.
    A failed partial copy is retained with its unique temporary name.
    """
    source, backup_root = Path(source), Path(backup_root)
    manifest_hash, size = verify_checkpoint(source, expected)
    backup_root.mkdir(parents=True, exist_ok=True)
    if backup_root.resolve().is_relative_to(source.resolve()):
        raise ValueError('Backup cannot be inside its source checkpoint')
    destination = backup_root/source.name
    if destination.exists():
        if verify_checkpoint(destination, expected)[0] != manifest_hash:
            raise ValueError('Different backup already exists; preserved')
        return destination
    if shutil.disk_usage(backup_root).free < size + reserve_bytes:
        raise ValueError('Insufficient reported backup space')
    temporary = backup_root/f'.{source.name}-copying-{uuid.uuid4().hex}'
    shutil.copytree(source, temporary)
    # Validate under the final name in an isolated temporary parent.
    verification_parent = backup_root/f'.verify-{uuid.uuid4().hex}'
    verification_parent.mkdir()
    candidate = verification_parent/source.name
    temporary.rename(candidate)
    if verify_checkpoint(candidate, expected)[0] != manifest_hash:
        raise ValueError('Copied manifest changed')
    for path in candidate.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    if destination.exists():
        raise ValueError('Backup appeared during copy; both copies preserved')
    candidate.rename(destination)
    verification_parent.rmdir()
    if verify_checkpoint(destination, expected)[0] != manifest_hash:
        raise ValueError('Final backup readback mismatch')
    return destination


class OfflineRecorder:
    """Poll from a bounded foreground wrapper, separate from the trainer.

    This component does not start training, authenticate, sync online or prove
    remote storage durability. Always close it in the wrapper's finally block.
    """
    def __init__(self, training_root, backup_root, expected, start_step=0, start_updates=0):
        import wandb
        if wandb.__version__ != '0.30.0':
            raise ValueError('Use isolated recording environment with wandb==0.30.0')
        if any(key not in expected for key in IDENTITY_KEYS):
            raise ValueError('Complete reviewed checkpoint identity required')
        if any(type(value) is not int or value < 0 for value in (start_step, start_updates)):
            raise ValueError('Nonnegative resume counters required')
        self.training_root, self.backup_root = Path(training_root), Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=False)
        self.expected = {key: expected[key] for key in IDENTITY_KEYS}
        self.start_step, self.start_updates = start_step, start_updates
        self.last_step = start_step
        self.last_updates = start_updates
        self.backed_up = set()
        self._metrics_bytes = b''
        self._closed = False
        self.run = wandb.init(project='hsr-smolvla', mode='offline',
                              dir=str(self.backup_root), save_code=False,
                              config=dict(self.expected, recordingOnly=True,
                                          trainingCompletion='unverified'),
                              settings=wandb.Settings(console='off', disable_git=True,
                                                      disable_code=True, x_disable_meta=True,
                                                      x_disable_stats=True))
        if (self.run.settings.mode != 'offline' or
            not Path(self.run.dir).resolve().is_relative_to(self.backup_root.resolve())):
            self.run.finish(exit_code=1)
            raise ValueError('W&B did not use the requested offline recording directory')

    def poll(self):
        if self._closed:
            raise ValueError('Recorder is closed')
        rows = read_metrics(self.training_root/'training-metrics.jsonl', self.start_step, self.start_updates)
        data = ''.join(json.dumps(row)+'\n' for row in rows).encode()
        if not data.startswith(self._metrics_bytes):
            raise ValueError('Previously recorded training metrics were rewritten')
        if data != self._metrics_bytes:
            temporary = self.backup_root/f'.metrics-{uuid.uuid4().hex}'
            with temporary.open('xb') as stream:
                stream.write(data); stream.flush(); os.fsync(stream.fileno())
            temporary.replace(self.backup_root/'training-metrics.jsonl')
            self._metrics_bytes = data
        for row in rows:
            if row['step'] > self.last_step:
                self.run.log(row, step=row['step'])
                self.last_step, self.last_updates = row['step'], row['trainedUpdates']
        for source in sorted(self.training_root.glob('step-*')):
            if source.name not in self.backed_up:
                backup_checkpoint(source, self.backup_root/'checkpoints', self.expected)
                self.backed_up.add(source.name)
        return {'lastStep': self.last_step, 'trainedUpdates': self.last_updates,
                'backedUpCheckpoints': sorted(self.backed_up)}

    def close(self, exit_code=0):
        if not self._closed:
            self.run.summary.update({'recordedAttempts': self.last_step-self.start_step,
                                     'trainedUpdates': self.last_updates,
                                     'backedUpCheckpoints': sorted(self.backed_up),
                                     'trainingCompletion': 'unverified'})
            self.run.finish(exit_code=exit_code)
            self._closed = True
