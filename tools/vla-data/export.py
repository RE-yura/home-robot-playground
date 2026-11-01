"""Local calibration export only; never publishes or computes policy normalization."""
import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
LEROBOT_REVISION = 'ca69a2068462a37f7cdcb74180927a2f863d2bf7'
MAX_BYTES = 64 * 1024 * 1024


def installed_lerobot_revision():
    direct = importlib.metadata.distribution('lerobot').read_text('direct_url.json')
    return json.loads(direct or '{}').get('vcs_info', {}).get('commit_id')


def validate_packet(source):
    with source.open('rb') as file:
        raw = file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('Recording exceeds size limit')
    # Share the authoritative strict schema with the browser instead of drifting
    # camera calibration, travel limits or revision constants in a second language.
    script = """
import {validateLearningEpisode} from './src/learning/recording.ts';
let text='';for await(const part of process.stdin) text+=part;
try{validateLearningEpisode(JSON.parse(text));}catch(e){console.error(e.message);process.exitCode=1;}
"""
    checked = subprocess.run(['node', '--input-type=module', '-e', script], cwd=ROOT,
                             input=raw, capture_output=True)
    if checked.returncode:
        raise ValueError(checked.stderr.decode().strip())
    episode = json.loads(raw)
    if episode['error'] is not None:
        raise ValueError('Failed capture: ' + episode['error'])
    if not episode['samples']:
        raise ValueError('Empty calibration recording')
    for sample in episode['samples']:
        if any(abs(value) > float(np.finfo(np.float32).max) for key in ('state', 'action') for value in sample[key]):
            raise ValueError('Feature overflows float32')
    decoded = {}
    for key, frame in episode['images'].items():
        try:
            data = base64.b64decode(frame['dataUrl'].split(',')[1], validate=True)
            with Image.open(io.BytesIO(data)) as image:
                if image.format != 'JPEG' or image.mode != 'RGB' or image.size != (320, 240):
                    raise ValueError('Expected 320x240 RGB JPEG')
                pixels = np.array(image)
            decoded[key] = (data, pixels)
        except Exception as error:
            raise ValueError('Invalid image ' + key) from error
    return episode, decoded, hashlib.sha256(raw).hexdigest()


def export_episode(source: Path, output: Path):
    if output.exists():
        raise FileExistsError('Output already exists: ' + str(output))
    episode, decoded, source_hash = validate_packet(source)
    if installed_lerobot_revision() != LEROBOT_REVISION:
        raise ValueError('Installed LeRobot revision does not match the recording')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = episode['manifest']['axisNames']
    features = {'observation.state': {'dtype': 'float32', 'shape': (6,), 'names': names},
                'action': {'dtype': 'float32', 'shape': (6,), 'names': names}}
    for view in ('head', 'hand'):
        features[f'observation.images.{view}'] = {'dtype': 'image', 'shape': (3, 240, 320), 'names': ['channels', 'height', 'width']}
    dataset = LeRobotDataset.create('local/hsr-calibration', fps=10, features=features, root=output,
                                    robot_type=episode['manifest']['robotModelId'], use_videos=False, video_backend='pyav')
    try:
        for sample in episode['samples']:
            frame = {'observation.state': np.array(sample['state'], dtype=np.float32),
                     'action': np.array(sample['action'], dtype=np.float32), 'task': episode['instruction']}
            for view in ('head', 'hand'):
                frame[f'observation.images.{view}'] = decoded[sample['images'][view]][1]
            # Native API creates relative timestamp=index/fps and episode indexes.
            dataset.add_frame(frame)
        dataset.save_episode()
    finally:
        dataset.finalize()
    for key, (data, _) in decoded.items():
        destination = output / 'source-images' / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    metadata = {key: value for key, value in episode.items() if key != 'images'}
    sidecar = {'kind': 'calibration', 'normalizationUsable': False, 'sourceSha256': source_hash, 'episode': metadata,
               'imageSha256': {key: hashlib.sha256(data).hexdigest() for key, (data, _) in decoded.items()}}
    (output / 'hsr-calibration.json').write_text(json.dumps(sidecar, indent=2) + '\n')
    return sidecar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export_episode(args.input, args.output)
    print(json.dumps({'output': str(args.output), 'frames': len(result['episode']['samples']), 'kind': 'calibration'}))

if __name__ == '__main__':
    main()
