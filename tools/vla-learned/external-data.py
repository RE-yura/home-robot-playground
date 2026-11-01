"""Separate the unchanged FP32 ONNX weights and verify exact tensors/native parity.

Uses a fresh output; never rewrites the checkpoint, original graph or fixtures.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

GRAPH_HASH = '7ac9785cdafb5e763761769484c4b037f8d397d173475d048b9d7dcf85ede1ff'
GRAPH_BYTES = 1592644520
CHECKPOINT_HASH = 'dfa2e7db574031640480695d7ccc174e2a1416a997e80d4d963c5971824a6976'
WEIGHTS_NAME = 'hsr-smolvla.weights'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensors(graph):
    import onnx
    if graph.sparse_initializer:
        raise ValueError('Sparse initializers require separate identity verification')
    yield from graph.initializer
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                yield attribute.t
            elif attribute.type == onnx.AttributeProto.TENSORS:
                yield from attribute.tensors
            elif attribute.type == onnx.AttributeProto.GRAPH:
                yield from tensors(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from tensors(subgraph)


def tensor_identity(model):
    import onnx
    inventory = []
    for tensor in tensors(model.graph):
        array = onnx.numpy_helper.to_array(tensor)
        if array.dtype.hasobject:
            raise ValueError('Object-valued tensor requires explicit verification')
        inventory.append({'name': tensor.name, 'type': tensor.data_type,
                          'shape': list(tensor.dims), 'bytes': array.nbytes,
                          'sha256': hashlib.sha256(array.tobytes()).hexdigest()})
    encoded = json.dumps(inventory, separators=(',', ':'), sort_keys=True).encode()
    return {'sha256': hashlib.sha256(encoded).hexdigest(), 'count': len(inventory),
            'bytes': sum(t['bytes'] for t in inventory)}


def graph_identity_without_tensor_storage(model):
    # Call only when this loaded model is no longer needed for serialization.
    for tensor in tensors(model.graph):
        for field in ('raw_data', 'float_data', 'int32_data', 'string_data',
                      'int64_data', 'double_data', 'uint64_data', 'external_data'):
            tensor.ClearField(field)
        tensor.ClearField('data_location')
    return hashlib.sha256(model.SerializeToString(deterministic=True)).hexdigest()


def main(export, output, source):
    import numpy as np
    import onnx
    import onnxruntime as ort
    sys.path.insert(0, str(source))
    from onnx_export import parity_report
    started = time.monotonic()
    original = export / 'hsr-smolvla.onnx'
    native = json.loads((export / 'onnx-parity.json').read_text())
    if not native['passed'] or native['sourceCheckpointManifestSha256'] != CHECKPOINT_HASH or len(native['cases']) != 3:
        raise ValueError('Original verified learned export required')
    if original.stat().st_size != GRAPH_BYTES or digest(original) != GRAPH_HASH:
        raise ValueError('Original FP32 graph identity changed')
    files = {item['name']: item for item in native['files']}
    for index in range(3):
        path = export / f'case-{index}.npz'
        if path.stat().st_size != files[path.name]['bytes'] or digest(path) != files[path.name]['sha256']:
            raise ValueError('Original native fixture identity changed')
    if output.exists():
        raise FileExistsError('Use a fresh output and preserve previous attempts')
    output.mkdir(parents=True)
    target = output / 'hsr-smolvla.onnx'
    model = onnx.load(original)
    before_tensors = tensor_identity(model)
    onnx.save_model(model, target, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=WEIGHTS_NAME, size_threshold=1024, convert_attribute=False)
    before_graph = graph_identity_without_tensor_storage(model)
    del model
    gc.collect()
    metadata = onnx.load(target, load_external_data=False)
    locations = {entry.value for tensor in tensors(metadata.graph) for entry in tensor.external_data if entry.key == 'location'}
    if locations != {WEIGHTS_NAME}:
        raise ValueError('Unexpected external weight location')
    del metadata
    model = onnx.load(target, load_external_data=True)
    after_tensors = tensor_identity(model)
    after_graph = graph_identity_without_tensor_storage(model)
    del model
    gc.collect()
    if before_tensors != after_tensors or before_graph != after_graph:
        raise ValueError('External serialization changed graph operations or tensor bytes')
    onnx.checker.check_model(str(target), full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(target), sess_options=options, providers=['CPUExecutionProvider'])
    config = json.loads((export.parent.parent / 'training-config.json').read_text())
    normalization = config['expectedIdentity']['normalization']['action']
    mean, std = (np.asarray(normalization[key], dtype=np.float32) for key in ('mean', 'std'))
    cases = []
    for index in range(3):
        with np.load(export / f'case-{index}.npz', allow_pickle=False) as reference:
            feeds = {item.name: reference[item.name] for item in session.get_inputs()}
            begin = time.monotonic()
            actual = session.run(['hsr_action_chunk'], feeds)[0]
            physical = parity_report(reference['expected'], actual)
            normalized = parity_report((reference['expected'] - mean) / std, (actual - mean) / std)
            cases.append({'case': index, 'passed': physical['passed'] and normalized['passed'],
                          'physical': physical, 'normalized': normalized, 'comparedValues': 96,
                          'inferenceSeconds': time.monotonic() - begin})
    del session
    weights = output / WEIGHTS_NAME
    proof = {'schema': 'hsr-learned-external-data-v1', 'passed': all(c['passed'] for c in cases),
             'sourceGraph': {'bytes': GRAPH_BYTES, 'sha256': GRAPH_HASH},
             'outputGraph': {'bytes': target.stat().st_size, 'sha256': digest(target), 'format': 'onnx-external-data'},
             'externalData': [{'path': WEIGHTS_NAME, 'bytes': weights.stat().st_size, 'sha256': digest(weights)}],
             'precision': 'unchanged-FP32', 'unchangedTensorIdentity': before_tensors,
             'unchangedGraphStructureSha256': before_graph, 'provider': 'CPUExecutionProvider',
             'onnx': onnx.__version__, 'onnxruntime': ort.__version__, 'cases': cases, 'comparedValues': 288,
             'originalGraphPreserved': digest(original) == GRAPH_HASH, 'elapsedSeconds': time.monotonic() - started,
             'browserParity': 'unverified', 'learnedRolloutGate': 'unverified', 'publicationAllowed': False}
    (output / 'conversion-receipt.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof), flush=True)
    if not proof['passed'] or not proof['originalGraphPreserved']:
        raise ValueError('All unchanged physical/normalized tolerances must pass')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('export', 'output', 'source'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    main(args.export, args.output, args.source)
