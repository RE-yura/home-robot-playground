"""Compile an external-data FP32 learned graph into the two browser graphs.

1. fold_fixed_trig: precompute constant double Sin/Cos chains (the c802 graph).
2. shared_ensemble: one four-member graph sharing the observation prefix (the 8785 graph).
Both keep every initializer, so the same hsr-smolvla.weights file (copied into the output; the ONNX checker refuses links) serves all three graphs.
Usage: python compile-browser-graph.py <dir with hsr-smolvla.onnx> <fresh output dir>
"""
import hashlib, json, shutil, sys
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fold_fixed_trig import fold_fixed_trig
from shared_ensemble import share_ensemble


def compile_graphs(source, output):
    original = onnx.load(source / 'hsr-smolvla.onnx', load_external_data=False)
    compiled, folded = fold_fixed_trig(original)
    if folded <= 0:
        raise ValueError('Expected fixed trig chains to fold')
    for field in ('initializer', 'input', 'output'):
        if [x.SerializeToString() for x in getattr(original.graph, field)] != [x.SerializeToString() for x in getattr(compiled.graph, field)]:
            raise ValueError(f'Folding changed graph {field}')
    shared, proof = share_ensemble(compiled)
    output.mkdir(parents=True)
    shutil.copyfile(source / 'hsr-smolvla.weights', output / 'hsr-smolvla.weights')
    result = {'schema': 'hsr-browser-graph-compilation-v1', 'source': {'sha256': hashlib.sha256((source / 'hsr-smolvla.onnx').read_bytes()).hexdigest()},
              'foldedFixedTrigChainNodes': folded, 'sharedEnsemble': proof}
    for name, model in (('hsr-smolvla.onnx', compiled), ('hsr-smolvla-shared.onnx', shared)):
        data = model.SerializeToString()
        (output / name).write_bytes(data)
        onnx.checker.check_model(str(output / name))
        result[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    (output / 'compilation-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(compile_graphs(Path(sys.argv[1]), Path(sys.argv[2])), indent=1))
