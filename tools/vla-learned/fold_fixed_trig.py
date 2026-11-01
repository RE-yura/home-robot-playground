"""Precompute literal double trig and its fixed concatenation/float cast."""
import copy

import numpy as np
import onnx
from onnx import helper, numpy_helper


def fold_fixed_trig(model):
    optimized = copy.deepcopy(model)
    constants = {}
    for node in model.graph.node:
        if node.op_type == 'Constant' and not node.domain:
            values = [attr.t for attr in node.attribute if attr.name == 'value']
            if len(values) == 1 and values[0].data_type == onnx.TensorProto.DOUBLE:
                constants[node.output[0]] = numpy_helper.to_array(values[0])

    computed = {}
    count = 0
    for node in optimized.graph.node:
        if node.domain:
            continue
        if node.op_type in ('Sin', 'Cos') and node.input[0] in constants:
            operation = np.sin if node.op_type == 'Sin' else np.cos
            value = operation(constants[node.input[0]])
        elif node.op_type == 'Concat' and all(name in computed for name in node.input):
            axis = next(attr.i for attr in node.attribute if attr.name == 'axis')
            value = np.concatenate([computed[name] for name in node.input], axis=axis)
        elif node.op_type == 'Cast' and node.input[0] in computed and any(
                attr.name == 'to' and attr.i == onnx.TensorProto.FLOAT for attr in node.attribute):
            value = computed[node.input[0]].astype(np.float32)
        else:
            continue
        computed[node.output[0]] = value
        node.CopyFrom(helper.make_node('Constant', [], list(node.output), name=node.name,
                                      value=numpy_helper.from_array(value)))
        count += 1
    return optimized, count
