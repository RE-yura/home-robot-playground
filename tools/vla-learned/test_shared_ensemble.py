"""share_ensemble keeps the executor's action prefix as an observation-side input shared by all four members."""
import unittest

import numpy as np
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

from shared_ensemble import share_ensemble

BASE = ['head_rgb', 'hand_rgb', 'language_tokens', 'language_mask', 'hsr_state', 'initial_noise']


def toy(prefix_inputs):
    names = BASE + prefix_inputs
    nodes = [helper.make_node('Add', ['head_rgb', 'hsr_state'], ['observation']),
             helper.make_node('Add', ['observation', 'initial_noise'], ['denoised'])]
    if prefix_inputs:
        nodes += [helper.make_node('Mul', ['action_prefix', 'prefix_mask'], ['kept']),
                  helper.make_node('Add', ['denoised', 'kept'], ['hsr_action_chunk'])]
    else:
        nodes += [helper.make_node('Identity', ['denoised'], ['hsr_action_chunk'])]
    graph = helper.make_graph(nodes, 'toy', [helper.make_tensor_value_info(n, TensorProto.FLOAT, [2]) for n in names],
                              [helper.make_tensor_value_info('hsr_action_chunk', TensorProto.FLOAT, [2])])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])


class SharedEnsembleTests(unittest.TestCase):
    def test_prefix_inputs_are_shared_by_the_four_members(self):
        shared, proof = share_ensemble(toy(['action_prefix', 'prefix_mask']))
        self.assertEqual({x.name for x in shared.graph.input}, set(BASE) | {'action_prefix', 'prefix_mask', 'second_noise'})
        self.assertEqual(sum(n.op_type == 'Mul' for n in shared.graph.node), 1)
        self.assertEqual(proof['members'], 4)
        f = lambda *v: np.array(v, dtype=np.float32)
        feed = dict(head_rgb=f(1, 2), hand_rgb=f(0, 0), language_tokens=f(0, 0), language_mask=f(0, 0), hsr_state=f(10, 20),
                    initial_noise=f(.5, -.5), second_noise=f(.25, .75), action_prefix=f(3, 4), prefix_mask=f(1, 0))
        outputs = ReferenceEvaluator(shared).run(None, feed)
        base = f(11, 22) + f(3, 0)
        for actual, noise in zip(outputs, [f(.5, -.5), f(-.5, .5), f(.25, .75), f(-.25, -.75)]):
            np.testing.assert_array_equal(actual, base + noise)

    def test_original_six_input_graph_is_unchanged_behaviour(self):
        shared, _ = share_ensemble(toy([]))
        self.assertEqual({x.name for x in shared.graph.input}, set(BASE) | {'second_noise'})

    def test_partial_prefix_inputs_are_refused(self):
        with self.assertRaises(ValueError):
            share_ensemble(toy(['action_prefix']))


if __name__ == '__main__':
    unittest.main()
