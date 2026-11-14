"""Check that layer-at-a-time emission preserves the merged chip weights."""

import unittest

import torch
from chipformer.q35.emit import allocate, emit_layers
from chipformer.q35.ir import Graph
from chipformer.q35.tiny import BASE, pick_layout

from qwengpu.merge9b import DST_TEXT, emit_chip_bf16


class MergeWeightsTests(unittest.TestCase):
    def test_streamed_weights_equal_full_emission(self):
        graph = Graph()
        source = graph.token_feature("input", lambda token: token & 1)
        graph.token_feature("spare", lambda token: (token >> 1) & 1)
        value = graph.ffn([(1, source, graph.one)], "first")
        read, = graph.head("read", {0: value}, {0: graph.one}, {0: value},
                           graph.one, {0: graph.one}, [(0, 1)], min_level=6)
        result = graph.ffn([(1, read, graph.one)], "exit.result")
        keep = set(result.terms)
        allocation = allocate(graph, None, keep, 64)
        layout = pick_layout(allocation.n_slots + 1)
        layout.slots = allocation.slots
        config = dict(BASE, hidden_size=layout.hidden, intermediate_size=64,
                      linear_num_key_heads=1, linear_num_value_heads=1)
        self.assertTrue(allocation.subtract)
        self.assertTrue(allocation.clears)
        self.assertGreater(graph.n_layers(), 2)
        levels = [node.level for node in graph.nodes]

        expected = emit_layers(graph, layout, config, 0, DST_TEXT, allocation, keep)
        actual = dict(emit_chip_bf16(graph, layout, config, allocation, keep))

        self.assertEqual(actual.keys(), expected.keys())
        self.assertEqual([node.level for node in graph.nodes], levels)
        for name, weight in expected.items():
            self.assertTrue(torch.equal(actual[name], weight.to(torch.bfloat16)), name)
