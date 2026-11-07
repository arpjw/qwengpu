"""Check the assembly boundary before uploading programs to Qwen."""

import unittest

from qwengpu.assemble import assemble


class AssembleTests(unittest.TestCase):
    def test_preserves_warp_numbers_and_launch_data(self) -> None:
        launch = assemble(
            ".warp 1\nhalt\n.warp 0\nadd32i 0x1f # increment\nHALT\n",
            {"row_pairs": 1},
        )
        self.assertEqual(launch["row_pairs"], 1)
        self.assertEqual(
            launch["programs"],
            [[{"op": "ADD32I", "arg": 31}, {"op": "HALT"}], [{"op": "HALT"}]],
        )

    def test_rejects_invalid_programs_before_loading(self) -> None:
        for source in (
            "HALT\n.warp 0\nHALT\n.warp 1\nHALT",
            ".warp 0\nHALT\n.warp 0\nHALT",
            ".warp 0\nHALT",
            ".warp 0\nADD32I xyz\nHALT\n.warp 1\nHALT",
            ".warp 0\nWHILE_BIT 31\nHALT\n.warp 1\nHALT",
            ".warp 0\nADD32I 256\nHALT\n.warp 1\nHALT",
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                assemble(source)

    def test_rejects_ambiguous_program_sources(self) -> None:
        with self.assertRaises(ValueError):
            assemble(".warp 0\nHALT\n.warp 1\nHALT", {"programs": []})

    def test_global_launch_uses_explicit_workgroup_descriptor(self) -> None:
        source = ".warp 0\nGLOBAL_ID\nSHL32I 1\nHALT\n.warp 1\nLANE_ID\nHALT"
        self.assertEqual(assemble(source, {"workgroups": 3})["workgroups"], 3)
        for data in (
            {},
            {"workgroups": 0},
            {"workgroups": 5},
            {"workgroups": True},
            {"workgroups": 2, "global_words": [0]},
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                assemble(source, data)

    def test_global_launch_rejects_tile_ops_and_invalid_immediates(self) -> None:
        for op in ("LOAD_TILE", "SHL32I 8", "MASK_LANES 9"):
            with self.subTest(op=op), self.assertRaises(ValueError):
                assemble(f".warp 0\n{op}\nHALT\n.warp 1\nHALT", {"workgroups": 2})
