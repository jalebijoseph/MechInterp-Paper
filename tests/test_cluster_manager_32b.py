"""Verify that the 32B lifecycle cannot select or address the other project's GPUs."""
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('manager32b', SCRIPTS / 'cluster_manager_32b.py')
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


class AllocationTests(unittest.TestCase):
    def test_pairs_use_only_idle_physical_zero_through_three(self):
        self.assertTrue(hasattr(manager, 'select_gpu_pairs'))
        inventory = '\n'.join(f'{i}, GPU-{i}, 0' for i in range(8))
        pairs = manager.select_gpu_pairs(inventory, '')
        self.assertEqual([[r['index'] for r in pair] for pair in pairs], [[0, 1], [2, 3]])
        pairs = manager.select_gpu_pairs(inventory, 'GPU-1, 999\n')
        self.assertEqual([[r['index'] for r in pair] for pair in pairs], [[0, 2]])
        self.assertEqual(manager.select_gpu_pairs(inventory, 'GPU-0, 998\nGPU-1, 999\nGPU-2, 997\n'), [])

    def test_server_visibility_uses_uuids_instead_of_cuda_ordinals(self):
        self.assertTrue(hasattr(manager, 'server_environment'))
        env = manager.server_environment({'EXISTING': 'value'}, [{'index': 2, 'uuid': 'GPU-two'},
                                                               {'index': 3, 'uuid': 'GPU-three'}])
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-two,GPU-three')
        self.assertEqual(env['JPEERS_MODEL'], 'qwen3-32b')
        self.assertEqual(env['EXISTING'], 'value')


if __name__ == '__main__':
    unittest.main()
