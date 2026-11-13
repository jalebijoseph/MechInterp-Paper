"""CUDA visibility must follow physical GPU UUIDs, independently of ordinal order."""
import importlib.util
from pathlib import Path
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BindingTests(unittest.TestCase):
    def test_8b_binding_uses_physical_uuid_and_rejects_busy_devices(self):
        manager = load('cluster_manager')
        self.assertTrue(hasattr(manager, 'gpu_uuid_bindings'))
        rows = '0, GPU-second-cuda-ordinal, 0\n1, GPU-first-cuda-ordinal, 0\n2, GPU-busy, 400\n'
        self.assertEqual(manager.gpu_uuid_bindings(rows, [0, 1]),
                         {0: 'GPU-second-cuda-ordinal', 1: 'GPU-first-cuda-ordinal'})
        with self.assertRaisesRegex(RuntimeError, 'occupied'):
            manager.gpu_uuid_bindings(rows, [2])

    def test_exp0_binding_uses_selected_inventory_and_enforces_allowed_devices(self):
        queue = load('queue_exp0_32b')
        self.assertTrue(hasattr(queue, 'selected_gpu_uuids'))
        inventory = [{'index': i, 'uuid': f'GPU-physical-{i}', 'memory_used_mib': 0,
                      'has_compute_process': False} for i in range(8)]
        self.assertEqual(queue.selected_gpu_uuids([2, 0], inventory), ['GPU-physical-2', 'GPU-physical-0'])
        with self.assertRaises(ValueError):
            queue.selected_gpu_uuids([0, 4], inventory)
        inventory[2]['has_compute_process'] = True
        with self.assertRaises(ValueError):
            queue.selected_gpu_uuids([0, 2], inventory)


if __name__ == '__main__':
    unittest.main()
