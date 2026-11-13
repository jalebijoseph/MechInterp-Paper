"""Completion and allocation checks for the durable follow-up controller."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


class RunnerTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.spec_from_file_location('followup_controller', SCRIPTS/'queue_followup.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_waits_for_terminal_predecessor_and_shutdown_even_when_gate_failed(self):
        mod = self.module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/'source/runs/full-suite/manifest.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'status': 'running'}))
            (root/'manager-status.json').write_text(json.dumps({'status':'running'}))
            self.assertFalse(mod.predecessor_released(root)[0])
            path.write_text(json.dumps({'status': 'incomplete'}))
            (root/'manager-status.json').write_text(json.dumps({'status':'failed'}))
            self.assertFalse(mod.predecessor_released(root)[0])
            (root/'manager-status.json').write_text(json.dumps(
                {'status':'failed','servers_stopped_at':'recorded'}))
            ready, info = mod.predecessor_released(root)
            self.assertTrue(ready)
            self.assertEqual(info['manifest_status'], 'incomplete')

    def test_requires_four_idle_allowed_gpus_and_ignores_unrelated_pair(self):
        mod = self.module()
        gpus = '\n'.join(f'{n}, GPU-{n}, 0' for n in range(8))
        pairs = mod.allowed_pairs(gpus, 'GPU-4, 100\nGPU-7, 101')
        self.assertEqual([[r['index'] for r in p] for p in pairs], [[0,1],[2,3]])
        self.assertEqual(mod.allowed_pairs(gpus, 'GPU-0, 99'), [])
        self.assertEqual(mod.allowed_pairs(gpus.replace('2, GPU-2, 0','2, GPU-2, 300'), ''), [])

    def test_worker_rejects_partial_results_and_missing_saved_traces(self):
        from followup_worker import validate_run
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{'trace_file':f'traces/{i}.npz'} for i in range(32)]
            (root/'traces').mkdir()
            for i in range(32):
                (root/f'traces/{i}.npz').write_bytes(b'test fixture')
            turns = rows + [{'trace_file':None} for _ in range(64)]
            summary = {'reviews':[{}]*32,'frozen_notes':[{}]*16,
                       'transfer_probes':[{}]*32,'no_peer_probes':[{}]*16}
            self.assertEqual(validate_run('exp1', summary, turns, root), [])
            (root/'traces/0.npz').unlink()
            self.assertTrue(validate_run('exp1', summary, turns, root))
            self.assertTrue(validate_run('exp1', summary, turns[:-1], root))

    def test_context_preflight_counts_ids_not_batch_encoding_keys(self):
        import preflight_followup
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return {'input_ids':list(range(123)), 'attention_mask':[1]*123} if kwargs['tokenize'] else 'native chat'
            def __call__(self, text, **kwargs):
                assert text=='native chat' and kwargs['add_special_tokens'] is False
                return {'input_ids':list(range(123)), 'attention_mask':[1]*123}
        self.assertEqual(preflight_followup.prompt_length(Tokenizer(),[{'role':'user','content':'x'}]),123)


if __name__ == '__main__':
    unittest.main()
