"""CPU-only protocol contracts for the runnable Experiment 5 pipeline."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jpeers.exp5 import protocol


class ProtocolTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "configs/exp5_study.json").read_text(encoding="utf-8"))

    def test_role_and_branch_seeds_are_independent(self):
        self.assertEqual(protocol.derived_seed(7, "main", "A", 2),
                         protocol.derived_seed(7, "main", "A", 2))
        seeds = {protocol.derived_seed(7, part) for part in ["main", "audit", "donor", "diagnostic"]}
        self.assertEqual(len(seeds), 4)
        self.assertEqual(len(protocol.CELL_ORDER), 4)

    def test_protocol_and_representable_schedules(self):
        config = self.config()
        protocol.validate_protocol(config)
        for section, key, value in [("model", "revision", "main"),
                                    ("inference", "enable_thinking", True),
                                    ("tasks", "rounds", 7),
                                    ("storage", "total_bytes", 1073741825)]:
            changed = copy.deepcopy(config)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                protocol.validate_protocol(changed)
        changed = copy.deepcopy(config)
        changed["delegation_choices"] = ["A", "B", "abstain"]
        with self.assertRaises(ValueError):
            protocol.validate_protocol(changed)

    def test_strict_response_contract(self):
        self.assertEqual(protocol.parse_choice('{"choice":"B"}'), "B")
        for value in ['{"choice":"abstain"}', '{"choice":"A","choice":"B"}',
                      '{"choice":"A","extra":true}', 'A', '{}']:
            self.assertIsNone(protocol.parse_choice(value))

    def test_protocol_hash_preserves_science_but_excludes_limits(self):
        config = self.config()
        changed = copy.deepcopy(config)
        changed["limits"]["max_model_calls"] = 500
        self.assertEqual(protocol.protocol_hash(config), protocol.protocol_hash(changed))
        changed["inference"]["peer"]["temperature"] = 0.5
        self.assertNotEqual(protocol.protocol_hash(config), protocol.protocol_hash(changed))


if __name__ == "__main__":
    unittest.main()
