from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from src.circuit_artifacts import write_json
from src.diversity_campaign import DEFAULT_PROTOCOL, load_protocol, resume_mapping, validate


class CampaignTest(TestCase):
    def test_all_32_compositions(self):
        self.assertEqual(validate(load_protocol(DEFAULT_PROTOCOL))["tasks"], 32)

    def test_resume_updates_parent_status_and_rejects_incomplete_sampling(self):
        with TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            protocol = load_protocol(DEFAULT_PROTOCOL)
            status = {"training_complete": True, "sampling_complete": False, "protocol": protocol}
            write_json(attempt / "status.json", status)
            with self.assertRaises(ValueError):
                resume_mapping(attempt, Path("abc"))
            status["sampling_complete"] = True
            status["error"] = "old failure"
            write_json(attempt / "status.json", status)
            with mock.patch("src.diversity_evaluation.evaluate", return_value={"complete": True}):
                result = resume_mapping(attempt, Path("abc"))
            self.assertTrue(result["complete"])
            self.assertTrue(result["mapping_complete"])
            self.assertNotIn("error", result)
            with mock.patch("src.diversity_evaluation.evaluate", side_effect=ValueError("identity mismatch")):
                with self.assertRaises(ValueError):
                    resume_mapping(attempt, Path("abc"))
