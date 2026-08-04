import unittest

from router.tool_selector import (
    compute_evidence_anchor,
    dimension_score,
    fuse_scores,
    sanitize_distortions,
    select_evidence_types,
    select_rule_profile,
)


class RouterTests(unittest.TestCase):
    def test_distortion_allowlist(self):
        values = sanitize_distortions(["Noise", "dataset:koniq", "Noise", 42, "Blurs"])
        self.assertEqual(values, ["Noise", "Blurs"])

    def test_profile_and_evidence_selection(self):
        self.assertEqual(select_rule_profile(["Compression"]), "compression")
        selected = select_evidence_types(["Compression"])
        self.assertEqual(selected, ["detail", "gradient", "noise"])
        self.assertEqual(select_evidence_types([]), ["detail"])

    def test_missing_brisque_is_renormalized(self):
        evidence = {
            "global_sharpness": 80.0,
            "local_sharpness": 70.0,
            "brisque_quality": None,
            "noise_severity": 20.0,
            "exposure_quality": 90.0,
            "blockiness_quality": 85.0,
        }
        score, weights, profile = compute_evidence_anchor(evidence, [])
        self.assertEqual(profile, "general")
        self.assertTrue(0.0 <= score <= 100.0)
        self.assertNotIn("brisque_quality", weights)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_fusion_is_bounded(self):
        score, weights = fuse_scores(20.0, 120.0, 90.0, 1.0)
        self.assertTrue(0.0 <= score <= 100.0)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_dimension_score_renormalizes_missing_values(self):
        score, weights = dimension_score(
            {"sharpness": 80, "noise_cleanliness": 60, "exposure": "invalid"}
        )
        self.assertAlmostEqual(score, 70.0)
        self.assertEqual(set(weights), {"sharpness", "noise_cleanliness"})


if __name__ == "__main__":
    unittest.main()
