import unittest

from evaluation.metrics import evaluate, label_to_100, score_100_to_native


class MetricsTests(unittest.TestCase):
    def test_koniq_scale_round_trip(self):
        self.assertAlmostEqual(label_to_100(1.0, 1.0, 5.0), 0.0)
        self.assertAlmostEqual(label_to_100(5.0, 1.0, 5.0), 100.0)
        self.assertAlmostEqual(score_100_to_native(50.0, 1.0, 5.0), 3.0)

    def test_spaq_scale_is_identity(self):
        self.assertAlmostEqual(label_to_100(72.5, 0.0, 100.0), 72.5)
        self.assertAlmostEqual(score_100_to_native(72.5, 0.0, 100.0), 72.5)

    def test_perfect_metrics(self):
        result = evaluate([0.0, 50.0, 100.0], [1.0, 3.0, 5.0], 1.0, 5.0)
        self.assertEqual(result["N"], 3)
        self.assertAlmostEqual(result["SRCC"], 1.0)
        self.assertAlmostEqual(result["MAE_100"], 0.0)
        self.assertAlmostEqual(result["MAE_native"], 0.0)

    def test_invalid_scale_fails(self):
        with self.assertRaises(ValueError):
            label_to_100(1.0, 5.0, 5.0)


if __name__ == "__main__":
    unittest.main()
