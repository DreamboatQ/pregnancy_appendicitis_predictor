import unittest

from app import app


COMPLETE_PATIENT = {
    "age": 29,
    "gestational_age": 18.9,
    "body_temperature": 36.7,
    "pain_duration": 24,
    "wbc": 13.87,
    "crp": 12.95,
    "neutrophils": 81.6,
    "diabetes": 0,
    "hypertension": 0,
    "placental_abnormalities": 0,
    "fever": 0,
    "rlq_effusion": 0,
    "primiparity": 1,
    "appendiceal_swelling": 1,
    "abscess": 0,
}


class AppTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_reports_expected_ensemble(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["model"], "Random Forest")
        self.assertEqual(response.json["models_loaded"], 20)

    def test_prediction_is_deterministic_and_shap_reconstructs_probability(self):
        first = self.client.post("/api/predict", json=COMPLETE_PATIENT)
        second = self.client.post("/api/predict", json=COMPLETE_PATIENT)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json["probability"], second.json["probability"])
        self.assertTrue(first.json["shap_plot"].startswith("data:image/png;base64,"))
        reconstructed = first.json["base_value"] + sum(
            item["shap"] for item in first.json["contributions"]
        )
        self.assertAlmostEqual(reconstructed, first.json["probability"], places=5)

    def test_missing_values_use_imputation(self):
        patient = dict(COMPLETE_PATIENT)
        patient["crp"] = None
        response = self.client.post("/api/predict", json=patient)
        self.assertEqual(response.status_code, 200)
        self.assertIn("CRP level", response.json["missing_fields"])

    def test_invalid_binary_value_is_rejected(self):
        patient = dict(COMPLETE_PATIENT)
        patient["fever"] = 2
        response = self.client.post("/api/predict", json=patient)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Fever", response.json["error"])


if __name__ == "__main__":
    unittest.main()
