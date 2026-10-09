
import json
import os
import unittest

import joblib
import pandas as pd


class TestIrisMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(os.path.exists("iris.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("iris_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        model = joblib.load("iris_model.pkl")

        data = pd.read_csv("iris.csv")
        features = data.drop(columns=["species"])

        sample = features.iloc[[0]]
        prediction = model.predict(sample)[0]

        self.assertIn(
            prediction,
            data["species"].unique()
        )

    def test_training_records(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertEqual(metrics["training_records"], 120)

    def test_testing_records(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertEqual(metrics["testing_records"], 30)


if __name__ == "__main__":
    unittest.main()
