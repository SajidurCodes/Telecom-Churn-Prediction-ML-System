import json
import joblib
import os
import pandas as pd

from features import create_features


def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    model_path = os.path.join(base_dir, "../src/Models/lightgbm_churn_model.pkl")
    metadata_path = os.path.join(base_dir, "../src/Models/metadata.json")

    model = joblib.load(model_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return model, metadata


def predict(input_path: str, output_path: str = "predictions.csv"):
    model, metadata = load_artifacts()

    df = pd.read_csv(input_path)
    df.columns = [col.strip() for col in df.columns]

    df = create_features(df)

    feature_names = metadata["features"]
    X = df[feature_names]

    churn_probability = model.predict(X)
    churn_prediction = (churn_probability >= 0.5).astype(int)

    result = df.copy()
    result["churn_probability"] = churn_probability
    result["churn_prediction"] = churn_prediction

    result.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    input_path = os.path.join(base_dir, "../Data/telecom_churn.csv")
    output_path = os.path.join(base_dir, "../predictions.csv")
    predict(input_path, output_path)