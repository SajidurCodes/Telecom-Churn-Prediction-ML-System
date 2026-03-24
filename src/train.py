import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from config import TARGET, RANDOM_STATE, N_SPLITS, MODEL_PARAMS
from features import create_features

def load_data(path: str) -> pd.DataFrame:
    print(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist. Please check the file path.")
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    df[TARGET] = df[TARGET].astype(int)
    return df


def run_cv(df: pd.DataFrame):
    df = create_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    neg = (y == 0).sum()
    pos = (y == 1).sum()
    scale_pos_weight = neg / pos

    params = MODEL_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos_weight

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(X))
    fold_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n===== Fold {fold} =====")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params=params,
            train_set=train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(100)
            ]
        )

        preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = preds

        fold_auc = roc_auc_score(y_val, preds)
        fold_scores.append(fold_auc)
        models.append(model)

        print(f"Fold {fold} AUC: {fold_auc:.6f}")

    final_auc = roc_auc_score(y, oof_preds)

    print("\n===== CV Summary =====")
    print("Fold scores:", [round(x, 6) for x in fold_scores])
    print("Mean AUC:", round(np.mean(fold_scores), 6))
    print("Final OOF AUC:", round(final_auc, 6))

    return models, X.columns.tolist(), final_auc, fold_scores, params


def train_full_model(df: pd.DataFrame, params: dict):
    df = create_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    full_data = lgb.Dataset(X, label=y)
    model = lgb.train(
        params=params,
        train_set=full_data,
        num_boost_round=300
    )

    return model, X.columns.tolist()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    data_path = os.path.join(base_dir, "../Data/telecom_churn.csv")  # Adjust path

    os.makedirs("Models", exist_ok=True)

    df = load_data(data_path)

    _, feature_names, final_auc, fold_scores, params = run_cv(df)
    final_model, feature_names = train_full_model(df, params)

    joblib.dump(final_model, "Models/lightgbm_churn_model.pkl")

    metadata = {
        "target": TARGET,
        "features": feature_names,
        "final_cv_auc": final_auc,
        "fold_scores": fold_scores,
        "params": params
    }

    with open("Models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nSaved model to Models/lightgbm_churn_model.pkl")
    print("Saved metadata to Models/metadata.json")


if __name__ == "__main__":
    main()