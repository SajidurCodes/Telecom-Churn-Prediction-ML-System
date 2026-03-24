TARGET = "Churn"
RANDOM_STATE = 1717
N_SPLITS = 5

MODEL_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.0635534357133,
    "num_leaves": 58,
    "max_depth": 5,
    "feature_fraction": 0.9079883802664838,
    "bagging_fraction": 0.6598623439714022,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": RANDOM_STATE
}