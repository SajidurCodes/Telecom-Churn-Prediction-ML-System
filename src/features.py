import pandas as pd


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df.columns = [col.strip() for col in df.columns]

    # =========================
    # 1. Interaction Features
    # =========================
    if {"DataUsage", "ContractRenewal"}.issubset(df.columns):
        df["datausage_x_contract"] = df["DataUsage"] * df["ContractRenewal"]

    if {"DataUsage", "DataPlan"}.issubset(df.columns):
        df["datausage_x_plan"] = df["DataUsage"] * df["DataPlan"]

    # =========================
    # 2. Ratio Features
    # =========================
    if {"CustServCalls", "AccountWeeks"}.issubset(df.columns):
        df["custserv_per_week"] = df["CustServCalls"] / (df["AccountWeeks"] + 1)

    if {"DayMins", "DayCalls"}.issubset(df.columns):
        df["mins_per_call"] = df["DayMins"] / (df["DayCalls"] + 1)

    if {"MonthlyCharge", "DayMins"}.issubset(df.columns):
        df["charge_per_min"] = df["MonthlyCharge"] / (df["DayMins"] + 1)

    # =========================
    # 3. Usage-Based Features
    # =========================
    if {"DayMins", "DataUsage", "RoamMins"}.issubset(df.columns):
        df["total_usage"] = df["DayMins"] + df["DataUsage"] + df["RoamMins"]

    # =========================
    # 4. Cost Behavior Features
    # =========================
    if {"MonthlyCharge", "OverageFee"}.issubset(df.columns):
        df["cost_intensity"] = df["MonthlyCharge"] + df["OverageFee"]

    if {"OverageFee", "DayMins"}.issubset(df.columns):
        df["overage_per_min"] = df["OverageFee"] / (df["DayMins"] + 1)

    # =========================
    # 5. Binary Flags
    # =========================
    if "CustServCalls" in df.columns:
        df["high_custserv_calls"] = (df["CustServCalls"] >= 4).astype(int)

    if "OverageFee" in df.columns:
        df["high_overage"] = (df["OverageFee"] > df["OverageFee"].median()).astype(int)

    if "RoamMins" in df.columns:
        df["high_roaming"] = (df["RoamMins"] > df["RoamMins"].median()).astype(int)

    return df