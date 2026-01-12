import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("autism_screening (1).csv")

# Rename column
df.rename(columns={"austim": "family_mem_with_ASD"}, inplace=True)

features = [
    "A1_Score","A2_Score","A3_Score","A4_Score","A5_Score",
    "A6_Score","A7_Score","A8_Score","A9_Score","A10_Score",
    "age","gender","family_mem_with_ASD"
]

X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df["Class/ASD"].map({"NO": 0, "YES": 1}).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "logistic_regression.pkl": LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="liblinear"
    ),
    "random_forest.pkl": RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
    ),
    "xgboost.pkl": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    )
}

for filename, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, filename)
    print(f"Saved {filename}")
