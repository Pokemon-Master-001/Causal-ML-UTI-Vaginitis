import pandas as pd
import numpy as np
from cd.runner import run_cd_methods
from features.build_causal_sets import merge_cd_results
from matching.ps_model import estimate_propensity
from matching.knn_matching import knn_1to1_match
from models.train_predictor import train_and_evaluate

df = pd.read_csv("data/processed/icu_data.csv")
X = df.drop(columns=["outcome"]).values
cd_results = run_cd_methods(X)

np.save("data/causal_features/cd_results.npy", cd_results)

cd_results = np.load("data/causal_features/cd_results.npy", allow_pickle=True).item()
df = pd.read_csv("data/processed/icu_data.csv")

features = df.columns.tolist()
y_idx = features.index("outcome")

causal_set = merge_cd_results(cd_results, features, y_idx)

pd.Series(causal_set).to_csv("data/causal_features/outcome_features.csv", index=False)

df = pd.read_csv("data/processed/cohort.csv")
cov = ["age", "gender", "diag", "surgery", "comorbidity"]

df_ps = estimate_propensity(df, "UVC", cov)
matched = knn_1to1_match(df_ps, treat_col="UVC")

matched.to_csv("data/matched/uvc_matched.csv", index=False)

train = pd.read_csv("data/matched/train.csv")
test = pd.read_csv("data/matched/test.csv")

y_train = train["outcome"]
X_train = train.drop(columns=["outcome"])

y_test = test["outcome"]
X_test = test.drop(columns=["outcome"])

results = train_and_evaluate(X_train, y_train, X_test, y_test)
print(results)
