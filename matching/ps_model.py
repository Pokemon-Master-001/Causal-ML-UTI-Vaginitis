# matching/ps_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression


def estimate_propensity(df, treatment_col, covariates):
    """
    Estimate propensity scores using logistic regression.
    """
    X = df[covariates]
    y = df[treatment_col]

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)

    ps = lr.predict_proba(X)[:, 1]
    df = df.copy()
    df["ps"] = ps
    return df
