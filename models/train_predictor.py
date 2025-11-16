# models/train_predictor.py
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from .classical import get_all_models


def train_and_evaluate(X_train, y_train, X_test, y_test):
    models = get_all_models()
    results = {}

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_test)[:, 1]

        results[name] = {
            "AUROC": roc_auc_score(y_test, prob),
            "AUPRC": average_precision_score(y_test, prob)
        }

    return results
