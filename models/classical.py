# models/classical.py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def get_all_models():
    return {
        "LR": LogisticRegression(max_iter=2000),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(n_estimators=300),
        "AdaBoost": AdaBoostClassifier(n_estimators=200),
        "SVM": SVC(probability=True),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic"
        ),
    }
