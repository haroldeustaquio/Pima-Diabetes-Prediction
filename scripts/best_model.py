import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from importlib import reload
import functions_model_eval_val as f
reload(f)
import joblib



df = pd.read_csv('../data/processed/data_remove_yes_balance_yes_scale_no.csv')

x_train, x_test, y_train, y_test = train_test_split(df.drop(columns='outcome'), df['outcome'], test_size=0.2, random_state=42)

param_grids = {
    "LogisticRegression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [100, 200, 300],
    },
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "leaf_size": [20, 30, 40]
    },
    "DecisionTreeClassifier": {
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy", "log_loss"],
        "splitter": ["best", "random"],
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [None, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy", "log_loss"],
    },
    "GaussianNB": {
        "var_smoothing": [1e-9, 1e-8, 1e-7]
    },
    "BaggingClassifier": {
        "n_estimators": [10, 50, 100],
        "max_samples": [0.5, 0.75, 1.0],
        "max_features": [0.5, 1.0],
        "bootstrap": [True, False]
    },
    "GradientBoostingClassifier": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0]
    },
    "AdaBoostClassifier": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.5, 1.0, 1.5]
    },
    "XGBClassifier": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "scale_pos_weight": [1, 2, 5, 10]  # Para clases desbalanceadas
    },
    "LGBMClassifier": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 63],
        "boosting_type": ["gbdt", "dart"],
        "feature_fraction": [0.8, 1.0],
    }
}

models = {
    "LogisticRegression": LogisticRegression(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "GaussianNB": GaussianNB(),
    "BaggingClassifier": BaggingClassifier(
        estimator=DecisionTreeClassifier()
    ),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(
        estimator=DecisionTreeClassifier()
    ),
    "XGBClassifier": XGBClassifier(),
    "LGBMClassifier": LGBMClassifier(verbose=-1)
}

classification_metrics_pro, proba_predictions_dict_pro, models_pro = f.classification_hyperparameter_tuning(x_train, y_train, x_test, y_test, models, param_grids, n_iter=100, cv=5, scoring='recall')

# Verified in notebools/03_model_optimization.ipynb

best_model = models_pro['RandomForestClassifier Pro']

joblib.dump(best_model, '../models/best_model.pkl')

print("Best model saved successfully.")