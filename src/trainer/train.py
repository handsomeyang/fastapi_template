import json
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
import joblib
from utils import (
    get_data_dir,
    get_artifacts_dir,
    encode_binary_features,
    numerical_features,
    categorical_features,
    binary_features,
)
from colorama import init, Fore, Style


init()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the term subscription prediction model."
    )
    parser.add_argument(
        "--hyper-tune-iter",
        type=int,
        default=100,
        help="Number of iterations of hyperparameter tuning.",
    )
    parser.add_argument("--cv-fold", type=int, default=5, help="Number of CV folds.")

    args = parser.parse_args()

    print(Fore.CYAN + "========== Creating data pipeline ==========" + Style.RESET_ALL)

    print("Loading dataset.csv")
    df = pd.read_csv(get_data_dir() / "dataset.csv", sep=";")

    with open(get_artifacts_dir() / "binary_features.json", "w") as f:
        json.dump(binary_features, f)

    print("Encoding binary features")
    encode_binary_features(df, binary_features + ["y"])

    print("Creating data processor (numerical features + categorical features)")
    # Define data processor:
    # Z-score scaling for numerical features
    # One-Hot Encoding for categorical features
    # Pass binary encoding through for binary features
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    # The problem is essentially an unbalanced binary classification problem
    # Use class weighting to put more weights on the negative samples
    positive_count = df["y"].sum()
    negative_count = len(df) - positive_count
    scale_weight = negative_count / positive_count
    print(
        f"Positive count: {positive_count}, negative count: {negative_count}, neg/pos: {scale_weight}"
    )

    print(Fore.CYAN + "========== Creating model pipeline ==========" + Style.RESET_ALL)

    print("Creating model")
    xgb_model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=42
    )

    print(
        Fore.CYAN
        + "========== Creating final pipeline (data + model) =========="
        + Style.RESET_ALL
    )
    cv_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", xgb_model)]
    )

    print(
        Fore.CYAN
        + "========== Running hyperparameter tuning via stratified CV =========="
        + Style.RESET_ALL
    )

    X = df.drop(columns=["y"])
    y = df["y"]

    with open(get_artifacts_dir() / "training_features.json", "w") as f:
        json.dump(X.columns.to_list(), f)

    print("Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Creating parameter distributions")
    param_distributions = {
        "classifier__n_estimators": randint(100, 1000),
        "classifier__learning_rate": uniform(loc=0.01, scale=0.29),  # Range 0.01 to 0.3
        "classifier__max_depth": randint(3, 10),
        "classifier__subsample": uniform(loc=0.6, scale=0.4),  # Range 0.6 to 1.0
        "classifier__colsample_bytree": uniform(loc=0.6, scale=0.4),  # Range 0.6 to 1.0
        "classifier__gamma": uniform(loc=0, scale=0.5),  # Range 0 to 0.5
        "classifier__scale_pos_weight": uniform(
            loc=scale_weight * 0.9,
            scale=scale_weight * 0.2,  # Range scale_weight * 0.9 to scale_weight * 1.1
        ),
    }

    print("Creating CV strategy")
    k_folds = args.cv_fold
    cv_strategy = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    print("Creating randomized search")
    n_iter = args.hyper_tune_iter
    random_search = RandomizedSearchCV(
        estimator=cv_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv_strategy,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    print("Running randomized search")
    random_search.fit(X_train, y_train)

    best_roc_auc_score = random_search.best_score_
    best_params = random_search.best_params_
    best_pipeline = random_search.best_estimator_

    print(Fore.YELLOW + f"Best ROC AUC score: {best_roc_auc_score}" + Style.RESET_ALL)
    print(Fore.YELLOW + "Best performing parameters:" + Style.RESET_ALL)
    for k, v in best_params.items():
        print(Fore.YELLOW + f"  {k}: {v}" + Style.RESET_ALL)

    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    final_roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(
        Fore.YELLOW
        + f"ROC AUC score of best model on holdout test set: {final_roc_auc}"
        + Style.RESET_ALL
    )

    print("Saving pipeline (data + best performing model)")
    joblib.dump(best_pipeline, get_artifacts_dir() / "best_ml_pipeline.joblib")


if __name__ == "__main__":
    main()
