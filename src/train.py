"""
train.py
--------
Model training, hyperparameter tuning, and MLflow experiment tracking.
Trains Logistic Regression and XGBoost models, logs all experiments,
and registers the best model in the MLflow Model Registry.

Task 5 implementation.
"""

import logging
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "is_high_risk"


def load_processed_data(path: str):
    """
    Load the processed parquet dataset and split into features and target.

    Args:
        path: Path to processed parquet file

    Returns:
        Tuple of (X, y) DataFrames

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the target column is missing
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found at: {path}")

    logger.info(f"Loading processed data from {path}")
    df = pd.read_parquet(path)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        logger.warning(f"Dropping remaining categorical columns: {cat_cols}")
        df = df.drop(columns=cat_cols)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    logger.info(
        f"Dataset: {X.shape[0]:,} rows x {X.shape[1]} features | "
        f"Target distribution: {y.value_counts(normalize=True).to_dict()}"
    )
    return X, y


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Compute classification metrics for a fitted model.

    Args:
        model:  Fitted sklearn-compatible model
        X_test: Test features
        y_test: True labels

    Returns:
        Dictionary of metric name -> value
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }

    logger.info("Evaluation metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k:12s}: {v}")

    logger.info(
        f"\nClassification Report:\n"
        f"{classification_report(y_test, y_pred, zero_division=0)}"
    )
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return metrics


def build_lr_pipeline() -> Pipeline:
    """Build a Logistic Regression pipeline with imputation and scaling."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight="balanced",
        )),
    ])


def build_xgb_pipeline() -> Pipeline:
    """Build an XGBoost pipeline with imputation."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        )),
    ])


LR_PARAM_GRID = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__penalty": ["l1", "l2"],
    "model__solver": ["liblinear"],
}

XGB_PARAM_GRID = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0],
}


def train_model(
    pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    experiment_name: str,
) -> tuple:
    """
    Train a model with GridSearchCV and log everything to MLflow.

    Returns:
        Tuple of (best_estimator, metrics_dict)
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name):

        logger.info(f"Training {model_name} with GridSearchCV...")

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("best_cv_roc_auc", round(grid_search.best_score_, 4))
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param.replace("model__", ""), value)

        metrics = evaluate_model(best_model, X_test, y_test)
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        if hasattr(best_model.named_steps.get("model"), "feature_importances_"):
            importances = best_model.named_steps["model"].feature_importances_
            feat_imp = pd.Series(importances, index=X_train.columns)
            feat_imp_path = "/tmp/feature_importance.csv"
            feat_imp.sort_values(ascending=False).to_csv(feat_imp_path)
            mlflow.log_artifact(feat_imp_path, artifact_path="feature_importance")
            logger.info(
                f"Top 10 features:\n"
                f"{feat_imp.sort_values(ascending=False).head(10)}"
            )

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run ID: {run_id}")

    return best_model, metrics


def train(
    processed_data_path: str,
    experiment_name: str = "credit-risk",
    mlflow_tracking_uri: str = "sqlite:///mlflow.db",
) -> str:
    """
    Full training workflow: load data, train models, compare, register best.

    Returns:
        Name of the best registered model
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow tracking URI: {mlflow_tracking_uri}")

    X, y = load_processed_data(processed_data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} | "
        f"Positive rate (train): {y_train.mean():.3f}"
    )

    results = {}

    lr_model, lr_metrics = train_model(
        pipeline=build_lr_pipeline(),
        param_grid=LR_PARAM_GRID,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        model_name="LogisticRegression",
        experiment_name=experiment_name,
    )
    results["LogisticRegression"] = {"model": lr_model, "metrics": lr_metrics}

    xgb_model, xgb_metrics = train_model(
        pipeline=build_xgb_pipeline(),
        param_grid=XGB_PARAM_GRID,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        model_name="XGBoost",
        experiment_name=experiment_name,
    )
    results["XGBoost"] = {"model": xgb_model, "metrics": xgb_metrics}

    logger.info("\n" + "="*50)
    logger.info("MODEL COMPARISON")
    logger.info("="*50)
    comparison = pd.DataFrame({
        name: data["metrics"] for name, data in results.items()
    }).T
    logger.info(f"\n{comparison.to_string()}")

    best_name = comparison["roc_auc"].idxmax()
    best_metrics = results[best_name]["metrics"]
    logger.info(
        f"\nBest model: {best_name} "
        f"(ROC-AUC: {best_metrics['roc_auc']:.4f})"
    )

    return best_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train credit risk models and log to MLflow"
    )
    parser.add_argument("--data", required=True, help="Path to processed parquet file")
    parser.add_argument("--experiment", default="credit-risk")
    parser.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()

    best = train(
        processed_data_path=args.data,
        experiment_name=args.experiment,
        mlflow_tracking_uri=args.mlflow_uri,
    )
    logger.info(f"Training complete. Best model: {best}")
