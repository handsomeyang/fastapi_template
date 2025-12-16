import json
import joblib
from typing import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
import pandas as pd
from utils import get_artifacts_dir, encode_binary_features, TRAINING_FEATURES
from .models import CustomerData, HealthCheckResult, PredictionResult
import logging


logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    pipeline_path = get_artifacts_dir() / "best_ml_pipeline.joblib"

    if not pipeline_path.exists():
        logger.critical(
            f"FATAL: Model pipeline file not found at {pipeline_path}. Server cannot start."
        )
        raise SystemExit(f"Missing critical file: {pipeline_path}")

    try:
        app.state.ml_pipeline = joblib.load(pipeline_path)
        logger.info("Model loaded successfully. Server ready to start.")
    except Exception as e:
        logger.critical(
            f"FATAL: Failed to load model pipeline from {pipeline_path}.", exc_info=True
        )
        raise SystemExit(f"Failed to load model pipeline: {e}")

    training_features_path = get_artifacts_dir() / "training_features.json"
    try:
        with open(training_features_path, "r") as f:
            app.state.training_features = json.load(f)
    except Exception:
        logger.warning(
            f"WARN: Failed to load training features from {training_features_path}. Using default training features."
        )
        app.state.training_features = TRAINING_FEATURES

    binary_features_path = get_artifacts_dir() / "binary_features.json"
    try:
        with open(binary_features_path, "r") as f:
            app.state.binary_features = json.load(f)
    except Exception:
        logger.warning(
            f"WARN: Failed to load binary features from {binary_features_path}. Using default binary features."
        )
        app.state.binary_features = ["default", "housing", "loan"]

    yield

    if app.state.ml_pipeline is not None:
        app.state.ml_pipeline = None


app = FastAPI(
    title="Term subscription prediction API",
    description="Predicts whether a customer will subscribe to a term deposit",
    version="0.1.0",
    contact={"email": "handsomeyang@gmail.com"},
    lifespan=lifespan,
)


@app.get("/", response_model=HealthCheckResult)
def health_check() -> HealthCheckResult:
    return HealthCheckResult(status="OK")


@app.post("/predict")
async def predict_subscription(data: CustomerData) -> PredictionResult:
    data_dict = data.model_dump()

    logger.info(f"Prediction initiated for new request: {data.model_dump_json()}")

    try:
        # Use training_features persisted together with the trained pipeline to construct a data frame with the same
        # column order as the training data
        input_df = pd.DataFrame([data_dict], columns=app.state.training_features)
        encode_binary_features(input_df, app.state.binary_features)
        subscription_prob = app.state.ml_pipeline.predict_proba(input_df)[0][1]

        result = PredictionResult(
            status="Success", prediction="yes" if subscription_prob > 0.5 else "no"
        )

        logger.info(f"Prediction completed successfully: {result.model_dump_json()}")

        return result
    except Exception as e:
        logger.exception("ERROR: Prediction failed due to error")
        raise e
