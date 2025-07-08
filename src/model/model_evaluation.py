import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os

# MLflow and DagsHub tracking setup
mlflow.set_tracking_uri('https://dagshub.com/neerabhi/mlops_mini_project.mlflow')
dagshub.init(repo_owner='neerabhi', repo_name='mlops_mini_project', mlflow=True)
mlflow.set_experiment("dvc-pipeline")

# Logging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('Model file not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Error loading model: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Test data loaded from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading test data: %s', e)
        raise


def evaluate_model(clf, X_test, y_test) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba)
        }

        logger.debug('Model evaluated successfully.')
        return metrics
    except Exception as e:
        logger.error('Error during evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error saving metrics: %s', e)
        raise


def save_model_info(run_id: str, model_path: str, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        model_info = {
            "run_id": run_id,
            "model_uri": f"runs:/{run_id}/{model_path}"
        }
        with open(file_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        logger.debug("Model info saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving model info: %s", e)
        raise


def main():
    try:
        with mlflow.start_run() as run:
            clf = load_model('models/model.pkl')
            test_df = load_data('data/processed/test_bow.csv')

            X_test = test_df.iloc[:, :-1].values
            y_test = test_df.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)
            save_metrics(metrics, 'reports/metrics.json')
            print("Model evaluation completed successfully.")

            save_metrics(run.info.run_id, 'reports/experiment_info.json')
            print("Model evaluation completed successfully.")

            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log model parameters if available
            if hasattr(clf, 'get_params'):
                for k, v in clf.get_params().items():
                    mlflow.log_param(k, v)

            # Log model
            mlflow.sklearn.log_model(clf, "model")

            # # Save and log model info
            # print("Saving model info...")

            # save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')

            # print("Model info saved successfully.")

            # Log artifacts to MLflow
            mlflow.log_artifact('reports/metrics.json')
            mlflow.log_artifact('reports/experiment_info.json')
            mlflow.log_artifact('model_evaluation_errors.log')

    except Exception as e:
        logger.error("Model evaluation failed: %s", e)
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
