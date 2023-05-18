from abc import ABCMeta, abstractclassmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
import joblib

from autoML_service.preparers.preparer import DataPreparer
from autoML_service.constants import APP_PATH, MODELS_PATH


class AbstractAutoMLFramework(metaclass=ABCMeta):
    def __init__(self, task: str) -> None:
        # The framework name should be overwriten in the child class.
        # The package name must be the same as the one listed in pip freeze.
        self.framework = "AbrstractFramework"
        self.package = "AbstractFrameworkPackage"
        self.task = task
        self.name = "AbstractModel"

        self.imputer_required = True
        self.split_data_to_dataframes = False

    @abstractclassmethod
    def create_task(self):
        pass

    @abstractclassmethod
    def fit(self):
        pass

    @abstractclassmethod
    def model_info(self):
        pass

    def get_metrics(self, y_test, y_pred):
        """This method calculates the metrics for the trained model.

        Metrics do not depend on framework, only on task.
        As long as predict method is the same for the framwework,
        no need to overwrite this method.
        """

        # if not hasattr(self, "model"):
        #     self.scores = "Model not trained!"
        #     return

        # y_predictions = self.model.predict(X_test)

        if self.task == "classification":
            
            if self.framework != "LightAutoML":      
                self.scores = {                    
                    "accuracy": np.round(accuracy_score(y_test, y_pred), 3),
                    "precision": np.round(precision_score(y_test, y_pred), 3),
                    "recall": np.round(recall_score(y_test, y_pred), 3),
                    "f1": np.round(f1_score(y_test, y_pred), 3),
                }

            self.scores["roc_auc"] = np.round(roc_auc_score(y_test, y_pred), 3)

        elif self.task == "regression":
            self.scores = {
                "r2": np.round(r2_score(y_test, y_pred), 3),
                "mse": np.round(mean_squared_error(y_test, y_pred), 3),
                "mae": np.round(mean_absolute_error(y_test, y_pred), 3),
            }
        else:
            self.scores = "Task not recognized!"

    def save_model(self, model=None, preparer: DataPreparer = None):
        """Saves the model including additional parameters.

        This method should be called from child class!
        Parameter 'model' can be acquired in different ways depending on the framework,
        but the rest of the save process should be the same.

        In child class, first obtain the model, then call the parent method with obtained model as a parameter.
        """

        text = Path(os.path.join(APP_PATH, "requirements.txt")).read_text()

        requirements_list = [f"{self.package}", "scikit-learn", "numpy"]

        requirements = {
            req: re.search("%s(.*)%s" % (req + "==", "\n"), text).group(1)
            for req in requirements_list
        }

        model_info = {
            "name": self.name,
            "framework": self.framework,
            "task": self.task,
            "requirements": requirements,
            "model": model,
        }

        if hasattr(self, "scores"):
            model_info["scores"] = self.scores
        else:
            model_info["scores"] = "Scores not obtained!"

        if hasattr(self, "fitted_model_info"):
            model_info["fitted_model_info"] = self.fitted_model_info
        else:
            model_info["fitted_model_info"] = "Fitted model info not obtained"

        if preparer is not None:
            if hasattr(preparer, "features"):
                model_info["features"] = preparer.features
            else:
                model_info["features"] = None
            if hasattr(preparer, "classes"):
                model_info["classes"] = preparer.classes
            else:
                model_info["classes"] = None
        else:
            model_info["features"] = None
            model_info["classes"] = None

        filepath = "".join([MODELS_PATH, f"{self.name}.automl"])
        joblib.dump(model_info, filepath)
