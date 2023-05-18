import pandas as pd

from lightautoml.automl.presets.tabular_presets import TabularUtilizedAutoML
from lightautoml.tasks import Task

from autoML_service.frameworks.abstract_framework import AbstractAutoMLFramework
from autoML_service.preparers.preparer import DataPreparer


class LightautomlFramework(AbstractAutoMLFramework):
    def __init__(self, task: str, name: str = "lightautoml", target: str=None, parameters: dict=None) -> None:
        self.framework = "LightAutoML"
        self.package = "lightautoml"

        self.imputer_required = False
        self.split_data_to_dataframes = True
    
        self.task = task
        self.name = name
        self.target = target
        self.parameters = parameters
        
    def create_task(self):
        task = None
        
        if self.task == "classification":
            task = Task("binary")
        elif self.task == "regression":
            task = Task("reg")
            
        if task is not None:
            self.model = TabularUtilizedAutoML(task=task, **self.parameters)
        else:
            self.model = None

    def fit(self, splitted_data):

        roles = {
            "target": self.target,
        }

        df_train, df_test = splitted_data
        self.model.fit_predict(df_train, roles=roles, verbose=0)

    def get_metrics(self, splitted_data, **kwargs):
        df_train, df_test = splitted_data

        y_test = df_test[self.target].to_numpy()
        y_pred = self.model.predict(df_test)

        super().get_metrics(y_test, y_pred.data)

    def model_info(self):
        self.fitted_model_info = self.model.create_model_str_desc()

    def save_model(self, preparer: DataPreparer):
        if hasattr(self, "model"):
            model = self.model
        else:
            model = "Model not trained!"

        super().save_model(model, preparer)
