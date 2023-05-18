from flaml import AutoML

from autoML_service.frameworks.abstract_framework import AbstractAutoMLFramework
from autoML_service.preparers.preparer import DataPreparer


class FlamlFramework(AbstractAutoMLFramework):
    def __init__(self, task: str, name: str = "Flaml", target: str=None, parameters: dict=None) -> None:
        self.framework = "FLAML"
        self.package = "FLAML"

        self.imputer_required = True
        self.split_data_to_dataframes = False        
        
        self.task = task
        self.name = name
        self.target = target
        self.parameters = parameters

    def create_task(self):
        if self.task in ["classification", "regression"]:
            self.model = AutoML(task = self.task, **self.parameters)
        else:
            self.model = None

    def fit(self, splitted_data):
        X_train, y_train, X_test, y_test = splitted_data
        self.model.fit(X_train, y_train)

    def get_metrics(self, splitted_data):
        X_train, y_train, X_test, y_test = splitted_data
        y_pred = self.model.predict(X_test)

        super().get_metrics(y_test, y_pred)

    def model_info(self):
        model_info = {"model": self.model.model.estimator_class.__name__}
        parameters = {key: value for key, value in self.model.model.params.items()}
        model_info["parameters"] = parameters

        self.fitted_model_info = model_info

    def save_model(self, preparer: DataPreparer):
        if hasattr(self, "model"):
            model = self.model
        else:
            model = "Model not trained!"

        super().save_model(model, preparer)
