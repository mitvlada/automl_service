from tpot import TPOTClassifier, TPOTRegressor

from autoML_service.frameworks.abstract_framework import AbstractAutoMLFramework
from autoML_service.preparers.preparer import DataPreparer


class TpotFramework(AbstractAutoMLFramework):
    def __init__(self, task: str, name: str = "tpot", target: str=None, parameters: dict=None) -> None:
        self.framework = "TPOT"
        self.package = "TPOT"
        
        self.imputer_required = False
        self.split_data_to_dataframes = False
        
        self.task = task
        self.name = name
        self.target = target
        self.parameters = parameters

    def create_task(self):
        if self.task == "classification":
            self.model = TPOTClassifier(**self.parameters)
        elif self.task == "regression":
            self.model = TPOTRegressor(**self.parameters)
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
        self.fitted_model_info = str(self.model.fitted_pipeline_)

    def save_model(self, preparer: DataPreparer):
        if hasattr(self, "model"):
            model = self.model.fitted_pipeline_
        else:
            model = "Model not trained!"

        super().save_model(model, preparer)
