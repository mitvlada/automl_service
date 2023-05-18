import pandas as pd
from datetime import datetime

from autoML_service.preparers.preparer import DataPreparer

from autoML_service.frameworks.autosklearn1.model import AutoSklearn1Framework
from autoML_service.frameworks.tpot.model import TpotFramework
from autoML_service.frameworks.flaml.model import FlamlFramework
from autoML_service.frameworks.lightautoml.model import LightautomlFramework

from autoML_service.constants import EXCHANGE_FILE_PATH

OPTIONS = {
    "AutoSklearn1": AutoSklearn1Framework,
    "TPOT": TpotFramework,
    "FLAML": FlamlFramework,
    "LightAutoML": LightautomlFramework,
}

class AutoMLSession():

    def __init__(self, parameters=None):    
        self.time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        self.parameters = parameters
        self.session_validity = self.check_session_validity(parameters)

    def check_session_validity(self, parameters):
        """
        This method should check if the passed parameters are valid.
        
        For a demo, only check if parameters are of type dictionary.
        """

        self.filename = parameters["filename"]
        self.framework = parameters["framework"]

        self.task = parameters["task"]
        self.name = parameters["name"]
        self.target = parameters["target"]
        self.parameters = parameters["parameters"]
        
        return isinstance(self.parameters, dict)
        
    def run_session(self):    
        if not self.session_validity:
            raise ValueError("Parameters for AutoML session are not valid")

        filepath = "".join([EXCHANGE_FILE_PATH, self.filename])
        df = pd.read_csv(filepath, index_col=0)
        
        self.preparer = DataPreparer(df, self.target)
        
        autoML_framework = OPTIONS[self.framework]
        self.autoML_framework = autoML_framework(self.task, self.name, self.target, self.parameters)

        if self.autoML_framework.imputer_required:
            self.preparer.impute_missing_values()

        self.preparer.desribe_features_and_classes()
        
        if self.autoML_framework.split_data_to_dataframes:
            splited_data = self.preparer.split_data_to_dataframes()
        else:
            splited_data = self.preparer.split_data()

        self.autoML_framework.create_task()
        self.autoML_framework.fit(splited_data)
        self.autoML_framework.get_metrics(splited_data)
        self.autoML_framework.model_info()
        self.autoML_framework.save_model(self.preparer)
