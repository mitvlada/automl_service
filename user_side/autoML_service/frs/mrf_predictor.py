import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from autoML_service.constants import EXCHANGE_FILE_PATH
from autoML_service.constants import ML_TASKS, ML_TASK_CLASSIFICATION, ML_TASK_REGRESSION 
from autoML_service.constants import MRF_FILE_PATH, MRF_CLASSIFICATION, MRF_REGRESSION
from autoML_service.constants import MRF_CLASSIFICATION_METRIC, MRF_REGRESSION_METRIC

from autoML_service.autoML.constants import FRAMEWORKS


class MRFPredictor():

    def __init__(self, task: str):

        if not task in ML_TASKS:
            self.mrf = None
            self.metric = None
            return
        
        self.task = task

        if task == ML_TASK_CLASSIFICATION:
            filepath = "".join([MRF_FILE_PATH, MRF_CLASSIFICATION])
            self.metric = "predicted_" + MRF_CLASSIFICATION_METRIC
        elif task == ML_TASK_REGRESSION:
            filepath = "".join([MRF_FILE_PATH, MRF_REGRESSION])
            self.metric = "predicted_" + MRF_REGRESSION_METRIC

        self.mrf = joblib.load(filepath)
        
    def get_ranked_predictions(self, dataset, time_budget: int):

        model = self.mrf["model"]
        label_encoder = self.mrf["label_encoder"]

        dataset_charachteristics = self.get_dataset_charachteristics(dataset)
        
        predictions = []
        for framework in FRAMEWORKS:
            framework_label = label_encoder.transform([framework])[0]

            mrf_input = dataset_charachteristics.copy()
            mrf_input.append(framework_label)
            mrf_input.append(time_budget)

            new_data = np.array(mrf_input).reshape(1, -1)

            prediction = {
                "framework": framework,
                self.metric: np.round(model.predict(new_data)[0], 3)
            }

            predictions.append(prediction)
            
        predictions.sort(key=lambda x: x[self.metric], reverse=True)
        for i in range(len(predictions)):
            predictions[i]["rank"] = i + 1

        df_rankings = pd.DataFrame(predictions, columns=["rank", "framework", self.metric])
        # df_rankings.style.set_caption("Framework recommendations")
        # df_rankings.set_index("rank", inplace=True)

        self.df_rankings = df_rankings

    @staticmethod
    def get_dataset_charachteristics(file: str):

        filepath = "".join([EXCHANGE_FILE_PATH, file])
        dataset = pd.read_csv(filepath, index_col=0)

        instances, features = dataset.shape

        # More sopfisticated approach should be considered for distinguisihing between numerical, categorical and other feature types.
        numerical = dataset.select_dtypes(include='number').shape[1]
        numerical_ratio = np.round(numerical/features, 2)
        categorical_ratio = np.round(1 - numerical_ratio, 2)  # This will also include date/time, text etc.

        missing = dataset.isnull().sum().sum()
        missing_ratio = np.round(missing/(instances*features), 2)

        dataset_charachteristics = [instances, features, numerical_ratio, categorical_ratio, missing_ratio]

        return dataset_charachteristics
