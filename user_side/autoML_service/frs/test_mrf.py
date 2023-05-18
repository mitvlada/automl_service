import sys
import traceback
import logging
import pandas as pd

sys.path.append("/app")

from mrf_predictor import MRFPredictor


# EXCHANGE_FILE_PATH = "/exchange/"
# filepath = "".join([EXCHANGE_FILE_PATH, "diabetes.csv"])
# DATASET = pd.read_csv(filepath, index_col=0)

file = "diabetes.csv"
task = "classification"
# task = "regression"
time_budget = 60

try:
    mrf = MRFPredictor(task)
    mrf.get_ranked_predictions(file, time_budget)
    print(mrf.df_rankings)
    print("PASS")
    
except Exception as e:
    logging.error(traceback.format_exc())
    print("Test status: FAIL")