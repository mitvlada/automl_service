import sys
import traceback
import logging

sys.path.append("/app")
from autoML_service.autoML.session import AutoMLSession


PARAMETERS = {
  "columns": [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "median_house_value",
    "ocean_proximity"
  ],
  "filename": "diabetes.csv",
  "framework": "TPOT",
  "name": "tpot_test_classification",
  "parameters": {
    "generations": 5,
    "population_size": 50,
    # "offspring_size": None,
    # "mutation_rate": 0.9,
    # "crossover_rate": 0.1,
    # "scoring": "accuracy",
    # "cv": 5,
    # "subsample": 1,
    "max_time_mins": 1,
    "max_eval_time_mins": 2,
    # "random_state": None,
    # "early_stop": None,
  },
  "target": "Outcome",
  "task": "classification",
  "time_budget": 2
}


try:
    session = AutoMLSession(PARAMETERS)
    session.run_session()
    print("Test status: PASS")
except Exception as e:
    logging.error(traceback.format_exc())
    print("Test status: FAIL")
