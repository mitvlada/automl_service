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
  "framework": "AutoSklearn1",
  "name": "autosklearn1_test_classification",
  "parameters": {
    "time_left_for_this_task": 120,
    "per_run_time_limit": 60,
    "memory_limit": 102400,
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
