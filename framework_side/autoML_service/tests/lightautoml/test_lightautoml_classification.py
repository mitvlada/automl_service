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
  "framework": "LightAutoML",
  "name": "lightautoml_test_classification",
  "parameters": {
    "timeout": 120,
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
