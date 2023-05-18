import pandas as pd
import sys
import traceback
import logging

sys.path.append("/app")
from autoML_service.preparers.preparer import DataPreparer


EXCHANGE_FILE_PATH = "/exchange/"
filepath = "".join([EXCHANGE_FILE_PATH, "housing.csv"])
DATASET = pd.read_csv(filepath)
OUTPUT = "Outcome"

preparer = DataPreparer(DATASET, OUTPUT)
print(preparer.data.info(), "\n")

try:
    print(preparer.categorical_to_numerical())
    print(preparer.data.info(), "\n")
    print("Test status: PASS")

except Exception as e:
    logging.error(traceback.format_exc())
    print("Test status: FAIL")
