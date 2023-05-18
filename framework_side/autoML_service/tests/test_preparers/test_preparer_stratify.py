import pandas as pd
import sys
import traceback
import logging

sys.path.append("/app")
from preparers.preparer import DataPreparer

# Pass dataset
EXCHANGE_FILE_PATH = "/exchange/"
filepath = "".join([EXCHANGE_FILE_PATH, "housing.csv"])
DATASET = pd.read_csv(filepath)
OUTPUT = "Outcome"

# Fail dataset
# To be implemented

try:
    preparer = DataPreparer(DATASET, OUTPUT)
    print(preparer.stratify)
    print("Test status: PASS")

except Exception as e:
    logging.error(traceback.format_exc())
    print("Test status: FAIL")
