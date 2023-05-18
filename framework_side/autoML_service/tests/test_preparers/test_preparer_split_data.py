import pandas as pd
import sys
import traceback
import logging

sys.path.append("/app")
from autoML_service.constants import EXCHANGE_PATH

from autoML_service.preparers.preparer import DataPreparer


filepath = "".join([EXCHANGE_PATH, "housing.csv"])
DATASET = pd.read_csv(filepath, index_col=0)
OUTPUT = "median_house_value"

preparer = DataPreparer(DATASET, OUTPUT)
print(preparer.data.info(), "\n")

try:
    X_train, y_train, X_test, y_test = preparer.split_data()
    print(X_train)
    print("Test status: PASS")

except Exception as e:
    logging.error(traceback.format_exc())
    print("Test status: FAIL")
