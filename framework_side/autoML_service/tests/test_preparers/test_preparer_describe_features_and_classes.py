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
    preparer.desribe_features_and_classes()
    print(preparer.features_descriptor)
    print()
    print(preparer.classes_descriptor)
    print("Test status: PASS")

except Exception as e:
    logging.error(traceback.format_exc())
    print("Test status: FAIL")
