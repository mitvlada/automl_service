from typing import Union, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreparer:
    def __init__(self, df: pd.DataFrame, target: Union[str, List[str]]):
        # Store data in argument which will be changed
        self.data = df

        # Perserve data in non-changing property
        self.__original_data = df

        if isinstance(target, list):
            self.__target = target
        else:
            self.__target = [target]

        self.__stratify = self.determine_stratify()

        self.data_descriptors = self.describe_data()

    @property
    def original_data(self):
        return self.__original_data

    @property
    def target(self):
        return self.__target

    @property
    def stratify(self):
        return self.__stratify

    @property
    def features(self):
        return self.__features

    @property
    def classes(self):
        return self.__classes

    def determine_stratify(self) -> bool:
        """Determine if it is possible to have stratified split of dataset.
        Each category of a class should have more than 2 members.
        This is to accomodate "train_test_split" method of "sklearn" library
        """

        # TBD: How to determine split size?
        SPLIT_SIZE = 0.25

        # To avoid stratify error, minimum class members per split has to be 2
        MINIMUM_CLASS_MEMBERS = 2 * (1 / SPLIT_SIZE)

        # Determine if stratify option is applicable for splitting
        object_columns = list(self.original_data.select_dtypes(["object"]).columns)
        stratify = True
        for column in object_columns:
            class_members = self.original_data[column].value_counts()
            for key, value in class_members.items():
                if value < MINIMUM_CLASS_MEMBERS:
                    stratify = False

        return stratify

    def split_data_to_dataframes(self):
        """Split data into train and test dataframes.
        Condition for stratification is checked priory.
        """
        SPLIT_SIZE = 0.2

        if self.stratify == True:
            stratify = self.data[self.target]
        else:
            stratify = None

        df_train, df_test = train_test_split(
            self.data, test_size=SPLIT_SIZE, shuffle=True, stratify=stratify
        )

        return df_train, df_test

    def split_data(self):
        """Split data into train and test features and targets.
        Condition for stratification is checked priory.
        """

        SPLIT_SIZE = 0.2

        if self.stratify == True:
            stratify = self.data[self.target]
        else:
            stratify = None

        df_train, df_test = train_test_split(
            self.data, test_size=SPLIT_SIZE, shuffle=True, stratify=stratify
        )

        X_train = df_train[
            df_train.columns.difference(self.target, sort=False)
        ].to_numpy()
        y_train = df_train[self.target].to_numpy()

        X_test = df_test[df_test.columns.difference(self.target, sort=False)].to_numpy()
        y_test = df_test[self.target].to_numpy()

        return X_train, y_train, X_test, y_test

    def drop_overleveled_categories(self):
        """This method removes categorical features that have too many levels.

        Currently, this is a placeholder for the idea.
        More sophisticated approach should be considered.
        """

        object_columns = list(self.data.select_dtypes(["object"]).columns)

        drop_criteria = self.data_descriptors["Number_of_rows"] * 0.05
        drop_columns = []
        for column in object_columns:
            if self.data[column].unique().size > drop_criteria:
                drop_columns.append(column)

        self.data.drop([drop_columns], axis=1)

    def impute_missing_values(self):
        """Impute missing values in dataset.

        Numerical values are imputed with mean of the column.
        Categorical values are imputed with first previous value.
        """

        columns = self.data.columns[self.data.isnull().any()].tolist()

        for column in columns:
            if self.data[column].dtype in ["float64", "int64"]:
                self.data[column].fillna(value=self.data[column].mean(), inplace=True)
            elif self.data[column].dtype == "object":
                self.data[column].fillna(method="bfill", inplace=True)
            else:
                self.data[column].fillna(method="bfill", inplace=True)

    def categorical_to_numerical(self, column: str):
        """Maps labels of categorical feauture or class to numerical value.

        This method is called through "desribe_features_and_classes" method
        """

        le = LabelEncoder()
        self.data[column] = le.fit_transform(self.data[column])
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        return mapping

    def desribe_features_and_classes(self):
        """Determine type and other type-dependant parameters of each feature and class.

        These parameters are used when the user accesses the trained model.
        """

        column_names = self.data.columns.values.tolist()
        column_types = self.data.dtypes.tolist()

        features = {}
        classes = {}
        for column_name, column_type in zip(column_names, column_types):
            parameters = {"type": column_type}

            if column_type in ["float64", "int64"]:
                parameters["max"] = self.data[column_name].max()
                parameters["min"] = self.data[column_name].min()
                parameters["mean"] = np.round(self.data[column_name].mean(), 3)
                parameters["median"] = self.data[column_name].median()
                parameters["std"] = np.round(self.data[column_name].std(), 3)

            elif column_type == "object":
                mapping = self.categorical_to_numerical(column_name)
                parameters["mapping"] = mapping

            if column_name not in self.target:
                features[column_name] = parameters
            elif column_name in self.target:
                classes[column_name] = parameters

        self.__features = features
        self.__classes = classes

    def describe_data(self):
        data_count = self.data.shape[0]
        features_count = self.data.shape[1]

        data_descriptors = {
            "Number_of_rows": data_count,
            "Number_of_features": features_count,
        }

        return data_descriptors
