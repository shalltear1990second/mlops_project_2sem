import pandas as pd
import numpy as np
import re
import pickle
import warnings
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

DATA_RAW_SOURCE_PATH = \
    "./data_raw_source/car.csv"
DATA_TRAIN_PATH = \
    '/'.join(DATA_RAW_SOURCE_PATH.split('/')[:-2]) \
    + '/data_train/car_train.csv'
DATA_TRAIN_PREPROC_PATH = \
    '/'.join(DATA_RAW_SOURCE_PATH.split('/')[:-2]) \
    + '/data_train/car_train_preproc.csv'
DATA_TEST_PATH = \
    '/'.join(DATA_RAW_SOURCE_PATH.split('/')[:-2]) \
    + '/data_test/car_test.csv'
DATA_TEST_PREPROC_PATH = \
    '/'.join(DATA_RAW_SOURCE_PATH.split('/')[:-2]) \
    + '/data_test/car_test_preproc.csv'
PREPROC_PIPE_PATH = \
    '/'.join(DATA_RAW_SOURCE_PATH.split('/')[:-2]) \
    + '/preproc_pipe.pkl'
SEPARATOR = ','


class PreprocessingFirstStepTransformer(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(['No'], axis=1)

        Make_list = []
        for each in X['Car_name']:
            Make_list.append(each.split()[0])
        X.insert(0, 'Make', pd.Series(Make_list))
        X = X.drop(['Car_name'], axis=1)

        def f1(int_str):
            return int(int_str[:-3])
        X['Engine'] = pd.Series(map(f1, X['Engine']))

        def f2(int_str):
            return int(int_str[:-6])
        X.insert(3, 'Seats_number', pd.Series(map(f2, X['Seats'])))

        def f3(int_str):
            return int(re.sub(',', '', int_str[:-4]))
        X['kms_driven'] = pd.Series(map(f3, X['kms_driven']))

        def f4_1(value_str):
            if value_str == '1nd Owner':
                return '1st Owner'
            else:
                return value_str
        X['Ownership'] = pd.Series(map(f4_1, X['Ownership']))

        def f4_2(int_str):
            return int(int_str[0:1])
        X.insert(8, 'Ownership_number', pd.Series(map(f4_2, X['Ownership'])))
        from decimal import Decimal

        def f5(float_str):
            return int(Decimal(float_str[:-5]) * 100000)
        X['Car_prices'] = pd.Series(map(f5, X['Car_prices']))

        X = X.drop_duplicates()
        X = X.reset_index(drop=True)

        question = X[(X.Engine < 200)]
        X = X.drop(question.index)
        question = X[(X.Ownership_number == 0)]
        X = X.drop(question.index)
        X = X.reset_index(drop=True)

        X = X.drop(['Seats_number'], axis=1)

        X = X.drop(['Ownership'], axis=1)
        X = X.reset_index(drop=True)

        return X


class RareGrouper(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.05, other_value='Other'):
        self.threshold = threshold
        self.other_value = other_value
        self.freq_dict = {}

    def set_output(self, *, transform=None):
        return self

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=['object']):
            freq = X[col].value_counts(normalize=True)
            self.freq_dict[col] = freq[freq >= self.threshold].index.tolist()
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for col in X.select_dtypes(include=['object']):
            X_copy[col] = X_copy[col].apply(
                lambda x: x if x in self.freq_dict[col] else self.other_value
                )
        return X_copy


class FixColumnNamesTransformer(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        column_names = [x.split('__')[-1] for x in list(X.columns)]
        column_names = [(x, x[:-6])[x[:5] == 'Seats'] for x in column_names]
        X.columns = column_names
        return X

if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    num_car_prices = ['Car_prices']

    num_pipe_engine_ownership_number_manufacture = Pipeline([
        ('scaler', StandardScaler())
    ])
    num_engine_ownership_number_manufacture = [
        'Engine',
        'Ownership_number',
        'Manufacture'
        ]

    num_pipe_kms_driven = Pipeline([
        ('power', PowerTransformer())
    ])
    num_kms_driven = ['kms_driven']

    cat_pipe_make = Pipeline([
        ('replace_rare', RareGrouper(threshold=0.001, other_value='rare')),
        ('encoder', OneHotEncoder(
            drop='if_binary',
            handle_unknown='ignore',
            sparse_output=False,
            dtype=np.int8))
    ])
    cat_make = ['Make']

    cat_pipe_seats_fuel_type = Pipeline([
        ('encoder', OneHotEncoder(
            drop='if_binary',
            handle_unknown='ignore',
            sparse_output=False,
            dtype=np.int8))
    ])
    cat_seats_fuel_type = [
        'Seats',
        'Fuel_type'
        ]

    cat_pipe_transmission = Pipeline([
        ('encoder', OrdinalEncoder(dtype=np.int8))
    ])
    cat_transmission = ['Transmission']

    preproc_second_step_transformer = ColumnTransformer(transformers=[
        (
            'num_car_prices',
            'passthrough',
            num_car_prices
        ),
        (
            'num_engine_ownership_number_manufacture',
            num_pipe_engine_ownership_number_manufacture,
            num_engine_ownership_number_manufacture
        ),
        (
            'num_kms_driven',
            num_pipe_kms_driven,
            num_kms_driven
        ),
        (
            'cat_make',
            cat_pipe_make,
            cat_make
        ),
        (
            'cat_seats_fuel_type',
            cat_pipe_seats_fuel_type,
            cat_seats_fuel_type
        ),
        (
            'cat_transmission',
            cat_pipe_transmission,
            cat_transmission
        )
    ]).set_output(transform='pandas')

    full_preproc_pipeline = Pipeline([
        ('preproc_first_step', PreprocessingFirstStepTransformer()),
        ('preproc_second_step', preproc_second_step_transformer),
        ('fix_column_names', FixColumnNamesTransformer())
    ])

    df_raw = pd.read_csv(DATA_RAW_SOURCE_PATH, sep=SEPARATOR)

    full_preproc_pipeline.fit(df_raw)

    with open(PREPROC_PIPE_PATH, 'wb') as output:
        pickle.dump(full_preproc_pipeline, output)

    df_train = pd.read_csv(DATA_TRAIN_PATH, sep=SEPARATOR)
    df_test = pd.read_csv(DATA_TEST_PATH, sep=SEPARATOR)

    df_train_preproc = full_preproc_pipeline.transform(df_train)
    df_test_preproc = full_preproc_pipeline.transform(df_test)

    df_train_preproc.to_csv(DATA_TRAIN_PREPROC_PATH, sep=SEPARATOR, index=False)
    df_test_preproc.to_csv(DATA_TEST_PREPROC_PATH, sep=SEPARATOR, index=False)

    print("model_preprocessing - успешно выполнен")
