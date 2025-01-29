import pytest
import pandas as pd
import numpy as np

DATA_RAW_SOURCE_PATH = \
    "./data_raw_source/car.csv"
DATA_TRAIN_PREPROC_PATH = \
    '/'.join(DATA_RAW_SOURCE_PATH.split('/')[:-2]) \
    + '/data_train/car_train_preproc.csv'
DATA_TEST_PREPROC_PATH = \
    '/'.join(DATA_RAW_SOURCE_PATH.split('/')[:-2]) \
    + '/data_test/car_test_preproc.csv'
SEPARATOR = ','


@pytest.fixture()
def init_data():

    df_train = pd.read_csv(DATA_TRAIN_PREPROC_PATH, sep=SEPARATOR)
    df_test = pd.read_csv(DATA_TEST_PREPROC_PATH, sep=SEPARATOR)

    res = {
        'df_train': df_train,
        'df_test': df_test
        }

    return res


def test_duplicates_train_preproc(init_data):
    res = init_data['df_train'].duplicated()
    res = res[res == True]
    res = res.shape[0]
    assert res == 0


def test_duplicates_test_preproc(init_data):
    res = init_data['df_test'].duplicated()
    res = res[res == True]
    res = res.shape[0]
    assert res == 0


def test_null_train_preproc(init_data):
    res = init_data['df_train'].isnull().any(axis=1)
    res = res[res == True]
    res = res.shape[0]
    assert res == 0


def test_null_test_preproc(init_data):
    res = init_data['df_test'].isnull().any(axis=1)
    res = res[res == True]
    res = res.shape[0]
    assert res == 0


def test_only_valid_types_train_preproc(init_data):
    res = init_data['df_train']
    res = res.dtypes
    res = res.apply(type)
    res = res[(res != np.dtypes.Int64DType) & (res != np.dtypes.Float64DType)]
    res = res.shape[0]
    assert res == 0


def test_only_valid_types_test_preproc(init_data):
    res = init_data['df_test']
    res = res.dtypes
    res = res.apply(type)
    res = res[(res != np.dtypes.Int64DType) & (res != np.dtypes.Float64DType)]
    res = res.shape[0]
    assert res == 0
