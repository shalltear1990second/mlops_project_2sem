import pandas as pd
import pickle
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import r2_score as R2

DATA_RAW_SOURCE_PATH = \
    "./data_raw_source/car.csv"
DATA_TRAIN_PREPROC_PATH = \
    '/'.join(DATA_RAW_SOURCE_PATH.split('/')[:-2]) \
    + '/data_train/car_train_preproc.csv'
DATA_TEST_PREPROC_PATH = \
    '/'.join(DATA_RAW_SOURCE_PATH.split('/')[:-2]) \
    + '/data_test/car_test_preproc.csv'
RANDOM_FOREST_MODEL_PATH = \
    '/'.join(DATA_RAW_SOURCE_PATH.split('/')[:-2]) \
    + '/random_forest_model.pkl'
SEPARATOR = ','
PRECISION_FORMAT = '{:,.' + str(4) + 'f}'

df_raw = pd.read_csv(DATA_RAW_SOURCE_PATH, sep=SEPARATOR)
df_train = pd.read_csv(DATA_TRAIN_PREPROC_PATH, sep=SEPARATOR)
df_test = pd.read_csv(DATA_TEST_PREPROC_PATH, sep=SEPARATOR)

X_train = df_train.drop(['Car_prices'], axis=1).values
y_train = df_train['Car_prices'].values
X_test = df_test.drop(['Car_prices'], axis=1).values
y_test = df_test['Car_prices'].values

with open(RANDOM_FOREST_MODEL_PATH, 'rb') as input:
    model = pickle.load(input)

y_train_predict = model.predict(X_train)
y_test_predict = model.predict(X_test)

print()
print("Тренировочные данные:")
print()
print("RMSE = "
      + PRECISION_FORMAT.format(
          RMSE(y_train, y_train_predict)
          ).replace(',', '_'))
print("MAPE = "
      + PRECISION_FORMAT.format(
          MAPE(y_train, y_train_predict)
          ).replace(',', '_'))
print("R2   = "
      + PRECISION_FORMAT.format(
          R2(y_train, y_train_predict)
          ).replace(',', '_'))
print()
print("Тестовые данные:")
print()
print("RMSE = "
      + PRECISION_FORMAT.format(
          RMSE(y_test, y_test_predict)
          ).replace(',', '_'))
print("MAPE = "
      + PRECISION_FORMAT.format(
          MAPE(y_test, y_test_predict)
          ).replace(',', '_'))
print("R2   = "
      + PRECISION_FORMAT.format(
          R2(y_test, y_test_predict)
          ).replace(',', '_'))
print()
print("model_testing - успешно выполнен")
