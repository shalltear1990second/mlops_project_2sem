import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

DATA_RAW_SOURCE_PATH = \
    "./data_raw_source/car.csv"
DATA_TRAIN_PREPROC_PATH = \
    '/'.join(DATA_RAW_SOURCE_PATH.split('/')[:-2]) \
    + '/data_train/car_train_preproc.csv'
RANDOM_FOREST_MODEL_PATH = \
    '/'.join(DATA_RAW_SOURCE_PATH.split('/')[:-2]) \
    + '/random_forest_model.pkl'
SEPARATOR = ','

df_train = pd.read_csv(DATA_TRAIN_PREPROC_PATH, sep=SEPARATOR)

X_train = df_train.drop(['Car_prices'], axis=1).values
y_train = df_train['Car_prices'].values

model = RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42)
model.fit(X_train, y_train)

with open(RANDOM_FOREST_MODEL_PATH, 'wb') as output:
    pickle.dump(model, output)

print("model_preparation - успешно выполнен")
