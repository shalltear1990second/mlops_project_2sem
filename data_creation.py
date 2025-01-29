import pandas as pd

DATA_RAW_SOURCE_PATH = \
    "./data_raw_source/car.csv"
DATA_TRAIN_PATH = \
    '/'.join(DATA_RAW_SOURCE_PATH.split('/')[:-2]) \
    + '/data_train/car_train.csv'
DATA_TEST_PATH = \
    '/'.join(DATA_RAW_SOURCE_PATH.split('/')[:-2]) \
    + '/data_test/car_test.csv'
SEPARATOR = ','

df_raw = pd.read_csv(DATA_RAW_SOURCE_PATH, sep=SEPARATOR)

df_train = df_raw.sample(frac=0.8, random_state=42)
df_test = df_raw.drop(df_train.index)

df_train.to_csv(DATA_TRAIN_PATH, sep=SEPARATOR, index=False)
df_test.to_csv(DATA_TEST_PATH, sep=SEPARATOR, index=False)

print("data_reation - успешно выполнен")
