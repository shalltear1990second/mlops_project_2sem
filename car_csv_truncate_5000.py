import pandas as pd

DATA_RAW_SOURCE_PATH = \
    "./data_raw_source/car.csv"
SEPARATOR = ','

df_raw_mod = pd.read_csv(DATA_RAW_SOURCE_PATH, sep=SEPARATOR)

max_index = min(5_000, df_raw_mod.shape[0])

df_raw_mod = df_raw_mod.iloc[:max_index]

df_raw_mod.to_csv(DATA_RAW_SOURCE_PATH, sep=SEPARATOR, index=False)

print("car_csv_truncate_5000 - успешно выполнен, car.csv был ИЗМЕНЁН!")
