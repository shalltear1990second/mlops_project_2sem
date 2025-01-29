#!/bin/bash

echo "ML-скрипт: создание необходимых папок"
echo "..."
mkdir -p ./data_raw_source
mkdir -p ./data_train
mkdir -p ./data_test
echo "ML-скрипт: создание необходимых папок (завершено)"

echo "ML-скрипт: получение, предобработка данных"
python3 ./data_creation.py
python3 ./model_preprocessing.py
echo "ML-скрипт: получение, предобработка данных (завершено)"

echo "ML-скрипт: обучение и тестирование модели"
python3 ./model_preparation.py
python3 ./model_testing.py
echo "ML-скрипт: обучение и тестирование модели (завершено)"

