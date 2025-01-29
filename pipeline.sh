#!/bin/bash

echo "ML-скрипт: создание необходимых папок"
echo "..."
mkdir -p ./data_raw_source
mkdir -p ./data_train
mkdir -p ./data_test
echo "ML-скрипт: создание необходимых папок (завершено)"

echo "ML-скрипт: получение, предобработка, тестирование данных"
python3 ./data_load.py
python3 ./data_creation.py
python3 ./model_preprocessing.py
pytest -v ./test_preproc_data.py
echo "ML-скрипт: получение, предобработка, тестирование данных (завершено)"

echo "ML-скрипт: обучение и тестирование модели"
python3 ./model_preparation.py
python3 ./model_testing.py
echo "ML-скрипт: обучение и тестирование модели (завершено)"

echo "ML-скрипт: проверка функций сервера"
pytest -v ./test_functions.py
echo "ML-скрипт: проверка функций сервера (завершено)"

echo "ML-скрипт: создание Docker образа"
if [ "$(docker images -a | grep "flask_server_image" | awk '{print $1}')" == "flask_server_image" ]; then
echo "(уже существующий flask_server_image будет перезаписан)"
docker rmi flask_server_image
else
echo "(flask_server_image создается впервые)"
fi
docker build -t flask_server_image .
echo "ML-скрипт: создание Docker образа (завершено)"
echo "-----------------------------------------------------------------------------------"
echo "ML-скрипт: имя созданного Docker образа: flask_server_image"
echo "ML-скрипт: команда для запуска: docker run -it --rm -p=5000:5000 flask_server_image"
echo "-----------------------------------------------------------------------------------"
