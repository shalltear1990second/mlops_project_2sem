# urfu_2_semester
# Веб-приложение для предсказания цены автомобиля
For the MLOps course

## Описание
Данное приложение представляет из себя проект по курсу автоматизации MLOps - конвейер машинного обучения, автоматически собирающий веб-сервер на flask в докер образе. Так же, одновременно, отдельные этапы данной работы являются выполненными лабораторными работами по этому же курсу. Так как проект был объединен с лабораторными, все данные расположены в одной директории. Помимо общих файлов, файлы, отвечающие за конкретные работы следующие:

- (1 лабораторная работа): pipeline.sh (так же, именно этот файл и является главным исполнительным файлом проекта)
- (2 лабораторная работа): Jenkinsfile (для Jenkins-pipeline)
- (3 лабораторная работа): Dockerfile, flask_server.py
- (4 лабораторная работа): соответствующие метафайлы + dvc хранилище (https://disk.yandex.ru/d/ELgaS-_EZ1BcUA) (хранилище локальное, т.к. gdrive больше не работает с dvc, Amazon S3 недоступен, а Yandex DataSphere платный) (ветка репозитория test_dvc)
- (5 лабораторная работа): test_preproc_data.py (тестирование данных), test_functions.py (модульное тестирование)

## Данные
Используемый в данной работе датасет (https://www.kaggle.com/datasets/yash0956/car-price-dataset) представляет из себя данные о стоимости машин на вторичном рынке, взятые с индийского портала CarDekho. Данный датасет использовался в лабораторных работах по математическим основам машинного обучения и подробный исследовательский анализ данных можно посмотреть в ноутбуке по ссылке: https://disk.yandex.ru/d/b8XREOsaIMqLwA .

## Инструкция по установке и применению

 - любым удобным образом скопируйте себе данный репозиторий
 - убедитесь, что все требуемые зависимости были успешно установлены (не только requirements.txt, но и Java, Jenkins, Docker, dvc)
 - скачайте (https://disk.yandex.ru/d/ELgaS-_EZ1BcUA) и разместите в любой удобной директории папку локального хранилища dvc
 - находясь в папке с проектом, выполните команду 'dvc remote add -d local_storage <путь_к_скачанной_папке>', а затем 'dvc pull'
 - находясь в папке с проектом, запустите скрипт pipeline.sh
 - с помощью команды 'docker run -it --rm -p=5000:5000 flask_server_image' запустите контейнер с веб-сервером на выполнение
 - воспользуйтесь в зависимости от необходимости localhost:5000/predict/get либо localhost:5000/predict/post
