FROM python:3.10
WORKDIR /sernik
COPY ./model_preprocessing.py ./
COPY ./flask_server.py ./
COPY ./preproc_pipe.pkl ./
COPY ./random_forest_model.pkl ./
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python","/sernik/flask_server.py"]
