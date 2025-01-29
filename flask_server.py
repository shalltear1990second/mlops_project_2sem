import pandas as pd
import pickle
# Нужно для preproc_pipe
# ---------------------------------------------------------------
from model_preprocessing import PreprocessingFirstStepTransformer
from model_preprocessing import RareGrouper
from model_preprocessing import FixColumnNamesTransformer
# ---------------------------------------------------------------
from tabulate import tabulate
from flask import Flask, request
import warnings
warnings.filterwarnings('ignore')
app = Flask(__name__)


PREPROC_PIPE_PATH = \
    'preproc_pipe.pkl'
RANDOM_FOREST_MODEL_PATH = \
    'random_forest_model.pkl'


# ПАРАМЕТРЫ ЗАПРОСА К СЕРВЕРУ (8 параметров):
# ------------------------------------------------------------------------
# Car_name=<марка_модель(строка)>
# Engine=<объем_двигателя(целое положительное число, не меньше 200)>
# Seats=<число_мест(2, 4, 5, 6, 7 или 8)>
# kms_driven=<пробег_в_км(целое положительное число)>
# Fuel_type=<тип_топлива('Diesel', 'Petrol', 'Cng', 'Electric' или 'Lpg')>
# Transmission=<тип_коробки_передач('Manual' или 'Automatic')>
# Ownership=<число_бывших_владельцев(1, 2, 3, 4 или 5)>
# Manufacture=<год_производства(целое положительное четырехзначное число)>
# ------------------------------------------------------------------------


def parameters_check_and_to_df(
        Car_name: str = None,
        Engine: int = None,
        Seats: int = None,
        kms_driven: int = None,
        Fuel_type: str = None,
        Transmission: str = None,
        Ownership: int = None,
        Manufacture: int = None
        ) -> pd.DataFrame:

    if type(Car_name) is not str:
        Car_name = 'no_valid_name'

    if type(Engine) is not int or Engine < 200:
        Engine = '1250 cc'
    else:
        Engine = str(Engine) + ' cc'

    if type(Seats) is not int or Seats not in [2, 4, 5, 6, 7, 8]:
        Seats = '5 Seats'
    else:
        Seats = str(Seats) + ' Seats'

    if type(kms_driven) is not int:
        kms_driven = '400,000 kms'
    else:
        kms_driven = str(kms_driven) + ' kms'

    if str(Fuel_type) not in ['Diesel', 'Petrol', 'Cng', 'Electric', 'Lpg']:
        Fuel_type = 'Petrol'

    if str(Transmission) not in ['Manual', 'Automatic']:
        Transmission = 'Manual'

    if type(Ownership) is not int or Ownership not in [1, 2, 3, 4, 5]:
        Ownership = '1st Owner'
    else:
        Ownership = str(Ownership)

    if type(Manufacture) is not int or len(str(Manufacture)) != 4:
        Manufacture = 2016

    res = pd.DataFrame({
        'No': [1.0],
        'Car_name': [Car_name],
        'Engine': [Engine],
        'Seats': [Seats],
        'kms_driven': [kms_driven],
        'Fuel_type': [Fuel_type],
        'Transmission': [Transmission],
        'Ownership': [Ownership],
        'Manufacture': [Manufacture],
        'Car_prices': ['4.0 Lakh']
    })

    return res


def get_predict_string(
        Car_name,
        Engine,
        Seats,
        kms_driven,
        Fuel_type,
        Transmission,
        Ownership,
        Manufacture) -> str:

    with open(RANDOM_FOREST_MODEL_PATH, 'rb') as input:
        model = pickle.load(input)
    with open(PREPROC_PIPE_PATH, 'rb') as input:
        pipe = pickle.load(input)

    df = parameters_check_and_to_df(
        Car_name,
        Engine,
        Seats,
        kms_driven,
        Fuel_type,
        Transmission,
        Ownership,
        Manufacture
        )

    predicted_price = pipe.transform(df).drop(['Car_prices'], axis=1).values
    predicted_price = model.predict(predicted_price)[0]

    del df['No']
    del df['Car_prices']

    res = '<pre style="font-size: 20px;">'

    res += 'Сформированный запрос:\n\n'
    res += tabulate(df, headers='keys')
    res += '\n\n'
    res += 'Прогнозируемая цена:\n\n'
    print(type(predicted_price))
    res += "{:_}".format(int(predicted_price))
    res += ' рупий'

    res = res.replace('\n', '<br>')
    res += '</pre>'

    return res


def get_predict_json(
        Car_name,
        Engine,
        Seats,
        kms_driven,
        Fuel_type,
        Transmission,
        Ownership,
        Manufacture):

    with open(RANDOM_FOREST_MODEL_PATH, 'rb') as input:
        model = pickle.load(input)
    with open(PREPROC_PIPE_PATH, 'rb') as input:
        pipe = pickle.load(input)

    df = parameters_check_and_to_df(
        Car_name,
        Engine,
        Seats,
        kms_driven,
        Fuel_type,
        Transmission,
        Ownership,
        Manufacture
        )

    predicted_price = pipe.transform(df).drop(['Car_prices'], axis=1).values
    predicted_price = model.predict(predicted_price)[0]

    del df['No']
    del df['Car_prices']

    df['Predicted_price'] = ["{:_}".format(int(predicted_price))]

    return df.iloc[0].to_json()


@app.route('/')
def root_func():
    return '<pre style="font-size: 20px;">flask_server.py активен!</pre>'


@app.route('/predict/get')
def predict_get_func():

    Car_name = request.args.get('Car_name', type=str)
    Engine = request.args.get('Engine', type=int)
    Seats = request.args.get('Seats', type=int)
    kms_driven = request.args.get('kms_driven', type=int)
    Fuel_type = request.args.get('Fuel_type', type=str)
    Transmission = request.args.get('Transmission', type=str)
    Ownership = request.args.get('Ownership', type=int)
    Manufacture = request.args.get('Manufacture', type=int)

    res = get_predict_string(
        Car_name,
        Engine,
        Seats,
        kms_driven,
        Fuel_type,
        Transmission,
        Ownership,
        Manufacture
    )

    return res


@app.route('/predict/post', methods=['POST'])
def predict_post_func():

    Car_name = request.json.get('Car_name')
    Engine = request.json.get('Engine')
    Seats = request.json.get('Seats')
    kms_driven = request.json.get('kms_driven')
    Fuel_type = request.json.get('Fuel_type')
    Transmission = request.json.get('Transmission')
    Ownership = request.json.get('Ownership')
    Manufacture = request.json.get('Manufacture')

    res = get_predict_json(
        Car_name,
        Engine,
        Seats,
        kms_driven,
        Fuel_type,
        Transmission,
        Ownership,
        Manufacture
    )

    return res


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)
