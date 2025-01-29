import pytest
from flask_server import parameters_check_and_to_df


def test_parameters_check_and_to_df():
    df = parameters_check_and_to_df(
        Car_name='Honda',
        kms_driven=1_000,
        Manufacture=2020,
        Transmission='Automatic',
        Seats='wrong_value'
        )

    assert df['No'].iloc[0] == 1.0

    assert df['Car_name'].iloc[0] == 'Honda'
    assert df['Engine'].iloc[0] == '1250 cc'
    assert df['Seats'].iloc[0] == '5 Seats'
    assert df['kms_driven'].iloc[0] == '1000 kms'
    assert df['Fuel_type'].iloc[0] == 'Petrol'
    assert df['Transmission'].iloc[0] == 'Automatic'
    assert df['Ownership'].iloc[0] == '1st Owner'
    assert df['Manufacture'].iloc[0] == 2020

    assert df['Car_prices'].iloc[0] == '4.0 Lakh'
