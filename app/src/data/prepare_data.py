import pickle
import numpy as np
from app import ROOT_DIR, cos, client
from copy import copy


def prepare_data():
    # Importamos los datos para entrenar el modelo
    with open('../../data/train_data.dat', 'rb') as f:
        (X_train, y_train) = pickle.load(f)
    with open('../../data/test_data.dat', 'rb') as f:
        (X_test, y_test) = pickle.load(f)

    # Cargamos el pipeline de procesamiento de datos generado en el notebook. Consiste en los
    # siguientes pasos:
    #   1. Escalado de los píxeles, para que estén en el rango [-0.5, 0.5]
    #   2. Extracción de hog features
    data_processing = cos.get_object_in_cos('data_pipeline.pkl', bucket_name='proyecto4-uem')

    X_train = data_processing.fit_transform(X_train)
    X_test = data_processing.transform(X_test)


    return copy(X_train), copy(y_train), copy(X_test), copy(y_test)