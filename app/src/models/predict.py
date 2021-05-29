from app.src.utils.model_utils import *
from ..data.prepare_data import prepare_new_data

# import numpy as np
# from PIL import Image

def predict_pipeline(data, model_info_db_name='cloudant-db'):

    """
        Función para gestionar el pipeline completo de inferencia
        del modelo.
        Args:
            path (str):  Ruta hacia los datos.
        Kwargs:
            model_info_db_name (str):  base de datos a usar para almacenar
            la info del modelo.
        Returns:
            list. Lista con las predicciones hechas.
    """

    # Carga de la configuración de entrenamiento
    model_config = load_model_config(model_info_db_name)['model_config']

    # obteniendo la información del modelo en producción
    model_info = get_best_model_info(model_info_db_name)
    # cargando y transformando los datos de entrada
    data_new = prepare_new_data(data)

    # Descargando el objeto del modelo
    model_name = model_info['name']+'.pkl'
    print('------> Loading the model {} object from the cloud'.format(model_name))
    model = load_model(model_name)

    y_pred = int(model.predict(data_new))
    return y_pred

