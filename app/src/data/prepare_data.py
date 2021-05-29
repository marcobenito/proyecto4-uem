import pickle
import os
from app import client
from copy import copy
import numpy as np
from sklearn.pipeline import Pipeline
# from sklearn.base import BaseEstimator, TransformerMixin
# from skimage.feature import hog
from app.src.data.pipeline_fn import Scaler, Flatten, Preprocessing
from cloudant.query import Query
from PIL import Image
from io import BytesIO
import base64



#

def prepare_dataset(path, model_info_db_name='cloudant-db'):
    # Importamos los datos para entrenar el modelo
    with open(os.path.join(path, 'train_data.dat'),'rb') as f:
        (X_train, y_train) = pickle.load(f)
    with open(os.path.join(path, 'test_data.dat'), 'rb') as f:
        (X_test, y_test) = pickle.load(f)

    print('Datos cargados correctamente')

    data_config = load_data_config(model_info_db_name)['data_config']
    scaling_step = ('scaler', Scaler(a=data_config['scaler_a'], b=data_config['scaler_b']))
    flatten_step = ('flatten', Flatten())
    # hog_step = ('hog_features', HogFeaturesExtraction())
    #
    data_processing = Pipeline([scaling_step, flatten_step])

    X_train = data_processing.fit_transform(X_train[::2,:,:])
    print('x train procesados correctamente')
    X_test = data_processing.transform(X_test[::2,:,:])
    print('Datos procesados correctamente')

    return copy(X_train), copy(y_train[::2]), copy(X_test), copy(y_test[::2])

def prepare_new_data(data, model_info_db_name='cloudant-db'):

    X = get_raw_data_from_request(data)
    data_config = load_data_config(model_info_db_name)['data_config']
    # print('Pipeline cargado correctamente')
    prep_step = ('preprocessing', Preprocessing())
    scaling_step = ('scaler', Scaler(a=data_config['scaler_a'], b=data_config['scaler_b']))
    flatten_step = ('flatten', Flatten())
    data_processing = Pipeline([prep_step, scaling_step, flatten_step])

    X_new = data_processing.fit_transform(X)

    return X_new


def get_raw_data_from_request(data):
    """
        Función para obtener nuevas observaciones desde request
        Args:
           data (List):  Lista con la observación llegada por request.
        Returns:
           DataFrame. Dataset con los datos de entrada.
    """
    img = np.asarray(Image.open(BytesIO(base64.b64decode(data))))
    return img

def load_data_config(db_name):
    """
        Función para cargar la info del modelo desde IBM Cloudant.
        Args:
            db_name (str):  Nombre de la base de datos.
        Returns:
            dict. Documento con la configuración del modelo.
    """
    db = client.get_database(db_name)
    query = Query(db, selector={'_id': {'$eq': 'data_config'}})
    return query()['docs'][0]


# class CustomUnpickler(pickle.Unpickler):
#
#     def find_class(self, module, name):
#         if name == 'Scaler':
#             from app.src.data.pipeline_fn import Scaler
#             return Scaler
#         elif name == 'HogFeaturesExtraction':
#             from app.src.data.pipeline_fn import HogFeaturesExtraction
#             return HogFeaturesExtraction
#         return super().find_class(module, name)