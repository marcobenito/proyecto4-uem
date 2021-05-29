from flask import Flask, request
import os
import numpy as np
from app.src.models import train_model, predict
from app import ROOT_DIR
import warnings
import base64
# import cv2

warnings.filterwarnings('ignore')

app = Flask(__name__)

port = int(os.getenv('PORT', 8000))

@app.route('/', methods=['GET'])
def root():
    """
            Función para gestionar la salida de la ruta raíz.
            Returns:
               dict.  Mensaje de salida
        """
    # No hacemos nada. Solo devolvemos info (customizable a voluntad)
    return {'Proyecto': 'Mod. 4 - Ciclo de vida de modelos IA'}

# ruta para el lanzar el pipeline de entranamiento (Método GET)
@app.route('/train-model', methods=['GET'])
def train_model_route():
    """
        Función de lanzamiento del pipeline de entrenamiento.
        Returns:
           dict.  Mensaje de salida
    """
    # Ruta para la carga de datos locales
    df_path = os.path.join(ROOT_DIR, 'data/')

    # Lanzar el pipeline de entranamiento de nuestro modelo
    train_model.training_pipeline(df_path)

    # Se puede devolver lo que queramos (mensaje de éxito en el entrenamiento, métricas, etc.)
    return {'TRAINING MODEL': 'Mod. 4 - Ciclo de vida de modelos IA'}


# ruta para el lanzar el pipeline de inferencia (Método POST)
@app.route('/predict', methods=['POST'])
def predict_route():
    """
        Función de lanzamiento del pipeline de inferencia.
        Returns:
           dict.  Mensaje de salida (predicción)
    """

    # r = request
    # # convert string of image data to uint8
    # nparr = np.fromstring(r.data, np.uint8)
    # # decode image
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Obtener los datos pasados por el request
    data = request.get_json()

    # Lanzar la ejecución del pipeline de inferencia
    y_pred = predict.predict_pipeline(data['img'])

    return {'Predicted value': y_pred}


# main
if __name__ == '__main__':
    # ejecución de la app
    app.run(host='0.0.0.0', port=port, debug=True)