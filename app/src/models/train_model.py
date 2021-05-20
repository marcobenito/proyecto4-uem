from ..data.prepare_data import prepare_data
from ..evaluation.evaluate_model import evaluate_model
from app import ROOT_DIR, cos, client
from sklearn.ensemble import RandomForestClassifier
from cloudant.query import Query
import time



def training_pipeline(path, model_info_db_name='proyecto4-uem'):
    """
        Función para gestionar el pipeline completo de entrenamiento
        del modelo.
        Args:
            path (str):  Ruta hacia los datos.
        Kwargs:
            model_info_db_name (str):  base de datos a usar para almacenar
            la info del modelo.
    """
    # Carga del modelo desde IBM COS (fichero pkl generado en el notebook)
    model = load_model('model_pipeline.pkl')

    # timestamp usado para versionar el modelo y los objetos
    ts = time.time()

    # carga y transformación de los datos de train y test
    X_train, y_train, X_test, y_test = prepare_data(path)

    print('---> Training a model with the following configuration:')
    print('Parametros')

    # Ajuste del modelo con los datos de entrenamiento
    model.fit(X_train, y_train)

    # guardado del modelo en IBM COS
    print('------> Saving the model {} object on the cloud'.format('model_'+str(int(ts))))
    save_model(model, 'model',  ts)

    # Evaluación del modelo y recolección de información relevante
    print('---> Evaluating the model')
    metrics_dict = evaluate_model(model, X_test, y_test, ts, 'RandomForest')

    # Guardado de la info del modelo en BBDD documental
    print('------> Saving the model information on the cloud')
    info_saved_check = save_model_info(model_info_db_name, metrics_dict)

    # Check de guardado de info del modelo
    if info_saved_check:
        print('------> Model info saved SUCCESSFULLY!!')
    else:
        if info_saved_check:
            print('------> ERROR saving the model info!!')

    # selección del mejor modelo para producción
    print('---> Putting best model in production')
    put_best_model_in_production(metrics_dict, model_info_db_name)


def save_model(obj, name, timestamp, bucket_name='models-uem'):
    """
        Función para guardar el modelo en IBM COS
        Args:
            obj (sklearn-object): Objeto de modelo entrenado.
            name (str):  Nombre de objeto a usar en el guardado.
            timestamp (float):  Representación temporal en segundos.
        Kwargs:
            bucket_name (str):  depósito de IBM COS a usar.
    """
    cos.save_object_in_cos(obj, name, timestamp)

def load_model(file, bucket_name='proyecto4-uem'):
    """
        Función para cargar el modelo desde IBM COS
        Args:
            file (str): Nombre del archivo del modelo
        Kwargs:
            bucket_name (str):  depósito de IBM COS a usar.
    """
    model = cos.get_object_in_cos(file, bucket_name)
    return model


def save_model_info(db_name, metrics_dict):
    """
        Función para guardar la info del modelo en IBM Cloudant
        Args:
            db_name (str):  Nombre de la base de datos.
            metrics_dict (dict):  Info del modelo.
        Returns:
            boolean. Comprobación de si el documento se ha creado.
    """
    db = client.get_database(db_name)
    client.create_document(db, metrics_dict)

    return metrics_dict['_id'] in db


def put_best_model_in_production(model_metrics, db_name):
    """
        Función para poner el mejor modelo en producción.
        Args:
            model_metrics (dict):  Info del modelo.
            db_name (str):  Nombre de la base de datos.
    """

    # conexión a la base de datos elegida
    db = client.get_database(db_name)
    # consulta para traer el documento con la info del modelo en producción
    query = Query(db, selector={'status': {'$eq': 'in_production'}})
    res = query()['docs']
    #  id del modelo en producción
    best_model_id = model_metrics['_id']

    # en caso de que SÍ haya un modelo en producción
    if len(res) != 0:
        # se realiza una comparación entre el modelo entrenado y el modelo en producción
        best_model_id, worse_model_id = get_best_model(model_metrics, res[0])
        # se marca el peor modelo (entre ambos) como "NO en producción"
        worse_model_doc = db[worse_model_id]
        worse_model_doc['status'] = 'none'
        # se actualiza el marcado en la BDD
        worse_model_doc.save()
    else:
        # primer modelo entrenado va a automáticamente a producción
        print('------> FIRST model going in production')

    # se marca el mejor modelo como "SÍ en producción"
    best_model_doc = db[best_model_id]
    best_model_doc['status'] = 'in_production'
    # se actualiza el marcado en la BDD
    best_model_doc.save()


def get_best_model(model_metrics1, model_metrics2):
    """
        Función para comparar modelos.
        Args:
            model_metrics1 (dict):  Info del primer modelo.
            model_metrics2 (str):  Info del segundo modelo.
        Returns:
            str, str. Ids del mejor y peor modelo en la comparación.
    """

    # comparación de modelos usando la métrica AUC score.
    auc1 = model_metrics1['model_metrics']['roc_auc_score']
    auc2 = model_metrics2['model_metrics']['roc_auc_score']
    print('------> Model comparison:')
    print('---------> TRAINED model {} with AUC score: {}'.format(model_metrics1['_id'], str(round(auc1, 3))))
    print('---------> CURRENT model in PROD {} with AUC score: {}'.format(model_metrics2['_id'], str(round(auc2, 3))))

    # el orden de la salida debe ser (mejor modelo, peor modelo)
    if auc1 >= auc2:
        print('------> TRAINED model going in production')
        return model_metrics1['_id'], model_metrics2['_id']
    else:
        print('------> NO CHANGE of model in production')
        return model_metrics2['_id'], model_metrics1['_id']


def load_model_config(db_name):
    """
        Función para cargar la info del modelo desde IBM Cloudant.
        Args:
            db_name (str):  Nombre de la base de datos.
        Returns:
            dict. Documento con la configuración del modelo.
    """
    db = client.get_database(db_name)
    query = Query(db, selector={'_id': {'$eq': 'model_config'}})
    return query()['docs'][0]