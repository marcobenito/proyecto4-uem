from app import cos, client
from cloudant.query import Query


def save_model(obj, name, timestamp, bucket_name='proyecto4-db'):
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

def load_model(file, bucket_name='proyecto4-uem-db'):
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
    ac1 = model_metrics1['model_metrics']['accuracy_score']
    ac2 = model_metrics2['model_metrics']['accuracy_score']
    print('------> Model comparison:')
    print('---------> TRAINED model {} with Accuracy: {}'.format(model_metrics1['_id'], str(round(ac1, 3))))
    print('---------> CURRENT model in PROD {} with Accuracy: {}'.format(model_metrics2['_id'], str(round(ac2, 3))))

    # el orden de la salida debe ser (mejor modelo, peor modelo)
    if ac1 >= ac2:
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


def get_best_model_info(db_name):
    """
         Función para cargar la info del modelo de IBM Cloudant
         Args:
             db_name (str):  base de datos a usar.
         Kwargs:
             bucket_name (str):  depósito de IBM COS a usar.
        Returns:
            dict. Info del modelo.
     """
    db = client.get_database(db_name)
    query = Query(db, selector={'status': {'$eq': 'in_production'}})
    return query()['docs'][0]