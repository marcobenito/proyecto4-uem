from ..data.prepare_data import prepare_dataset
from ..evaluation.evaluate_model import evaluate_model
from sklearn.ensemble import RandomForestClassifier
from app.src.utils.model_utils import *
from datetime import datetime




def training_pipeline(path, model_info_db_name='cloudant-db'):
    """
        Función para gestionar el pipeline completo de entrenamiento
        del modelo.
        Args:
            path (str):  Ruta hacia los datos.
        Kwargs:
            model_info_db_name (str):  base de datos a usar para almacenar
            la info del modelo.
    """

    # Carga de la configuración de entrenamiento
    model_config = load_model_config(model_info_db_name)['model_config']
    model = RandomForestClassifier(max_depth=model_config['max_depth'],
                                   min_samples_leaf=model_config['min_samples_leaf'],
                                   n_jobs=-1)


    # timestamp usado para versionar el modelo y los objetos
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    ts1 = datetime.now().strftime("%H%M")
    ts = "20210520" + ts1

    # carga y transformación de los datos de train y test
    X_train, y_train, X_test, y_test = prepare_dataset(path)

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


