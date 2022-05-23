import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_restx import Api, Resource, fields
from flask import render_template
from werkzeug.middleware.proxy_fix import ProxyFix
import pickle
import numpy

from ml_model import get_confusion_matrix, get_simple_confusion_matrix, get_roc_curve

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

db_uri = 'sqlite:///{}/prods_datos.db'.format(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
db.init_app(app)

api = Api(
    app,
    version='1.7', title='API REST',
    description='API REST para el Modelo de Ciencia de Datos',
)

ns = api.namespace('predicciones', description='predicciones')

from db_models import Prediction

db.create_all()

observacion_repr = api.model('Observacion', {
    'sepal_length': fields.Float(description="Longitud del sépalo"),
    'sepal_width': fields.Float(description="Anchura del sépalo"),
    'petal_length': fields.Float(description="Longitud del pétalo"),
    'petal_width': fields.Float(description="Anchura del pétalo"),
})

classified_observation = api.model('ObservacionCalificada', {
    'sepal_length': fields.Float(description="Longitud del sépalo"),
    'sepal_width': fields.Float(description="Anchura del sépalo"),
    'petal_length': fields.Float(description="Longitud del pétalo"),
    'petal_width': fields.Float(description="Anchura del pétalo"),
    'observed_class': fields.String(description='Clase real de la flor')
})

predictive_model = pickle.load(open('simple_model.pkl', 'rb'))


@ns.route('/', methods=['GET', 'POST'])
class PredictionListAPI(Resource):
    """ Manejador del listado de predicciones.
        GET devuelve la lista de predicciones históricas
        POST agrega una nueva predicción a la lista de predicciones
    """

    # -----------------------------------------------------------------------------------
    def get(self):
        """ Maneja la solicitud GET del listado de predicciones
        """
        return [
                   marshall_prediction(prediction) for prediction in Prediction.query.all()
               ], 200

    # -----------------------------------------------------------------------------------
    # La siguiente línea de código sirve para asegurar que el método POST recibe un
    # recurso representado por la observación descrita arriba (observacion_repr).
    @ns.expect(observacion_repr)
    def post(self):
        """ Procesa un nuevo recurso para que se agregue a la lista de predicciones
        """

        prediction = Prediction(representation=api.payload)

        # ---------------------------------------------------------------------
        # Aqui llama a tu modelo predictivo para crear un score o una inferencia
        # o el nombre del valor que devuelve el modelo. Para fines de este
        # ejemplo simplemente se calcula un valor aleatorio.

        # Crea una observación para alimentar el modelo predicitivo, usando los
        # datos de entrada del API.
        model_data = [numpy.array([
            prediction.sepal_length, prediction.sepal_width,
            prediction.petal_length, prediction.petal_width,
        ])]
        prediction.predicted_class = str(predictive_model.predict(model_data)[0])
        prediction.observed_class = str(predictive_model.predict(model_data)[0])
        # ---------------------------------------------------------------------

        # Las siguientes dos líneas de código insertan la predicción a la base
        # de datos mediante la biblioteca SQL Alchemy.
        db.session.add(prediction)
        db.session.commit()

        # Formar la respuesta de la predicción del modelo
        response_url = api.url_for(PredictionAPI, prediction_id=prediction.prediction_id)
        response = {
            "class": prediction.predicted_class,  # la clase que predijo el modelo
            "url": f'{api.base_url[:-1]}{response_url}',  # el URL de esta predicción
            "api_id": prediction.prediction_id  # El identificador de la base de datos
        }
        # La siguiente línea devuelve la respuesta a la solicitud POST con los datos
        # de la nueva predicción, acompañados del código HTTP 201: Created
        return response, 201


# =======================================================================================
# La siguiente línea de código maneja las solicitudes GET del listado de predicciones
# acompañadas de un identificador de predicción, para obtener los datos de una particular
#
# Los métodos PUT y PATCH actualizan una predicción con el resultado de la observación
# real, de tal forma que se puedan obtener métricas de desempeño del modelo.
def _update_observation(prediction_id):
    """ Actualiza una observación con la clase real. Tanto el método PUT como PATCH
        llaman a esta función.
        :param prediction_id: El identificador de predicción que se va a actualizar
    """
    prediction = Prediction.query.filter_by(prediction_id=prediction_id).first()
    if not prediction:
        return 'Id {} no existe en la base de datos'.format(prediction_id), 404
    else:
        # ---------------------------------------------------------------------------
        # Modifica este bloque de código para actualizar la observación con la clase
        # real. No olvides actualizar la observación en la base de datos, para poder
        # calcular las métricas de desempeño del modelo.
        observed_class = api.payload.get('observed_class')
        prediction.observed_class = observed_class
        db.session.commit()
        return 'Observación actualizada: {}'.format(observed_class), 200
        # ---------------------------------------------------------------------------


@ns.route('/<int:prediction_id>', methods=['GET', 'PUT', 'PATCH'])
class PredictionAPI(Resource):
    """ Manejador de una predicción particular
    """

    # -----------------------------------------------------------------------------------
    @ns.doc({'prediction_id': 'Identificador de la predicción'})
    def get(self, prediction_id):
        """ Procesa las solicitudes GET de una predicción particular
            :param prediction_id: El identificador de la predicción a buscar
        """

        prediction = Prediction.query.filter_by(prediction_id=prediction_id).first()
        if not prediction:
            return 'Id {} no existe en la base de datos'.format(prediction_id), 404
        else:
            return marshall_prediction(prediction), 200

    # -----------------------------------------------------------------------------------
    # ns.expect permite a la biblioteca RESTX esperar una representación de un recurso
    # en formato JSON. En este caso el recurso es un objeto de tipo classified_observation
    # que representa una observación clasificada.
    @ns.doc({'prediction_id': 'Identificador de la predicción'})
    @ns.expect(classified_observation)
    def put(self, prediction_id):
        """ Este método maneja la actualización de una observación con la clase que
            tiene en la realidad.

            PUT y PATCH realizan la misma operación.
        """
        return _update_observation(prediction_id)

    # -----------------------------------------------------------------------------------
    # ns.expect permite a la biblioteca RESTX esperar una representación de un recurso
    # en formato JSON. En este caso el recurso es un objeto de tipo classified_observation
    # que representa una observación clasificada.
    @ns.doc({'prediction_id': 'Identificador de la predicción'})
    @ns.expect(classified_observation)
    def patch(self, prediction_id):
        """ Este método maneja la actualización de una observación con la clase que
            tiene en la realidad.

            PUT y PATCH realizan la misma operación.
        """
        return _update_observation(prediction_id)


# =======================================================================================
# La clase ModelPerformanceAPI devuelve el desempeño del modelo, según las observaciones
# que se han actualizado con las clases reales.
# Modifica esta clase para que se adapte a tu modelo predictivo.
@ns.route('/performance/<string:metric>', methods=['GET'])
class ModelPerformanceAPI(Resource):
    """ CALCULO PARA METRICAS COMO ROC_AUC,ACCURACY, F1, CONFUSION_MATRIZ
    """

    # -----------------------------------------------------------------------------------
    @ns.doc({'metric': 'Nombre de la métrica a generar'})
    def get(self, metric):
        """ Estos valores son los que se pueden usar:
            confusion_matrix, roc_curve
        """
        if metric == 'confusion_matrix':
            # el método "isnot" de las propiedades del modelo permiten buscar las
            # observaciones que ya están calificadas
            reported_predictions = Prediction.query.filter(
                Prediction.observed_class.isnot(None)
            ).all()
            return get_confusion_matrix(reported_predictions), 200
        if metric == 'roc_curve':
            return get_roc_curve()
        else:
            return 'Métrica no soportada: {}'.format(metric), 400


# =======================================================================================
@app.route('/metrics/')
def render_metrics():
    """ Método que obtiene las métricas del modelo y las envía a la plantilla HTML para
        generar las gráficas.
    """
    reported_predictions = Prediction.query.filter(
        Prediction.observed_class.isnot(None)
    ).all()
    labels, matrix = get_simple_confusion_matrix(reported_predictions)
    return render_template('metrics.html', labels=labels, matrix=matrix)


# =======================================================================================
def marshall_prediction(prediction):
    """ Función utilería para transofmrar una Predicción de la base de datos a una
        representación de un recurso REST.
        :param prediction: La predicción a transformar
    """
    response_url = api.url_for(PredictionAPI, prediction_id=prediction.prediction_id)
    model_data = {
        'sepal_length': prediction.sepal_length,
        'sepal_width': prediction.sepal_width,
        'petal_length': prediction.petal_length,
        'petal_width': prediction.petal_width,
        "predicted_class": str(prediction.predicted_class),
    }
    if prediction.observed_class:
        model_data['observed_class'] = prediction.observed_class
    response = {
        "api_id": prediction.prediction_id,
        "url": f'{api.base_url[:-1]}{response_url}',
        "created_date": prediction.created_date.isoformat(),
        "prediction": model_data
    }
    return response


# ---------------------------------------------------------------------------------------
def trunc(number, digits):
    """ Función utilería para truncar un número a un número de dígitos dado
        :param digits: El número de digitos a truncar el número
    """
    import math
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

    