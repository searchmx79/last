# =======================================================================================
#                       IEXE Tec - Maestría en Ciencia de Datos 
#                       Productos de Datos. Proyecto Integrador
#
# Debes de adaptar este script para integrar tu modelo predictivo.
# =======================================================================================
import numpy as np
import sklearn
from pandas import read_csv
from pandas.core.common import random_state
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, roc_curve, euclidean_distances, auc
import pickle


# =======================================================================================
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize


def create_simple_model():
    """ Función para entrenar un modelo simple usando los datos de prueba de tipos de flores
        Consulta la documentación de este famoso conjunto de datos:
        https://en.wikipedia.org/wiki/Iris_flower_data_set

        Modifica esta función para integrar el modelo predictivo que quieres integrar a
        tu API REST.

        Esta función crea un modelo muy simple usando los datos de Tipos de Flores. Es un
        modelo muy simple usando el algoritmo de regresión logística. Para este ejemplo
        no hay ningún proceso de análisis de variables, entrenamiento o comparación de
        múltiples modelos, etc. Estas actividades son muy importantes y se realizan en una
        etapa posterior a la creación del modelo, en este punto tu modelo ya está entrenado
        y probado, listo para integrarse a un modelo predicitivo.

        Una vez que el modelo se encuentra listo usamos la biblioteca pickle
        (https://docs.python.org/3/library/pickle.html) para "guardar" el modelo en un
        archivo en disco duro. Este archivo contiene el estado del entrenamiento y
        pruebas del modelo, de esta forma se "desconecta" del entorno de laboratorio del
        modelo para que se pueda usar en una aplicación de cómputo.

        El archivo en disco duro que contiene el modelo se "recupera" en el API REST,
        quien es responsable de transformar la entrada del API en un formato que sea
        entendible por el modelo.
    """

    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv('iris.csv', names=names)
    array = dataset.values
    X = array[:, 0:4]
    y = array[:, 4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, y, test_size=0.20, random_state=1, shuffle=True
    )

    model = LogisticRegression(solver='liblinear', multi_class='ovr')
    model.fit(X_train, Y_train)

    pickle.dump(model, open('simple_model.pkl', 'wb'))


# =======================================================================================
def get_simple_confusion_matrix(predictions):
    """
    """
    y_true = []
    y_pred = []
    labels = set()
    for prediction in predictions:
        y_true.append(prediction.observed_class)
        y_pred.append(prediction.predicted_class)
        labels.add(prediction.observed_class)
        labels.add(prediction.predicted_class)
    labels = list(labels)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return labels, matrix


# =======================================================================================
def get_confusion_matrix(predictions):
    """ Esta función convierte las predicciones que ya tienen calificación en la matriz
        de confusión de desempeño del modelo.
        :param predictions: La lista de predicciones calificadas
        :return: Un diccionario con la matriz de confusión del modelo
    """

    labels, matrix = get_simple_confusion_matrix(predictions)
    result_matrix = {}

    for actual_label, row in zip(labels, matrix):
        result_matrix[actual_label] = {}
        for label, count in zip(labels, row):
            result_matrix[actual_label][label] = int(count)

    return result_matrix


def get_roc_curve():
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]
    classifier = OneVsRestClassifier(
        svm.SVC(kernel="linear", probability=True, random_state=random_state)
    )
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return roc_auc

# =======================================================================================
if __name__ == '__main__':
    """ Esta función ejecuta el programa de Python cuando se invoca desde 
        línea de comando.
    """
    create_simple_model()
