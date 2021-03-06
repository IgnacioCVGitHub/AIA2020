#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
# ===================================================================
# Ampliación de Inteligencia Artificial
# Implementación de clasificadores 
# Dpto. de CC. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS: Calcedo Vázquez
# NOMBRE: Ignacio
#
# Segundo componente (si se trata de un grupo):
#
# APELLIDOS: Philibert
# NOMBRE: Juliette
# ----------------------------------------------------------------------------


# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo
# que debe realizarse de manera individual. La discusión y el intercambio de
# información de carácter general con los compañeros se permite, pero NO AL
# NIVEL DE CÓDIGO. Igualmente el remitir código de terceros, OBTENIDO A TRAVÉS
# DE LA RED o cualquier otro medio, se considerará plagio. Si tienen
# dificultades para realizar el ejercicio, consulten con el profesor. En caso
# de detectarse plagio, supondrá una calificación de cero en la asignatura,
# para todos los alumnos involucrados. Sin perjuicio de las medidas
# disciplinarias que se pudieran tomar. 
# *****************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
# Y ATRIBUTOS QUE SE PIDEN. EN PARCTICULAR: NO HACERLO EN UN NOTEBOOK.

# NOTAS: 
# * En este trabajo NO SE PERMITE usar Scikit Learn (excepto las funciones que
#   se usan en carga_datos.py). 
# * Se permite (y se recomienda) usar numpy.  

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar todos los conjuntos de datos,
# basta con descomprimir el archivo datos-trabajo-aia.tgz y ejecutar el
# archivo carga_datos.py (algunos de estos conjuntos de datos se cargan usando
# utilidades de Scikit Learn, por lo que para que la carga se haga sin
# problemas, deberá estar instalado el módulosklearn). Todos los datos se
# cargan en arrays de numpy.

# * Datos sobre día adecuado para jugar al tenis: un ejemplo clásico "de
#   juguete", que puede servir para probar la implementación de Naive
#   Bayes. Se carga en las variables X_tenis, y_tenis. 

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.   

# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (republicano o demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos. 

# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.
  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos. Los textos
#   se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En digitdata.zip están todos los datos en formato
#   comprimido. Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 





import numpy as np
import random
import carga_datos
import operator



# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y su correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser aleatoria y
# estratificada respecto del valor de clasificación. Por supuesto, en el orden 
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

# >>> Xe_votos,Xp_votos,ye_votos,yp_votos
#           =particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# >>> y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los tres conjuntos de
# datos: 267/168=178/112=89/56

# >>> np.unique(y_votos,return_counts=True)
#  (array([0, 1]), array([168, 267]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array([0, 1]), array([112, 178]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array([0, 1]), array([56, 89]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=
#               particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------


def particion_entr_prueba(X, y, test=0.20):
    
    classes = list(set(y))
    
    '''
    obtenemos un array completo de los datos con su correspondiente clase.
    una vez creado esto, por cada clase posible vamos creando dos grupos,
    uno de test y otro de entrenamiento, siguiendo la proporción. Para ello,
    nos vamos a valer de random.shuffle y slice
    '''
    completedata = np.concatenate((X, y.reshape(len(y), 1)), axis=1)
    x_e = []
    x_t = []
    for c in classes:
        filtereddata = []
        for element in completedata:
            if element[-1] == c:
                filtereddata.append(element)
        # tenemos todos los datos filtrados para una clase c
        # mezclamos cada uno de los datos
        random.shuffle(filtereddata)
        # dividimos en entrenamiento y prueba
        longitud= len(filtereddata)
        i_seccion=int((1-test)*longitud)
        x_e += filtereddata[:i_seccion]
        x_t += filtereddata[i_seccion:]

    #una vez que tenemos todas las listas terminadas, las convertimos a array
    #y procedemos a dividir cada parte en x_t,x_e,y_t e y_e
    x_e= np.array(x_e)
    x_t= np.array(x_t)

    x_e_d = x_e[:, :-1]
    y_e = x_e[:, -1]
    x_t_d = x_t[:, :-1]
    y_t = x_t[:, -1]
    return x_e_d, x_t_d, y_e, y_t












# ========================================================
# EJERCICIO 2: IMPLEMENTACIÓN DEL CLASIFICADOR NAIVE BAYES
# ========================================================

# Se pide implementar el clasificador Naive Bayes, en su versión categórica
# con suavizado y log probabilidades. Esta versión categórica NO es la versión
# multinomial (específica de vectorización de textos) que se ha visto en el
# tema 3. Lo que se pide es la versión básica del algoritmo Naive Bayes, vista
# en la Sección 5 del tema 6 de la asignatura "Inteligencia Artificial" del
# primer cuatrimestre. 


# ----------------------------------
# 2.1) Implementación de Naive Bayes
# ----------------------------------

# Definir una clase NaiveBayes con la siguiente estructura:

# class NaiveBayes():

#     def __init__(self,k=1):
#                 
#          .....
         
#     def entrena(self,X,y):

#         ......

#     def clasifica_prob(self,ejemplo):

#         ......

#     def clasifica(self,ejemplo):

#         ......


# * El constructor recibe como argumento la constante k de suavizado (por
#   defecto 1) 
# * Método entrena, recibe como argumentos dos arrays de numpy, X e y, con los
#   datos y los valores de clasificación respectivamente. Tiene como efecto el
#   entrenamiento del modelo sobre los datos que se proporcionan.  
# * Método clasifica_prob: recibe un ejemplo (en forma de array de numpy) y
#   devuelve una distribución de probabilidades (en forma de diccionario) que
#   a cada clase le asigna la probabilidad que el modelo predice de que el
#   ejemplo pertenezca a esa clase. 
# * Método clasifica: recibe un ejemplo (en forma de array de numpy) y
#   devuelve la clase que el modelo predice para ese ejemplo.   

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:


class ClasificadorNoEntrenado(Exception):pass


class ErrorClasificador(ClasificadorNoEntrenado):
    def __init__(self, cadena):
        self.args = {cadena}
        self.cadena = cadena

    def mensaje(self):
        return self.cadena

# ------------------------------------------------------------------------------
# Ejemplo "jugar al tenis":

# >>> b_tenis=NaiveBayes(k=0.5)n
# >>> nb_tenis.entrena(X_tenis,y_tenis)
# >>> ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])
# >>> nb_tenis.clasifica_prob(ej_tenis)
# {'no': 0.7564841498559081, 'si': 0.24351585014409202}
# >>> nb_tenis.clasifica(ej_tenis)
# 'no'
# ------------------------------------------------------------------------------


class NaiveBayes():
    k = 0
    priori = None # probabilidad a priori
    cond = None   # probabilidad condicionadas

    def __init__(self, k=1):
        """
        Inicialización del modelo Naive Bayes
        :param k: constante de suavizado
        """
        self.k = k
        self.normalizacion = False
        self.entrenado = False

    def entrena(self, X, y):
        clases, frec_clases = np.unique(y, return_counts=True)
        N = len(y)                        # numero de ejemplo en los datos
        num_clase = len(clases)   # numero de clase
        num_at = len(X[0])                # numero de atributos para cada ejemplo
        complete_data = np.concatenate((X, y.reshape(len(y), 1)), axis=1)

        self.priori = {clases[c]: frec_clases[c]/N for c in range(num_clase)}
        self.cond = {}
        for a in range(num_at):
            tipo_a = np.unique(X[:, a])  # recoger los tipos posibles de un atributo
            for c in range(num_clase):
                X_clase = complete_data[complete_data[:, -1] == clases[c]][:, 0:-1]  # datos que pertenecen a la clase c solo
                for tipo in tipo_a:
                    tipo_en_c, count_tipo_en_c = np.unique(X_clase[:, a], return_counts=True)
                    if tipo not in tipo_en_c:
                        count_a_en_c = 0
                    else:
                        count_a_en_c = count_tipo_en_c[np.where(tipo_en_c == tipo)]
                    self.cond[(a, clases[c], tipo)] = (count_a_en_c + self.k) / (frec_clases[c] + self.k * len(tipo_a))
        self.entrenado = True

    def clasifica_prob(self, ejemplo):
        """
        Devuelve las log-probabilidades de clase de un ejemplo
        """
        if not self.entrenado:
            raise ErrorClasificador("Clasificador no entrenado")
        proba = {c: (self.priori[c]) for c in self.priori.keys()}
        for c in self.priori.keys():
            num_a = 0
            for e in ejemplo:
                proba[c] *= (self.cond[(num_a, c, e)])
                num_a += 1

        total = sum(proba.values())
        for c in self.priori.keys():
            proba[c] = proba[c] / total  # normalizar
        return proba


    def clasifica(self, ejemplo):
        """
        Devuelve la clase en la que es el ejmplo dado en argumento.
        """
        if not self.entrenado:
            raise ErrorClasificador("Clasificador no entrenado")
        proba_ej = self.clasifica_prob(ejemplo)
        clasificacion = max(proba_ej.items(), key=operator.itemgetter(1))[0]  # equivalente a un argmax en un diccionario
        return clasificacion


print("Test Naive Bayes con los datos tenis.")
X_tenis = carga_datos.X_tenis
y_tenis = carga_datos.y_tenis

nb_tenis = NaiveBayes(k=0.5)
nb_tenis.entrena(X_tenis, y_tenis)
ej_tenis = np.array(['Soleado', 'Baja', 'Alta', 'Fuerte'])
print("Ejemplo a clasificar :", ej_tenis)
print(nb_tenis.clasifica_prob(ej_tenis))
# {'no': 0.7564841498559081, 'si': 0.24351585014409202}
print("Clasificación:", nb_tenis.clasifica(ej_tenis))
# 'no'





# ----------------------------------------------
# 2.2) Implementación del cálculo de rendimiento
# ----------------------------------------------

# Definir una función "rendimiento(clasificador,X,y)" que devuelve la
# proporción de ejemplos bien clasificados (accuracy) que obtiene el
# clasificador sobre un conjunto de ejemplos X con clasificación esperada y. 

# ------------------------------------------------------------------------------
# Ejemplo:

# >>> rendimiento(nb_tenis,X_tenis,y_tenis)
# 0.9285714285714286
# ------------------------------------------------------------------------------


def rendimiento(clasificador, X, y):
    """
    Devuelve el porcentaje de ejemplos bien clasificados
    :param clasificador: clasificador a emplear
    :param X: conjunto de ejemplos X
    :param y: clasificacion esperada
    """
    pred = []  # tablero que contenga nuestra prediccion para todos los ejemplos en X con el clasificador dado en argumento
    for i in range(len(X)):
        pred.append(clasificador.clasifica(X[i]))
    return len(np.where(pred == y)[0]) / len(X)


print("Rendimiento :",rendimiento(nb_tenis, X_tenis, y_tenis),"\n")
# 0.9285714285714286


# --------------------------
# 2.3) Aplicando Naive Bayes
# --------------------------

# Usando el clasificador implementado, obtener clasificadores con el mejor
# rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Concesión de prestamos
# - Críticas de películas en IMDB 

# En todos los casos, será necesario separar los datos en entrenamiento y
# prueba, para dar la valoración final de los clasificadores obtenidos (usar
# para ello la función particion_entr_prueba anterior). Ajustar también el
# valor del parámetro de suavizado k. Mostrar el proceso realizado en cada
# caso, y los rendimientos obtenidos.  


# - Votos de congresistas US
print("-----APLICACIÓN DE NAIVE BAYES ------\n")
print("VOTOS DE CONGRESISTAS US \n")

X_votos = carga_datos.X_votos
y_votos = carga_datos.y_votos
t = 0.3
print("Separacion de los datos en ", 100-t*100, "% de entranamiento y ", 100*t, "% de prueba.\n")
X_train_votos, X_test_votos, y_train_votos, y_test_votos = particion_entr_prueba(X_votos, y_votos, test=t)

K = [0.1, 0.5, 1, 5, 10, 100]

for k in K:
    print("k=", k)
    nb_votos = NaiveBayes(k=k)
    nb_votos.entrena(X_train_votos, y_train_votos)
    print("Rendimiento sobre conjunto de entrenamiento", rendimiento(nb_votos, X_train_votos, y_train_votos))
    print("Rendimiento sobre conjunto de test", rendimiento(nb_votos, X_test_votos, y_test_votos), "\n")

print("------------\n")
# - Concesión de prestamos

print("CONCESION DE PRESTAMOS \n")

X_credito = carga_datos.X_credito
y_credito = carga_datos.y_credito
t = 0.3
print("Separacion de los datos en ", 100-t*100, "% de entranamiento y ", 100*t, "% de prueba.\n")
X_train_credito, X_test_credito, y_train_credito, y_test_credito = particion_entr_prueba(X_credito, y_credito, test=t)

K = [0.1, 0.5, 1, 5, 10, 100, 200, 300]

for k in K:
    print("k=", k)
    nb_credito = NaiveBayes(k=k)
    nb_credito.entrena(X_train_credito, y_train_credito)
    print("Rendimiento sobre conjunto de entrenamiento", rendimiento(nb_credito, X_train_credito, y_train_credito))
    print("Rendimiento sobre conjunto de test", rendimiento(nb_credito, X_test_credito, y_test_credito), "\n")

print("------------\n")


# - Críticas de películas en IMDB
"""
En esta sección vamos a estudiar como se comportan distintos clasificadores
de Naive Bayes con el conjunto de datos de imdb, el cual ya viene dividido
en entrenamiento y prueba, lo cual nos facilita un poco el trabajo.

"""

#carga de datos por separado
xe_imdb = carga_datos.X_train_imdb
ye_imdb = carga_datos.y_train_imdb
xt_imdb = carga_datos.X_test_imdb
yt_imdb = carga_datos.y_test_imdb

print("CRITICAS DE PELICULAS EN IMDB con Naive Bayes\n")

Kimdb = [0.5, 1, 5, 25, 100, 250, 300]

for k in Kimdb:
    print("k=", k)
    nb_imdb = NaiveBayes(k=k)
    nb_imdb.entrena(xe_imdb, ye_imdb)
    print("Rendimiento sobre conjunto de entrenamiento", rendimiento(nb_imdb, xe_imdb, ye_imdb))
    print("Rendimiento sobre conjunto de test", rendimiento(nb_imdb, xt_imdb, yt_imdb), "\n")

print("----------------------------------------------------------------")
'''
Para los valores de la lista Kimdb hemos creado un clasificador NaiveBayes
y hemos obtenido los siguientes resultados:

k= 0.5
Rendimiento sobre conjunto de entrenamiento 0.8445
Rendimiento sobre conjunto de test 0.785 

k= 1
Rendimiento sobre conjunto de entrenamiento 0.8455
Rendimiento sobre conjunto de test 0.785 

k= 5
Rendimiento sobre conjunto de entrenamiento 0.842
Rendimiento sobre conjunto de test 0.7825 

k= 25
Rendimiento sobre conjunto de entrenamiento 0.837
Rendimiento sobre conjunto de test 0.795 

k= 100
Rendimiento sobre conjunto de entrenamiento 0.812
Rendimiento sobre conjunto de test 0.7775 

k= 250
Rendimiento sobre conjunto de entrenamiento 0.762
Rendimiento sobre conjunto de test 0.7475 

k= 300
Rendimiento sobre conjunto de entrenamiento 0.742
Rendimiento sobre conjunto de test 0.7425

En lineas generales, los rendimientos han sido bastante homogéneos,
todos entre el 74% y el 80% para los conjuntos de test.
El mejor resultado en tests lo hemos obtenido para K=25
'''
# =================================================
# EJERCICIO 3: IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================

# Este ejercicio es OPCIONAL, y servirá para el ajuste de parámetros en los
# ejercicios posteriores. Si no se realiza, se podrían ajustar siguiendo el
# método "holdout" implementado en el ejercicio 1

# Definir una función 

#  rendimiento_validacion_cruzada(clase_clasificador,params,X,y,n=5)

# que devuelve el rendimiento medio de un clasificador, mediante la técnica de
# validación cruzada con n particiones. Los arrays X e y son los datos y la
# clasificación esperada, respectivamente. El argumento clase_clasificador es
# el nombre de la clase que implementa el clasificador. El argumento params es
# un diccionario cuyas claves son nombres de parámetros del constructor del
# clasificador y los valores asociados a esas claves son los valores de esos
# parámetros para llamar al constructor.

# INDICACIÓN: para usar params al llamar al constructor del clasificador, usar
# clase_clasificador(**params)  

# ------------------------------------------------------------------------------
# Ejemplo:
# --------
# Lo que sigue es un ejemplo de cómo podríamos usar esta función para
# ajustar el valor de algún parámetro. En este caso aplicamos validación
# cruzada, con n=4, en el conjunto de datos de los votos, para estimar cómo de
# bueno es el valor k=0.1 para suavizado en NaiveBayes. Usando la función que
# se pide sería (nótese que debido a la aleatoriedad, no tiene por qué
# coincidir exactamente el resultado):

# >>> rendimiento_validacion_cruzada(NaiveBayes,{"k":0.1},Xe_votos,ye_votos,n=4)
# 0.8963744588744589

# El resultado es la media de rendimientos obtenidos entrenando cada vez con
# todas las particiones menos una, y probando el rendimiento con la parte que
# se ha dejado fuera. 
 
# Si decidimos que es es un buen rendimiento (comparando con lo obtenido para
# otros valores de k), finalmente entrenaríamos con el conjunto de
# entrenamiento completo:

# >>> nb01=NaiveBayes(k=0.1)
# >>> nb01.entrena(Xe_votos,ye_votos)

# Ydaríamos como estimación final el rendimiento en el conjunto de prueba, que
# hasta ahora no hemos usado:
# >>> rendimiento(nb01,Xp_votos,yp_votos)
#  0.88195402298850575

#------------------------------------------------------------------------------


def rendimiento_validacion_cruzada(clase_clasificador, params, X, y, n=5):
    """
    Devuelve el rendimiento medio de un clasificador, mediante la técnica de validación cruzada
    :param clase_clasificador: nombre del clasificador a usar
    :param params: diccionario con los valores de parametros a utilizar
    :param X: datos
    :param y: clasificacion esperada
    :param n: numero de particiones
    :return: rendimiento medio del clasificador
    """

    completedata = np.concatenate((X, y.reshape(len(y), 1)), axis=1)

    N = len(completedata)  # tamaño de los datos
    suma_rend = 0

    inicio = 0
    fin = int(N/n)

    for i in range(n):
        clasificador = clase_clasificador(**params)
        X_train = np.concatenate((completedata[0:inicio, :-1], completedata[fin:, :-1]))
        y_train = np.concatenate((completedata[0:inicio, -1], completedata[fin:, -1]))
        X_test = completedata[inicio:fin, :-1]
        y_test = completedata[inicio:fin, -1]
        clasificador.entrena(X_train, y_train)
        if clasificador.normalizacion:
            suma_rend += rendimiento(clasificador, normaliza(X_test), y_test)
        else:
            suma_rend += rendimiento(clasificador, X_test, y_test)
        inicio = fin+1
        fin += int(N/n)
    return suma_rend/n  # media de los rendimientos

print("Test validación cruzada con los votos")
Xe_votos = carga_datos.X_votos
ye_votos = carga_datos.y_votos
print("Rendimiento:", rendimiento_validacion_cruzada(NaiveBayes, {"k": 0.1}, Xe_votos, ye_votos, n=4))
#  nos da 0.8950761509172724


# ========================================================
# EJERCICIO 4: MODELOS LINEALES PARA CLASIFICACIÓN BINARIA
# ========================================================

# En este ejercicio se pide implementar en Python un clasificador binario
# lineal, basado en regresión logística.


# ---------------------------------------------
# 4.1) Implementación de un clasificador lineal
# ---------------------------------------------

# En esta sección se pide implementar un clasificador BINARIO basado en
# regresión logística, con algoritmo de entrenamiento de descenso por el
# gradiente mini-batch (para minimizar la entropía cruzada).

# En concreto se pide implementar una clase: 

# class RegresionLogisticaMiniBatch():

#    def __init__(self,clases=[0,1],normalizacion=False,
#                 rate=0.1,rate_decay=False,batch_tam=64,n_epochs=200,
#                 pesos_iniciales=None):

#         .....
        
#     def entrena(self,X,y):

#         .....        

#     def clasifica_prob(self,ejemplo):

#         ......
    
#     def clasifica(self,ejemplo):
                        
#          ......

        

# Explicamos a continuación cada uno de estos elementos:


# * El constructor tiene los siguientes argumentos de entrada:

#   + Una lista clases (de longitud 2) con los nombres de las clases del
#     problema de clasificación, tal y como aparecen en el conjunto de datos. 
#     Por ejemplo, en el caso de los datos de las votaciones, esta lista sería
#     ["republicano","democrata"]. La clase que aparezca en segundo lugar de
#     esta lista se toma como la clase positiva.  

#   + El parámetro normalizacion, que puede ser True o False (False por
#     defecto). Indica si los datos se tienen que normalizar, tanto para el
#     entrenamiento como para la clasificación de nuevas instancias. La
#     normalización es la estándar: a cada característica se le resta la media
#     de los valores de esa característica en el conjunto de entrenamiento, y
#     se divide por la desviación típica de dichos valores.

#  + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#    durante todo el aprendizaje. Si rate_decay es True, rate es la
#    tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#  + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#    cada epoch. En concreto, si rate_decay es True, la tasa de
#    aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#    con la siguiente fórmula: 
#       rate_n= (rate_0)*(1/(1+n)) 
#    donde n es el número de epoch, y rate_0 es la cantidad introducida
#    en el parámetro rate anterior. Su valor por defecto es False. 

#  + batch_tam: indica el tamaño de los mini batches (por defecto 64) que se
#    usan para calcular cada actualización de pesos.

#  + n_epochs: número de veces que se itera sobre todo el conjunto de
#    entrenamiento.

#  + pesos_iniciales: si no es None, es un array con los pesos iniciales. Este
#    parámetro puede ser útil para empezar con unos pesos que se habían obtenido
#    y almacenado como consecuencia de un entrenamiento anterior.


# * El método entrena tiene como parámteros de entrada dos arrays X e y, con
#   los datos del conjunto de entrenamiento y su clasificación esperada,
#   respectivamente.


# * Los métodos clasifica y clasifica_prob se describen como en el caso del
#   clasificador NaiveBayes. Igualmente se debe devolver
#   ClasificadorNoEntrenado si llama a los métodos de clasificación antes de
#   entrenar. 

# Se recomienda definir la función sigmoide usando la función expit de
# scipy.special, para evitar "warnings" por "overflow":

# from scipy.special import expit    
#
# def sigmoide(x):
#    return expit(x)


# -------------------------------------------------------------

# Ejemplo, usando los datos del cáncer de mama:

# >>> Xe_cancer,Xp_cancer,ye_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer)

# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True,n_epochs=1000)

# >>> lr_cancer.entrena(Xe_cancer,ye_cancer)

# >>> rendimiento(lr_cancer,Xe_cancer,ye_cancer)
# 0.9912280701754386

# >>> rendimiento(lr_cancer,Xp_cancer,yp_cancer)
# 0.9557522123893806

# -----------------------------------------------------------------

from scipy.special import expit    


def suma_paralelo(a1, a2):
    """
    Recibe dos arrays y devuelve un array con las sumas de sus componentes sumadas
    """
    a3 = [a1[i]+a2[i] for i in range(len(a1))]
    return a3


def sigmoide(x):
    return expit(x)


def normaliza(X):
    """Normaliza los datos"""
    medias = np.mean(X, axis=0)
    desvs = np.std(X, axis=0)
    X_norm = (X - medias) / desvs
    return X_norm


class RegresionLogisticaMiniBatch():

    def __init__(self, clases=[0, 1], normalizacion=False,
                rate=0.1, rate_decay=False, batch_tam=64, n_epochs=200,
                 pesos_iniciales=None):
        
        mapa_reverse = {0: clases[0], 1: clases[1]}
        self.clases = clases
        self.mapa_reverse = mapa_reverse
        self.normalizacion = normalizacion
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.n_epochs = n_epochs
        self.pesos_iniciales = pesos_iniciales
        self.pesos = list()
        # si pesos está vacía, el clasificador no está entrenado
        
    def entrena(self, X, y):
        
        pesos = []
        if self.pesos_iniciales is not None:  # tomamos los pesos directamente de la clase
            pesos = list(self.pesos_iniciales)
        else:  # iniciamos los pesos de forma aleatoria
            dims = X.shape
            pesos = [random.random() for i in range(dims[1])]
        n_epochs = self.n_epochs
        y_2 = y.reshape(len(y), 1)

        if self.normalizacion:
            X = normaliza(X)
                    
        # merge de los array para trabajar mejor con ellos
        big_chunk = np.concatenate((X, y_2), axis=1)
        # inicialización de parámetros
        batch_tam = self.batch_tam
        tasa_l = self.rate
        tasa_l0 = self.rate
        for i in range(n_epochs):
            chunks = np.array_split(big_chunk, batch_tam)
            # dividimos los datos en subconjuntos
            for block in chunks:
                # tomamos un subgrupo de datos
                # para cada subconjunto actualizamos
                pesos_previos = [0.0 for _ in block[0][:-1]]
                for array in block:
                    sum_a = array[-1]-sigmoide(np.dot(pesos, array[:-1]))
                    sum_t = np.dot(sum_a, array[:-1])
                    pesos_previos = suma_paralelo(pesos_previos, sum_t)
                # una vez hecho todo el sumatorio de los elementos del subgrupo,
                # actualizamos los pesos reales multiplicando por la tasa de
                # aprendizaje y sumando
                
                act_b = np.dot(tasa_l, pesos_previos)
                pesos = suma_paralelo(pesos, act_b)
            if self.rate_decay:
                tasa_l = tasa_l0*(1/(1+i))
        self.pesos = pesos

    def clasifica_prob(self, ejemplo):
        if not self.pesos:
            raise ErrorClasificador("Clasificador no entrenado")
        else:
            result = sigmoide(np.dot(self.pesos, ejemplo))
            probs = dict()
            reverse= self.mapa_reverse
            probs[reverse.get(0)]=1-result
            probs[reverse.get(1)]=result
            return probs

    def clasifica(self, ejemplo):
        if not self.pesos:
            raise ErrorClasificador("Clasificador no entrenado")
        else:
            result = sigmoide(np.dot(self.pesos, ejemplo))
            return self.mapa_reverse.get(round(result))
            
Xe_cancer, Xp_cancer, ye_cancer, yp_cancer = particion_entr_prueba(carga_datos.X_cancer,carga_datos.y_cancer)

lr_cancer = RegresionLogisticaMiniBatch(rate=0.1, rate_decay=True, normalizacion=True, n_epochs=1000)

lr_cancer.entrena(Xe_cancer, ye_cancer)

print("Test Regresion logistica sobre los datos del cancer.")
print("Rendimiento:")
rendimiento(lr_cancer, normaliza(Xe_cancer), ye_cancer)
# sale 0.986784140969163




# -----------------------------------
# 4.2) Aplicando Regresión Logística 
# -----------------------------------

# Usando el clasificador implementado, obtener clasificadores con el mejor
# rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay. En alguno de los conjuntos de datos puede ser necesaria
# normalización. Si se ha hecho el ejercicio 3, usar validación cruzada para
# el ajuste (si no, usar el "holdout" del ejercicio 1). 

# Mostrar el proceso realizado en cada caso, y los rendimientos finales obtenidos. 

'''Vamos a procceder a crear para cada conjunto de datos distintos clasificadores
y probar sus rendimientos. Para ello, probaremos con la clase de validación cruzada para
comprobar los rendimientos de los distintos clasificadores

'''

#Votos de congresistas

'''Para este caso, no estimamos que sea necesaria la normalización de datos.
Los valores enteros oscilan entre -1 y 1, y las unidades son identicas, por lo que
no será necesario normalizar'''



# lr_votos = RegresionLogisticaMiniBatch(rate=0.1, rate_decay=True, n_epochs=1000)
# lr_votos2 = RegresionLogisticaMiniBatch(rate=0.25, rate_decay=True, n_epochs=2500)
# lr_votos3 = RegresionLogisticaMiniBatch(rate=0.1, rate_decay=True, n_epochs=1000, 
#                                         batch_tam=128)
# lr_votos4 = RegresionLogisticaMiniBatch(rate=0.1, rate_decay=False, n_epochs=1000,
#                                        batch_tam=256)
# lr_votos5 = RegresionLogisticaMiniBatch(rate=0.2, rate_decay=True, n_epochs=1750,batch_tam=128)


print("El rendimiento de lr_votos es:", rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, {"rate": 0.1,
        "rate_decay": True, "n_epochs": 1000}, Xe_votos, ye_votos))
print("El rendimiento de lr_votos2 es:", rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
   {"rate": 0.25, "rate_decay":True,"n_epochs": 2500}, Xe_votos, ye_votos))
print("El rendimiento de lr_votos3 es:", rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
   {"rate": 0.1, "rate_decay":True,"n_epochs": 1000,"batch_tam": 128}, Xe_votos, ye_votos))
print("El rendimiento de lr_votos4 es:", rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
   {"rate": 0.1, "rate_decay":False,"n_epochs": 1000, "batch_tam": 256}, Xe_votos, ye_votos))
print("El rendimiento de lr_votos5 es:", rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
   {"rate": 0.2, "rate_decay": True, "n_epochs": 1750, "batch_tam": 128}, Xe_votos, ye_votos))

'''Tras unos 10 minutos de espera, hemos obtenido el siguiente volcado de pantalla:
    El rendimiento de lr_votos es: 0.7793103448275861
    El rendimiento de lr_votos2 es: 0.7793103448275861
    El rendimiento de lr_votos3 es: 0.7793103448275861
    El rendimiento de lr_votos4 es: 0.7816091954022989
    El rendimiento de lr_votos5 es: 0.7793103448275861
    
    Los rendimientos han sido identicos en 4 ocasiones, mejorando un poco con
    rate":0.1,"rate_decay":False,"n_epochs":1000,"batch_tam":256
    
    No contentos con esto, vamos a seguir llevando esto adelante con más parámetros,
    para estudiar los rendimientos.
    
'''
print("El rendimiento de lr_votos6 es:", rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
   {"rate": 0.3, "rate_decay": True, "n_epochs": 3500, "batch_tam": 256}, Xe_votos, ye_votos))

print("El rendimiento de lr_votos7 es:", rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
   {"rate": 0.5, "rate_decay": False, "n_epochs": 3500, "batch_tam": 128}, Xe_votos, ye_votos))

print("El rendimiento de lr_votos8 es:",rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
   {"rate": 0.05, "rate_decay": True, "n_epochs": 2000, "batch_tam": 32}, Xe_votos, ye_votos))


'''
El rendimiento de lr_votos6 es: 0.7793103448275861
El rendimiento de lr_votos7 es: 0.7701149425287357
El rendimiento de lr_votos8 es: 0.7793103448275861

De nuevo, nos vemos estancados en ese 77.93% de rendimiento, llegando a empeorar incluso
para el caso de lr_votos7.
'''

#CANCER DE MAMA

'''Para este conjunto de datos, es conveniente normalizar los datos, ya que son muy
diversos. Una vez más, usaremos los mismos parámetros que con los votos de congresistas:'''

X_cancer = carga_datos.X_cancer
y_cancer = carga_datos.y_cancer

print("El rendimiento de lr_cancer es:",  rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, {"rate": 0.1,
        "rate_decay": True, "normalizacion": True,"n_epochs": 1000}, X_cancer, y_cancer))
print("El rendimiento de lr_cancer2 es:", rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
   {"rate": 0.25,  "normalizacion": True, "rate_decay": True, "n_epochs": 2500}, X_cancer, y_cancer))
print("El rendimiento de lr_cancer3 es:",  rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
   {"rate": 0.1, "normalizacion": True, "rate_decay": True,"n_epochs": 1000, "batch_tam": 128}, X_cancer, y_cancer))
print("El rendimiento de lr_cancer4 es:",  rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
   {"rate": 0.1,  "normalizacion": True, "rate_decay": False,"n_epochs": 1000,  "batch_tam": 256}, X_cancer, y_cancer))
print("El rendimiento de lr_cancer5 es:", rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
   {"rate": 0.2, "normalizacion": True, "rate_decay": True, "n_epochs":  1750, "batch_tam": 128}, X_cancer, y_cancer))
print("El rendimiento de lr_cancer6 es:", rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
   {"rate": 0.3, "normalizacion": True, "rate_decay": True, "n_epochs": 3500,  "batch_tam": 256}, X_cancer, y_cancer))
print("El rendimiento de lr_cancer7 es:", rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
   {"rate": 0.5, "normalizacion": True,  "rate_decay": False, "n_epochs": 3500, "batch_tam":  128}, X_cancer, y_cancer))
print("El rendimiento de lr_cancer8 es:", rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
   {"rate": 0.05, "normalizacion": True,  "rate_decay": True,"n_epochs": 2000, "batch_tam": 32}, X_cancer, y_cancer))

'''
Tras unos fantásticos 15 minutos de espera, obtenemos los siguientes resultados para los
clasificadores que hemos definido previamente:
El rendimiento de lr_cancer es: 0.9323798988621996
El rendimiento de lr_cancer2 es: 0.9305941845764855
El rendimiento de lr_cancer3 es: 0.9323798988621996
El rendimiento de lr_cancer4 es: 0.9037926675094816
El rendimiento de lr_cancer5 es: 0.9323798988621996
El rendimiento de lr_cancer6 es: 0.9252370417193425
El rendimiento de lr_cancer7 es: 0.9002212389380532
El rendimiento de lr_cancer8 es: 0.9341814159292035

De media, ha habido una mejoría sustanciosa con respecto a los votos de los congresistas,
de más de un 15% en algunos casos. Los rendimientos son más dispares, pero aun así
excelentes, siendo el mejor resultado el clasificador 8, por unas milésimas     
'''

# Clasificación de IMDB
X_imdb = np.concatenate((carga_datos.X_train_imdb, carga_datos.X_test_imdb))
y_imdb = np.concatenate((carga_datos.y_train_imdb, carga_datos.y_test_imdb))


'''Una vez que carguemos los datos, vamos a usar la misma configuración 
que hemos usado para los otros casos y estudiaremos cómo de efectivo ha sido.
Tras mirar brevemente el conjunto de datos, las características tienen valores 
de 0 y 1, por lo que no va a tener mucho sentido aplicar normalización. 

Cabe destacar que vamos a usar TODO el conjunto de datos que se nos muestra
para el primer clasificador,aunque sea un acercamiento masoquista. Probaremos también a crear clasificadores
independientes y probar con los conjuntos separados, agilizando el proceso un poco
con suerte
'''
print("El rendimiento de lr_imdb es:",  rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch, {"rate": 0.1,
        "rate_decay": True, "normalizacion": True,"n_epochs": 1000}, X_imdb, y_imdb))


'''
Tras esperar pacientemente una hora y media por nuestro resultados, 
obtenemos el siguiente rendimiento:
El rendimiento de lr_imdb es: 0.7295511482254697

El rendimiento roza el 73%, en la linea de lo que sucedía con los votos de
los congresistas.
Evidentemente para el resto de casos no vamos a malgastar otra hora y media de
nuestro tiempo de cómputo, así que vamos a cambiar el enfoque al método de 
hold out que se nos da con el conjunto de imdb
'''
#carga de datos por separado
xe_imdb=carga_datos.X_train_imdb
ye_imdb=carga_datos.y_train_imdb
xt_imdb=carga_datos.X_test_imdb
yt_imdb=carga_datos.y_test_imdb
#creación de los clasificadores
lr_imdb2=RegresionLogisticaMiniBatch(**{"rate": 0.25,  "normalizacion": True, "rate_decay": True, "n_epochs": 2500})
lr_imdb3=RegresionLogisticaMiniBatch(**{"rate": 0.1, "normalizacion": True, "rate_decay": True,"n_epochs": 1000, "batch_tam": 128})
lr_imdb4=RegresionLogisticaMiniBatch(**{"rate": 0.2, "normalizacion": True, "rate_decay": True, "n_epochs":  1750, "batch_tam": 128})
lr_imdb5=RegresionLogisticaMiniBatch(**{"rate": 0.5, "normalizacion": True,  "rate_decay": False, "n_epochs": 3500, "batch_tam":  128})
lr_imdb6=RegresionLogisticaMiniBatch(**{"rate": 0.05, "normalizacion": True,  "rate_decay": True,"n_epochs": 2000, "batch_tam": 32})

lr_imdb2.entrena(xe_imdb,ye_imdb)
lr_imdb3.entrena(xe_imdb,ye_imdb)
lr_imdb4.entrena(xe_imdb,ye_imdb)
lr_imdb5.entrena(xe_imdb,ye_imdb)
lr_imdb6.entrena(xe_imdb,ye_imdb)

print("El rendimiento de lr_imdb2 es:",rendimiento(lr_imdb2,xt_imdb,yt_imdb))
print("El rendimiento de lr_imdb3 es:",rendimiento(lr_imdb3,xt_imdb,yt_imdb))
print("El rendimiento de lr_imdb4 es:",rendimiento(lr_imdb4,xt_imdb,yt_imdb))
print("El rendimiento de lr_imdb5 es:",rendimiento(lr_imdb5,xt_imdb,yt_imdb))
print("El rendimiento de lr_imdb6 es:",rendimiento(lr_imdb6,xt_imdb,yt_imdb))


'''Resultados:
El rendimiento de lr_imdb2 es: 0.7525
El rendimiento de lr_imdb3 es: 0.765
El rendimiento de lr_imdb4 es: 0.7625
El rendimiento de lr_imdb5 es: 0.76
El rendimiento de lr_imdb6 es: 0.77

Vemos que los resultados son ligeramente mejores, en torno al 75%-77%, siendo
el mejor resultado aquel con los parámetros de "rate": 0.05, 
"normalizacion": True,"rate_decay": True,"n_epochs": 2000, "batch_tam": 32.

Aun así, estimamos que los datos no son los más adecuados para estudiar con
regresión logística.
'''

# =====================================
# EJERCICIO 5: CLASIFICACIÓN MULTICLASE
# =====================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica
# de "One vs Rest" (OvR)

# ------------------------------------
# 5.1) Implementación de One vs Rest
# ------------------------------------


#  En concreto, se pide implementar una clase python RL_OvR con la siguiente
#  estructura, y que implemente un clasificador OvR usando como base el
#  clasificador binario del apartado anterior.


# class RL_OvR():

#     def __init__(self,clases,rate=0.1,rate_decay=False,batch_tam=64,n_epochs=200):

#        ......

#     def entrena(self,X,y):

#        .......

#     def clasifica(self,ejemplo):

#        ......
            



#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, excepto que ahora "clases" puede ser una lista con más de dos
#  elementos. 

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

# >>> rl_iris=RL_OvR([0,1,2],rate=0.001,batch_tam=20,n_epochs=1000)

# >>> rl_iris.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris,Xe_iris,ye_iris)
# 0.9732142857142857

# >>> rendimiento(rl_iris,Xp_iris,yp_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------

class RL_OvR():

     def __init__(self, clases, rate=0.1, rate_decay=False, batch_tam=64, n_epochs=200):
         self.clases = clases
         self.rate = rate
         self.rate_decay = rate_decay
         self.batch_tam = batch_tam
         self.n_epochs = n_epochs
         self.pesos = list()
         self.reg = list()
         # si pesos está vacía, el clasificador no está entrenado

     def entrena(self, X, y):
        for i in range(len(self.clases)):
            self.reg.append(RegresionLogisticaMiniBatch(clases=[0, 1], normalizacion=False, rate=self.rate, rate_decay=self.rate_decay, batch_tam=self.batch_tam, n_epochs=self.n_epochs,pesos_iniciales=None))
            y_new = np.array([1 if j == self.clases[i] else 0 for j in y])
            self.reg[i].entrena(X, y_new)

     def clasifica(self, ejemplo):
         prob = []
         for i in range(len(self.clases)):
            prob.append(self.reg[i].clasifica_prob(ejemplo)[1])
         return self.clases[np.argmax(prob)]

print("Test One vs Rest con los datos iris.")

Xe_iris, Xp_iris, ye_iris, yp_iris = particion_entr_prueba(carga_datos.X_iris, carga_datos.y_iris)

rl_iris = RL_OvR([0, 1, 2], rate=0.001, batch_tam=20, n_epochs=1000)

rl_iris.entrena(Xe_iris, ye_iris)

print("Rendimiento:", rendimiento(rl_iris, Xe_iris, ye_iris), "\n")
print("-------------------------------------")

# ---------------------------------------------------------
# 5.2) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación del apartado anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. En este
#  caso concreto, NO USAR VALIDACIÓN CRUZADA para ajustar, ya que podría
#  tardar bastante (basta con ajustar comparando el rendimiento en
#  validación). Si el tiempo de cómputo en el entrenamiento no permite
#  terminar en un tiempo razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 

def leer_digitos(fichero):
    """
    Lee un fichero que nos da imagenes de 28x29 pixeles representando digitos.
    Devuelve un array numpy que contenga la representacion de los digitos de este fichero.
    """
    f = open(fichero)
    count = 0
    datos = []
    imagen = []
    for linea in f:
        transform_linea = [0 if c == " " else 1 for c in linea]
        imagen.append(transform_linea)
        count += 1
        if count == 28:
            count = 0
            datos.append(imagen)
            imagen = []
    f.close()
    return np.array(datos)


def leer_label(fichero):
    f = open(fichero)
    labels = []
    for l in f:
        labels.append(int(l))
    f.close()
    return np.array(labels)


Xtrain_digitos = leer_digitos("datos/trainingimages").reshape(5000, 28*29)[0:500]
ytrain_digitos = leer_label("datos/traininglabels")[0:600]

Xtest_digitos = leer_digitos("datos/testimages").reshape(1000, 28*29)
ytest_digitos = leer_label("datos/testlabels")

Xval_digitos = leer_digitos("datos/validationimages").reshape(1000, 28*29)
yval_digitos = leer_label("datos/validationlabels")

reg_digitos = RL_OvR(np.arange(10), rate=0.001, batch_tam=20, n_epochs=1000)
reg_digitos.entrena(Xtrain_digitos, ytrain_digitos)  # largo en tiempo = mas o menos 30 minutos para 500 datos

print("RL con rate=0.001, batch_tam=20, n_epochs=1000)
print("Rendimento sobre los datos de entranamiento", rendimiento(reg_digitos, Xtrain_digitos, ytrain_digitos))
# nos da 1 de rendimiento para los datos de entranamiento  -> sobreajuste
print("Rendimiento sobre los datos de test", rendimiento(reg_digitos, Xtest_digitos, ytest_digitos))
# 0.79 de rendimiento para los datos de test

reg_digitos = RL_OvR(np.arange(10), rate=0.3, batch_tam=50, n_epochs=100)
reg_digitos.entrena(Xtrain_digitos, ytrain_digitos)  # largo en tiempo = mas o menos 30 minutos para 500 datos

print("RL con rate=0.3, batch_tam=256, n_epochs=950")
print("Rendimento sobre los datos de entranamiento", rendimiento(reg_digitos, Xtrain_digitos, ytrain_digitos))
# nos da 1 de rendimiento para los datos de entranamiento -> sobreajuste
print("Rendimiento sobre los datos de test", rendimiento(reg_digitos, Xtest_digitos, ytest_digitos))
# 0.802 de rendimiento para los datos de test