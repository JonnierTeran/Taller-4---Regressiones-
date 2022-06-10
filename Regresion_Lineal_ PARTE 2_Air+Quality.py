# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:44:19 2022

@author: jonnier andres teran
"""
# Taller N4 PARTE 2 - Modulo 2   "Regresiones"
# Autor: Jonnier Andres Teran Morales
# Correo= Jonnier.teran@upb.edu.co
# ID No. 502195
# id: 1003064599
# Cel: 3245644212

# Importamos la libreriaS Y Modulos necesarios para el desarrollo de la actividad
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# leemos el archivos csv con la libreria de pandas
data = pd.read_excel("AirQualityUCI.xlsx")

# Declaramos las variables de los datos a evaluar  x  y  y
x = data["NO2(GT)"]

y = data["T"]

x,y = np.array(x).reshape(-1,1), np.array(y)

var_X = x[:8000]
var_Y = y[:8000]

p_X = x[8000:]
p_Y = y[8000:]

# Regresion 
modelo = LinearRegression().fit(var_X,var_Y)

#  R-cuadrado
r_sq = modelo.score(var_X,var_Y)
r_sq_test = modelo.score(p_X,p_Y)

# Predicion de valores a Futuro
y_pred = modelo.predict(p_X)

# Grafica de dispersion 
plt.scatter(p_X,p_Y)
plt.plot(p_X, y_pred)
plt.show()

# Imprimir
print("El R es:", r_sq)
print("Predicion", r_sq_test)
print("")
