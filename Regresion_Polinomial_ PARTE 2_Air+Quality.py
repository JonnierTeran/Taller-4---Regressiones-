# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:54:21 2022

@author: jonnier andres teran
"""
# Taller N4 PARTE 2 - Modulo 2   "Regresiones"
# Autor: Jonnier Andres Teran Morales
# Correo= Jonnier.teran@upb.edu.co
# ID No. 502195
# id: 1003064599
# Cel: 3245644212

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ----------------Regresion Polinomial --------------------
# leer el archivos xlsx con pandas
data = pd.read_excel("AirQualityUCI.xlsx")

# Declaramos las variable de X  y Y
x = data["NO2(GT)"]

y = data["T"]

var_X = x[:8000]
var_Y = y[:8000]

p_X = x[8000:]
p_Y = y[8000:]

# Model polinomial​

Modelo = np.poly1d(np.polyfit(var_X, var_Y, 3))

# Definimos el espaciamiento para la linea
myline = np.linspace(100, 1000, 8000)

# Graficamos la línea de regresión polinomial

plt.scatter(var_X,var_Y)
plt.plot(var_X, Modelo(myline))
plt.show()
r2 = r2_score(var_Y, Modelo(var_X))
print("El R cuadrado es:")
print(r2)