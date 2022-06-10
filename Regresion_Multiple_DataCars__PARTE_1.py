# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:02:26 2022

@author: jonnier andres teran
"""
# Taller N4 PARTE 1- Modulo 2   "Regresiones"
# Autor: Jonnier Andres Teran Morales
# Correo= Jonnier.teran@upb.edu.co
# ID No. 502195
# id: 1003064599
# Cel: 3245644212

# Importamos la libreriaS Y Modulos necesarios para el desarrollo de la actividad
import numpy as np
import pandas as pd
from sklearn import linear_model

# leer el archivos csv con pandas
df_cars = pd.read_csv("cars.csv")

# Creamoos una lista de Marcas de los vehiculos del archivo leido
Marcas_List = [
    (df_cars["Car"] == "Mitsubishi"),
    (df_cars["Car"] == "Ford"),
    (df_cars["Car"] == "Audi"),
    (df_cars["Car"] == "Honda"),
    (df_cars["Car"] == "Hundai"),
    (df_cars["Car"] == "Opel"),
    (df_cars["Car"] == "BMW"),
    (df_cars["Car"] == "Skoda"),
    (df_cars["Car"] == "Fiat"),
    (df_cars["Car"] == "Hyundai"),
    (df_cars["Car"] == "Suzuki"),
    (df_cars["Car"] == "Volvo"),
    (df_cars["Car"] == "Mazda"),
    (df_cars["Car"] == "Toyoty"),
    (df_cars["Car"] == "Mini"),
    (df_cars["Car"] == "VW"),
    (df_cars["Car"] == "Mercedes"),
   
]

# clasificamos con valores numericos las marcas de los vehiculos
 
Num_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

df_cars['marcaCars'] = np.select(Marcas_List, Num_list, default="not_specified")

#Generamos un nuevo dataframe
Df = pd.DataFrame()

Df["marca"] = df_cars["Car"].drop_duplicates()

Df["marcaCars"] = Num_list

#Declaramos nuestras  variable independientes 

x = df_cars[['Volume', 'Weight', 'CO2']]

# valor de la variable dependiente

y = df_cars["marcaCars"]

# Convertimos a tipo Arrays
x = np.array(x)
y = np.array(y)

reg = linear_model.LinearRegression()
reg.fit(x,y)

# Inprimimos los coeficientes
print(reg.coef_)

# Realizamos la Regresion_ Multiple 

predicted_Car = reg.predict([[2000, 1746, 117]])

marcaCarro=int(np.round(predicted_Car,decimals = 0))

nombre = Df[Df["marcaCars"].isin([marcaCarro])]
print(df_cars)

# inprimimos el valor de la marca
print("la Marca Selecionada  es: ",nombre["marca"].values[0])