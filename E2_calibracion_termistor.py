# -*- coding: utf-8 -*-
"""
Instituto de Física - Facultad de Ingeniería - Universidad de la República
This code is licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) license.

Script para ejecutar la calibración de un termistor a partir de las mediciones
de resistencia y temperatura.

Entradas (input): valores de resistencia, valores de temperatura
Salidas (output): parámetro B, parámetro Ro

@author: mosorio
@date: 202212
@email: mosorio@fing.edu.uy
"""

# Librerías para trabajar
import numpy as np

from support_funcs import min_cuad as min_cuad
from support_funcs import graficar_datos as graficar_datos
from support_funcs import graficar_datos_con_modelo as graficar_datos_con_modelo

try:
    import IPython
    shell = IPython.get_ipython()
    shell.enable_matplotlib(gui='qt')
except:
    pass  

# ---------------------------------- INPUT ------------------------------
# Ground Truth
B = 3000;
To = 25+273;
Ro = 47;

# Data with noise -> Input from students
T = np.linspace(15+273,50+273);
R = Ro*np.exp(B*((1/T)-(1/To)))
noise = np.random.normal(0,3,np.size(R))
measData = R + noise;
# ------------------------------- FIN INPUT ------------------------------

graficar_datos(measData,'T[K]','R [ohm] ', datos_x=T)

# Reorganizacion de datos para poder realizar la linealizacion
# Modelo: R(T)=Ro*exp(B*(1/T-1/To)) -> CV: y=ln(R(T)); x=1/T-1/To --> y = ln(Ro)+B*x
y = np.log(measData);
x = (1/T)-1/To;

graficar_datos(y, '1/T-1/To [1/K]', 'ln(R(T)) [u.a.]', datos_x=x)

# Proceso de Fit por Mínimos Cuadrados -> salida: coef (coeficientes)
coef = min_cuad(x,y,1,'1/T-1/To [1/K]','ln(R(T)) [u.a.]')

# Inversión del modelo a partir de los coeficientes obtenidos
graficar_datos_con_modelo(T, measData, T, np.exp(coef[1])*np.exp(coef[0]*x), 'T[K]', 'R [ohm] ')
