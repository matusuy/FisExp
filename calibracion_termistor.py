# -*- coding: utf-8 -*-
"""
Física Experimental 1 - Física Experimental 2
Instituto de Física - Facultad de Ingeniería - Universidad de la República

Script para ejecutar la calibración de un termistor a partir de las mediciones
de resistencia y temperatura.

Entradas (input): valores de resistencia, valores de temperatura
Salidas (output): parámetro B, parámetro Ro

@author: mosorio
@date: 221214
@email: mosorio@fing.edu.uy
"""
# Librerías para trabajar
import numpy as np
import matplotlib.pyplot as plt

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

plt.figure(1)
plt.plot(T,measData,'*')
plt.xlabel('T[K]',fontweight='bold',fontsize=12)
plt.ylabel('R [ohm] ',fontweight='bold',fontsize=12)
plt.legend()
plt.title('Datos medidos',fontweight='bold',fontsize=14)

# Reorganizacion de datos para poder realizar la linealizacion
# Modelo: R(T)=Ro*exp(B*(1/T-1/To)) -> CV: y=ln(R(T)); x=1/T-1/To --> y = ln(Ro)+B*x
y = np.log(measData);
x = (1/T)-1/To;

plt.figure(2)
plt.plot(x,y,'*')
plt.xlabel('1/T-1/To [1/K]',fontweight='bold',fontsize=12)
plt.xticks(rotation=50)
plt.ylabel('ln(R(T)) [u.a.]',fontweight='bold',fontsize=12)
plt.title('Proceso de linealización',fontweight='bold',fontsize=14)

# Proceso de Fit por Mínimos Cuadrados -> salida: coef (coeficientes), cov (matriz de covarianza)
coef,cov = np.polyfit(x,y,deg=1,cov=True); 

print('==================================')
print('Resultados de la regresión lineal:')
print('Pendiente: ' + str(coef[0]) + ' +- ' + str(np.sqrt(np.diag(cov))[0]))
print('Ordenada: ' + str(coef[1]) + ' +- ' + str(np.sqrt(np.diag(cov))[1]))
print('R2: ' + str(np.corrcoef(x, y)[0,1]*np.corrcoef(x, y)[0,1]))
print('==================================')


plt.figure(3)
plt.plot(x,y,'*',label='datos')
plt.plot(x,np.polyval(coef,x),label='fit lineal')
plt.xticks(rotation=50)
plt.xlabel('1/T-1/To [1/K]',fontweight='bold',fontsize=12)
plt.ylabel('ln(R(T)) [u.a.]',fontweight='bold',fontsize=12)
plt.title('Proceso de linealización',fontweight='bold',fontsize=14)
plt.legend()
plt.show()


# Inversión del modelo a partir de los coeficientes obtenidos
plt.figure(4)
plt.plot(T,measData,'*',label='datos')
plt.plot(T,np.exp(coef[1])*np.exp(coef[0]*x),label='modelo obtenido')
plt.xlabel('T[K]',fontweight='bold',fontsize=12)
plt.ylabel('R [ohm] ',fontweight='bold',fontsize=12)
plt.legend()
plt.title('Modelo obtenido',fontweight='bold',fontsize=14)