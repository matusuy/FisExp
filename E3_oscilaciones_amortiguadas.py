# -*- coding: utf-8 -*-
"""
Instituto de Física - Facultad de Ingeniería - Universidad de la República
This code is licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) license.

Script para realizar el análisis del movimiento amortiguado a partir de mediciones del movimiento del mismo

Entradas (input): 
Salidas (output): 

@author: mosorio
@date: 202212
@email: mosorio@fing.edu.uy
"""

# Librerías básicas
import numpy as np
import matplotlib.pyplot as plt

# Librería creada para el curso
from support_funcs import min_cuad as min_cuad
from support_funcs import graficar_datos as graficar_datos
from support_funcs import seleccion_manual_datos as seleccion_manual_datos

try:
    import IPython
    shell = IPython.get_ipython()
    shell.enable_matplotlib(gui='qt')
except:
    pass   

# ---------------------------------- INPUT ------------------------------

# Data with noise -> Input from students
# Tiempos medidos con cronómetro (en s)
#t = np.array([10.25,10.90,9.94,14.3,8.63]);

#----------------------------
# For develop purposes only
filename = "C:/Users/mosorio/Desktop/GIT/FisExp/files_doNotUpload2GIT/B_agua.txt"
data = np.loadtxt(filename)
#----------------------------

# Valor de constante de gravedad (en m/s^2)
g = 9.8;

# Valor de constante del resorte (en N/m)
k = 5;

# Valor de la masa acoplada (en kg)
m = 0.3;
dm= 0.01;

# Definicion de columnas para cada variable medida: tiempo, posición, velocidad, aceleración
t = 0; dx = 1; dv = 2; da = 3;

# ------------------------------- FIN INPUT ------------------------------

# Encontrar el valor Z0 de offset de posición tomando los últimos 50 puntos
Z0 = np.mean(data[-50,dx])

# Chequeo de seguridad: Z0 es la posición de equilibrio y debe ser igual a 
# la coordenada hallada cuando se iguala la ecuación de movimiento a cero
print('==============================================================')
print('Análisis de coordenadas de equilibrio halladas:')
print('--------------------------------------------------------------')
print('Z0 equilibrio a partir de los datos obtenidos [m]: ' + str(Z0));
print('Z0 equilibrio a partir de la ecuación de movimiento [m]: ' + str(m*g/k));
print('==============================================================')

graficar_datos(data[:,dx], 'tiempo [s]', 'posición [m]', datos_x=data[:,t])

# Corrección para extraer Z0 y así centrar el movimiento alrededor de la 
# posición de equilibrio del sistema
dx0 = data[:,dx] - Z0;

# Obtención de máximos a través del análisis gráficos de los picos
# https://matplotlib.org/stable/gallery/event_handling/ginput_manual_clabel_sgskip.html#sphx-glr-gallery-event-handling-ginput-manual-clabel-sgskip-py
graficar_datos(dx0, 'tiempo [s]', 'posición [m]', datos_x=data[:,t])

picos = seleccion_manual_datos(data[:,t], dx0)

print('==============================================================')
print('Datos seleccionados: ')
print(' ')
print('Columna 1: Posición [m] | Columna 2: Tiempo [s]')
print(picos)
print('==============================================================')

#### Regresión lineal para los datos obtenidos a través de los máximos/mínimos


# Reorganizacion de datos para poder realizar la linealizacion
# Modelo: A(T)=Ao*exp(-t/tau) -> CV: y=ln(A(T)); x=t --> y = ln(Ao)+B*x
y = np.log(picos[:,1]);
x = picos[:,0];

plt.figure(2)
plt.plot(x,y,'*')
plt.xlabel('t [s]',fontweight='bold',fontsize=12)
plt.xticks(rotation=50)
plt.ylabel('ln(A(t)) [u.a.]',fontweight='bold',fontsize=12)
plt.title('Proceso de linealización',fontweight='bold',fontsize=14)

# Proceso de Fit por Mínimos Cuadrados -> salida: coef (coeficientes), cov (matriz de covarianza)
coef = min_cuad(x,y,1,'t [s]','ln(A(T)) [u.a.]')

# Inversión del modelo a partir de los coeficientes obtenidos
plt.figure(4)
plt.plot(data[:,t],dx0,'*',label='datos')
plt.plot(data[:,t],np.exp(coef[1])*np.exp(coef[0]*data[:,t]),label='modelo obtenido')
plt.xlabel('t[s]',fontweight='bold',fontsize=12)
plt.ylabel('A(t) [m] ',fontweight='bold',fontsize=12)
plt.legend()
plt.title('Modelo obtenido',fontweight='bold',fontsize=14)

