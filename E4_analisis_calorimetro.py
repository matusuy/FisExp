# -*- coding: utf-8 -*-
"""
Física Experimental 1 - Física Experimental 2
Instituto de Física - Facultad de Ingeniería - Universidad de la República

Script para realizar el análisis termodinámico del proceso de intercambio de calor
entre dos cuerpos en un calorímetro

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
from support_funcs import min_cuad

try:
    import IPython
    shell = IPython.get_ipython()
    shell.enable_matplotlib(gui='qt')
except:
    pass   

# ---------------------------------- INPUT ------------------------------

# Data with noise -> Input from students
# Tiempos medidos con cronómetro (en s)

# Valor de las masas utilizadas (en kg)
m = np.array([0.1,
              0.2,
              0.3,
              0.4,
              0.5]);

dm = 0.01;

# Temperaturas finales medidas para cada masa utilizada (en K)
Tf = np.array([303,
              309,
              322,
              331,
              339]);

dTf = 0.5
# ------------------------------- FIN INPUT ------------------------------

plt.figure(1)
plt.plot(m,Tf,'*')
plt.xlabel('masa [kg]',fontweight='bold',fontsize=12)
plt.ylabel('temperatura final [K] ',fontweight='bold',fontsize=12)
plt.title('Datos medidos',fontweight='bold',fontsize=14)

# Reorganizacion de datos para poder realizar la linealizacion
# Modelo: Tf(m)=A+B*m

# Proceso de Fit por Mínimos Cuadrados -> salida: coef (coeficientes), cov (matriz de covarianza)
coef = min_cuad(m,Tf,1,'masa inicial [kg]','temperatura final [K]')


