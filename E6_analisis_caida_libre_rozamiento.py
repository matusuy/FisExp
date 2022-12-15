# -*- coding: utf-8 -*-
"""
Física Experimental 1 - Física Experimental 2
Instituto de Física - Facultad de Ingeniería - Universidad de la República

Script para realizar el análisis del movimiento de caída libre con rozamiento
del aire

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

#----------------------------
# For develop purposes only
filename = "C:/Users/mosorio/Desktop/GIT/FisExp/files_doNotUpload2GIT/B_agua.txt"
data = np.loadtxt(filename)
#----------------------------

# Definicion de columnas para cada variable medida: tiempo, posición, velocidad, aceleración
t = 0; dx = 1; dv = 2; da = 3;
# ------------------------------- FIN INPUT ------------------------------

# Figura donde se muestran los datos obtenidos
plt.figure(1)
plt.plot(data[:,t],data[:,dv],'*',label='datos medidos')
plt.xlabel('tiempo [s]',fontweight='bold')
plt.ylabel('velocidad [m/s]',fontweight='bold')
plt.legend(loc="upper right")
#plt.xticks(np.arange(1, np.size(data[:,t])+1, np.size(data[:,t])/10))
plt.show()

#ideas: https://www.wired.com/2017/04/lets-study-air-resistance-coffee-filters/