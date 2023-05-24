# -*- coding: utf-8 -*-
"""
Instituto de Física - Facultad de Ingeniería - Universidad de la República
This code is licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) license.

Script para realizar análisis estadísticos para una serie de datos

Entradas (input): serie de datos a analizar
Salidas (output): estadística de la serie

@author: mosorio
@date: 202305
@email: mosorio@fing.edu.uy
"""

# Librerías básicas
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import norm

# Librería creada para el curso
from support_funcs import criterio_descarte as criterio_descarte
from support_funcs import estadistica_datos as estadistica_datos
from support_funcs import graficar_datos as graficar_datos
from support_funcs import graficar_histograma as graficar_histograma

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
t = np.random.normal(10,1,100)
#----------------------------

# Generate data following a Gaussian distribution with noise
mean = 0  # Mean of the Gaussian distribution
std = 1   # Standard deviation of the Gaussian distribution
num_data_points = 100
data = np.random.normal(mean, std, num_data_points)

# Add noise to the data
noise_std = 0.2  # Standard deviation of the noise
data_with_noise = data + np.random.normal(0, noise_std, num_data_points)

# Add outliers with different degrees of separation
num_outliers = 10
outlier_degree_of_separation = [2.0, 3.0, -2.5, 4.0, -3.5, 2.7, -1.8, 3.2, -2.9, 4.5]

outlier_indices = np.random.choice(range(num_data_points), num_outliers, replace=False)
data_with_outliers = data_with_noise.copy()
for i, idx in enumerate(outlier_indices):
    data_with_outliers[idx] += outlier_degree_of_separation[i]
    
t = data_with_outliers
#----------------------------

# Cantidad de bins para graficar el histograma
nBins = 10;

# ------------------------------- FIN INPUT ------------------------------

# Figura donde se muestran los datos obtenidos
graficar_datos(t,'número de medida', 'datos')

# Análisis estadístico
estadistica_datos(t)

# Histograma de las mediciones junto a la campana de Gauss
graficar_histograma(t, nBins, 'longitud[cm]', 'frecuencia absoluta')
    
############ Descarte de datos ############
# Cantidad de veces a realizar
nIteraciones = 1
# Cantidad de sigmas respecto de la media
sigmas = 2

for i in range(0,nIteraciones):
    criterio = sigmas*np.std(t);
    t = criterio_descarte(t, criterio)

############ Post Descarte de datos ############

# Figura donde se muestran los datos obtenidos
graficar_datos(t,'número de medida', 'datos')

# Análisis estadístico
estadistica_datos(t)

# Histograma de las mediciones junto a la campana de Gauss
graficar_histograma(t, nBins, 'longitud[cm]', 'frecuencia absoluta')
