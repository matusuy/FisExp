# -*- coding: utf-8 -*-
"""
Física Experimental 1 - Física Experimental 2
Instituto de Física - Facultad de Ingeniería - Universidad de la República

Script para realizar análisis estadísticos para la medición del valor de la 
gravedad en Montevideo a partir del análisis del movimiento de un péndulo

Entradas (input): largo del péndulo L (en m), tiempo medido t (en s), 
                  cantidad de períodos observados T, masa M (en kg)
Salidas (output): valor de gravedad g 

@author: mosorio
@date: 202212
@email: mosorio@fing.edu.uy
"""
# Librerías básicas
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import norm

# Librería creada para el curso
from support_funcs import criterio_descarte

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
# Incertidumbre cronómetro
dt = 0.1;

#----------------------------
# Hay que hacer otra para el sensor!!!!!
#----------------------------


# Cantidad de períodos observados
T = 10;

# Tiempo por período = (tiempos medidos)/(cantidad de períodos observados)
t = t/T;

# Largo del péndulo (en metros)
L = 0.25;
# Incertidumbre del largo del péndulo
dL = 0.01;

# Masa acoplada  (en kg)
m = 0.5;
# Incertidumbre de la masa acoplada
dm = 0.01;

# Cantidad de bins para graficar el histograma
noBins = 15;

# ------------------------------- FIN INPUT ------------------------------

# Figura donde se muestran los datos obtenidos
plt.figure(1)
plt.plot(np.arange(1, np.size(t)+1, 1.0),t,'*')
plt.xlabel('número de medición',fontweight='bold')
plt.ylabel('período del movimiento [s]',fontweight='bold')
plt.xticks(np.arange(1, np.size(t)+1, np.size(t)/T))

# Análisis estadístico
print('==============================================================')
print('Análisis estadístico de los datos de período obtenidos')
print('--------------------------------------------------------------')
print('Cantidad de datos obtenidos: ' + str(np.size(t)))
print('Promedio de los datos obtenidos [s]: ' + str(np.mean(t)))
print('Desviación estándar de los datos obtenidos [s]: ' + str(np.std(t)))
print('Skew de la muestra: ' + str(skew(t)))
print('--------------------------------------------------------------')     
print('Información básica del skew:')
print(' ')
print('Si el skew es nulo o muy cercano a cero, la muestra se distribuye de manera normal')
print(' ')
print('Si skew > 0, los datos están más distribuidos hacia la parte izquierda de la cola de la distribución')
print(' ')
print('Si skew < 0, los datos están más distribuidos hacia la parte derecha de la cola de la distribución')
print('==============================================================')
print(' ')

# Histograma de las mediciones junto a la campana de Gauss
plt.figure(2)
plt.hist(t,bins=noBins)
# Dibujo campana de Gauss
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 200)
p = norm.pdf(x, np.mean(t), np.std(t))
plt.plot(x, p*noBins, 'k', linewidth=2)
plt.xlabel('período del movimiento [s]',fontweight='bold')
plt.ylabel('frecuencia de ocurrencia',fontweight='bold')
plt.show()

############ Descarte de datos ############
descarte = 1;

while int(descarte):
    criterio = 2*np.std(t);
    t = criterio_descarte(t,criterio);

    # Figura donde se muestran los datos finales
    plt.figure(3)
    plt.plot(np.arange(1, np.size(t)+1, 1.0),t,'*')
    plt.xlabel('número de datos',fontweight='bold')
    plt.ylabel('período del movimiento [s]',fontweight='bold')
    plt.xticks(np.arange(1, np.size(t)+1, np.size(t)/T))
    plt.title('gráfica de datos luego del criterio de descarte')
    plt.show()
    
    # Histograma de las mediciones junto a la campana de Gauss
    plt.figure(4)
    plt.hist(t,bins=noBins)
    # Dibujo campana de Gauss
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 200)
    p = norm.pdf(x, np.mean(t), np.std(t))
    plt.plot(x, p*noBins, 'k', linewidth=2)
    plt.xlabel('período del movimiento [s]',fontweight='bold')
    plt.ylabel('frecuencia de ocurrencia',fontweight='bold')
    plt.show()
    
    descarte=input("Es necesario más descarte de datos? (1-SI || 0-NO): ")

    
############ Cálculo de g ############
#### Método 1: para cada valor de período obtenido (t_i) se haya g_i 
#### y luego se promedian todos los datos obteniendo g

# Para cada valor de período obtenido, se calcula un valor de g, dados el largo del péndulo y la masa acoplada
g_i = ((2*np.pi/np.square(t))**2)*L

# Análisis estadístico del conjunto de valores individuales de g hallados
print('==============================================================')
print('Método 1 de cálculo de g')
print('Análisis estadístico de valores individuales de g')
print('--------------------------------------------------------------')
print('Cantidad de datos obtenidos: ' + str(np.size(g_i)))
print('Promedio de los datos obtenidos [m s^-2]: ' + str(np.mean(g_i)))
print('Desviación estándar de los datos obtenidos [m s^-2]: ' + str(np.std(g_i)))
print('Skew de la muestra: ' + str(skew(g_i)))
print('--------------------------------------------------------------')     
print('Información básica del skew:')
print(' ')
print('Si el skew es nulo o muy cercano a cero, la muestra se distribuye de manera normal')
print(' ')
print('Si skew > 0, los datos están más distribuidos hacia la parte izquierda de la cola de la distribución')
print(' ')
print('Si skew < 0, los datos están más distribuidos hacia la parte derecha de la cola de la distribución')
print('==============================================================')
print(' ')

# Figura donde se muestran los valores de g obtenidos
plt.figure(5)
plt.plot(np.arange(1, np.size(g_i)+1, 1.0),g_i,'*')
plt.xlabel('número de medición',fontweight='bold')
plt.ylabel('valor de g local [m s^-2]',fontweight='bold')
plt.xticks(np.arange(1, np.size(g_i)+1, np.size(t)/T))

# Histograma de las mediciones junto a la campana de Gauss
plt.figure(6)
plt.hist(g_i,bins=noBins)
# Dibujo campana de Gauss
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 200)
p = norm.pdf(x, np.mean(g_i), np.std(g_i))
plt.plot(x, p*noBins, 'k', linewidth=2)
p = np.random.poisson(9.8,np.size(t))
plt.plot(x, p, 'r', linewidth=2)
plt.xlabel('valor de g local [m s^-2]',fontweight='bold')
plt.ylabel('frecuencia de ocurrencia',fontweight='bold')
plt.show()

#### Método 2: se promedian todos los períodos obtenidos y se calcula un único 
#### valor de g para el largo del péndulo y la masa acoplada

g = ((2*np.pi/np.square(np.mean(t)))**2)*L;

print('==============================================================')
print('Método 2 de cálculo de g')
print(' ')
print('--------------------------------------------------------------')
print('Valor de g hallado [m s^-2]: ' + str(g));
print('==============================================================')

#FALTA DIBUJAR UNA POISSON ARRIBA DEL HISTOGRAMA DE ALGUAN COSA