# -*- coding: utf-8 -*-
"""
Instituto de Física - Facultad de Ingeniería - Universidad de la República
This code is licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) license.

support_funcs

Librería con funciones de soporte para resolver los scripts principales

@author: mosorio
@email: mosorio@fing.edu.uy
"""

def criterio_descarte(datos, umbral): 
    """función que aplica un criterio de descarte. todos los datos de x que estén
       a una distancia umbral mayor del promedio de los datos
    
    input : datos - vector de datos - requerido
            umbral - distancia a los datos - requerido
    
    output : vector de datos descartados
    
    @author: mosorio
    @date: 202212
    @email: mosorio@fing.edu.uy """
    
    import numpy as np
    
    mask = abs(datos-np.mean(datos))<umbral
    return datos[mask]

def min_cuad(datos_x, datos_y, orden, label_x, label_y):
    """función que aplica el método de mínimos cuadrados para un set de datos
       La función a fitear es polinómica de cierto orden
    
    input : x - eje X de datos - requerido
            y - eje Y de datos - requerido
            orden - orden del polinomio - requerido
            labelX - etiqueta de eje X - requerido
            labelY - etiqueta de eje Y - requerido
    
    output : información de interés de la regresión + coef: coeficientes obtenidos
    
    @author: mosorio
    @date: 202212
    @email: mosorio@fing.edu.uy """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Proceso de Fit por Mínimos Cuadrados -> salida: coef (coeficientes), cov (matriz de covarianza)
    coef, cov = np.polyfit(datos_x, datos_y, deg=orden, cov=True); 
    
    if orden==1:
        print('==================================')
        print('Resultados de la regresión lineal:')
        print('Pendiente: ' + str(coef[0]) + ' +- ' + str(np.sqrt(np.diag(cov))[0]))
        print('Ordenada: ' + str(coef[1]) + ' +- ' + str(np.sqrt(np.diag(cov))[1]))
        print('R2: ' + str(np.corrcoef(datos_x, datos_y)[0,1]*np.corrcoef(datos_x, datos_y)[0,1]))
        print('==================================')
    else:
        print('==================================')
        print('Resultados de la regresión lineal:')
        for i in range(0,np.size(coef),1):
            print('Coef. orden ' + str(i) + ' : ' + str(coef[np.size(coef)-i-1]))
        print('==================================')

    plt.figure()
    plt.plot(datos_x, datos_y,'*',label='datos')
    plt.plot(datos_x, np.polyval(coef,datos_x),label='fit lineal')
    plt.xticks(rotation=50)
    plt.xlabel(label_x,fontweight='bold',fontsize=12)
    plt.ylabel(label_y,fontweight='bold',fontsize=12)
    plt.title('Proceso de linealización',fontweight='bold',fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return coef

def graficar_datos(datos_y, label_x, label_y, **kwargs):
    """función que grafica un conjunto de datos, junto a sus incertidumbres
    
    input : datos_y - eje Y de los datos a graficar - requerido
            label_x - etiqueta de eje X - requerido
            label_y - etiqueta de eje Y - requerido
            datos_x - eje X de los datos a graficar - opcional
            error_x - incertidumbre en el eje X - opcional
            error_y - incertidumbre en el eje Y - opcional
    
    output : figura del conjunto de datos junto a sus correspondientes incertidumbres
    
    @author: mosorio
    @date: 202305
    @email: mosorio@fing.edu.uy """
    
    import matplotlib.pyplot as plt
    
    if 'datos_x' in kwargs:
        if 'error_y' in kwargs and 'error_x' in kwargs:
            plt.figure()
            plt.errorbar(kwargs['datos_x'], datos_y, yerr=kwargs['error_y'], xerr = kwargs['error_x'], marker='*')
            plt.xlabel(label_x,fontweight='bold',fontsize=12)
            plt.ylabel(label_y,fontweight='bold',fontsize=12)
        elif 'error_y' in kwargs:
            plt.figure()
            plt.errorbar(kwargs['datos_x'], datos_y, yerr=kwargs['error_y'], marker='*')
            plt.xlabel(label_x,fontweight='bold',fontsize=12)
            plt.ylabel(label_y,fontweight='bold',fontsize=12)
        else:
            plt.figure()
            plt.plot(kwargs['datos_x'], datos_y, '*')
            plt.xlabel(label_x,fontweight='bold',fontsize=12)
            plt.ylabel(label_y,fontweight='bold',fontsize=12)
    else:
        plt.figure()
        plt.plot(datos_y,'*')
        plt.xlabel(label_x,fontweight='bold',fontsize=12)
        plt.ylabel(label_y,fontweight='bold',fontsize=12)
        print('Warning: para la grafica no se tienen como entradas los datos en el eje horizontal o las incertidumbres de cada eje.')
    
def graficar_datos_con_modelo(datos_x, datos_y, modelo_x, modelo_y, label_x, label_y, **kwargs):
    """función que grafica un conjunto de datos, junto a sus incertidumbres y a un determinado modelo
    
    input : datos_x - eje X de los datos a graficar - requerido
            datos_y - eje Y de los datos a graficar - requerido
            modelo_x - eje X del modelo - requerido
            modelo_Y - eje Y del modelo - requerido
            label_x - etiqueta de eje X - requerido
            label_y - etiqueta de eje Y - requerido
            error_x - incertidumbre en el eje X - opcional
            error_y - incertidumbre en el eje Y - opcional
    
    output : figura del conjunto de datos junto a sus correspondientes incertidumbres
    
    @author: mosorio
    @date: 202305
    @email: mosorio@fing.edu.uy """
    
    import matplotlib.pyplot as plt
    
    plt.figure()
    if 'error_y' in kwargs and 'error_x' in kwargs:
        plt.errorbar(datos_x, datos_y, yerr=kwargs['error_y'], xerr=kwargs['error_x'], marker='*', label='datos exp.')
    elif 'error_y' in kwargs:
        plt.errorbar(datos_x, datos_y, yerr=kwargs['error_y'], marker='*', label='datos exp.')
    else:
        plt.plot(datos_x, datos_y, '*', label='datos exp.')
        print('Warning: para la grafica no se tienen como entradas las incertidumbres de cada eje.')
    plt.plot(modelo_x, modelo_y, '--', label='modelo teórico')
    plt.xlabel(label_x,fontweight='bold',fontsize=12)
    plt.ylabel(label_y,fontweight='bold',fontsize=12)
    plt.legend()
    
def estadistica_datos(datos):
    """función que despliega en pantalla información estadística básica de un conjunto de datos
      
    input : datos - datos bajo análisis - requerido
    
    output : parámetros estadísticos de los datos de interés
    
    @author: mosorio
    @date: 202305
    @email: mosorio@fing.edu.uy """
    
    import numpy as np

    print('==============================================================')
    print('Análisis estadístico de los datos de período obtenidos')
    print('--------------------------------------------------------------')
    print('Cantidad de datos obtenidos: ' + str(np.size(datos)))
    print('Promedio de los datos obtenidos: ' + str(np.mean(datos)))
    print('Desviación estándar de los datos obtenidos: ' + str(np.std(datos)))
    print('Valor mínimo: ' + str(np.min(datos)))
    print('Índice mínimo: ' + str(datos.argmin()))
    print('Valor máximo: ' + str(np.max(datos)))
    print('Índice máximo: ' + str(datos.argmax()))
    print('==============================================================')
    print(' ')
    
def graficar_histograma(datos, clases, label_x, label_y):    
    """función que grafica un histograma de los datos que se pasan como parámetros
       junto a una curva Gaussiana generada a través de la estadística de los mismos
    
    input : datos - datos bajo análisis - requerido
            clases - cantidad de clases en que se quiere fraccionar el histograma - requerido
            label_x - etiqueta de eje X - requerido
            label_y - etiqueta de eje Y - requerido
    
    output : figura del histograma junto a la curva Gaussiana correspondiente
    
    @author: mosorio
    @date: 202305
    @email: mosorio@fing.edu.uy """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    plt.figure()
    
    # Se grafica histograma con la frecuencia absoluta
    counts, bins, _ = plt.hist(datos, bins=clases, alpha=0.7, color='steelblue', edgecolor='black', label='Histograma')

    # Ancho de bin
    bin_width = bins[1] - bins[0]

    # Estadística de datos para armar Gaussiana
    cant_medidas = len(datos)
    mu, sigma = norm.fit(datos)

    # Generación de ejes X e Y para Gaussiana
    x = np.linspace(min(datos)*1.15, max(datos)*1.15, 200)
    y = norm.pdf(x, mu, sigma) * bin_width * cant_medidas

    # Grafica de histograma
    plt.plot(x, y, 'r-', linewidth=2, label='Curva Gauss.')
    plt.xlabel(label_x, fontweight='bold',fontsize=12)
    plt.ylabel(label_y, fontweight='bold',fontsize=12)
    plt.legend()
    plt.show()
    
def seleccion_manual_datos(datos_x, datos_y):
    """función que dado un conjunto de datos, selecciona algunos para su posterior procesamiento

    input : datos - datos bajo análisis - requerido

    output : datos_x - eje X de los datos seleccionados
             datos_y - eje Y de los datos seleccionados
             
    @author: mosorio
    @date: 202305
    @email: mosorio@fing.edu.uy """

    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(datos_x, datos_y, '*')
    plt.show()

    points = plt.ginput(n=-1, timeout=0)
    seleccion_x = [point[0] for point in points]
    seleccion_y = [point[1] for point in points]
    
    for x, y in zip(seleccion_x, seleccion_y):
        print(f"Dato seleccionado: x={x}, y={y}")

    return seleccion_x, seleccion_y
    

    