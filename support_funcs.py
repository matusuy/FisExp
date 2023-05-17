# -*- coding: utf-8 -*-
"""
support_funcs

librería de soporte

@author: mosorio
"""
def criterio_descarte(x, d): 
    """función que aplica un criterio de descarte. todos los datos de x que estén
       a una distancia d mayor del promedio de los datos
    
    input : x - vector de datos
            d - distancia a los datos
    
    output : vector de datos descartados
    @author: mosorio
    @date: 202212
    @email: mosorio@fing.edu.uy """
    
    import numpy as np
    
    mask = abs(x-np.mean(x))<d
    return x[mask]

def min_cuad(x,y,orden,labelX,labelY):
    """función que aplica el método de mínimos cuadrados para un set de datos.
       La función a fitear es polinómica de cierto orden
    
    input : x - eje X de datos
            y - eje Y de datos 
            orden - orden del polinomio
            labelX - etiqueta de eje X
            labelY - etiqueta de eje Y
    
    output : información de interés de la regresión + coef: coeficientes obtenidos
    @author: mosorio
    @date: 202212
    @email: mosorio@fing.edu.uy """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Proceso de Fit por Mínimos Cuadrados -> salida: coef (coeficientes), cov (matriz de covarianza)
    coef,cov = np.polyfit(x,y,deg=orden,cov=True); 
    
    if orden==1:
        print('==================================')
        print('Resultados de la regresión lineal:')
        print('Pendiente: ' + str(coef[0]) + ' +- ' + str(np.sqrt(np.diag(cov))[0]))
        print('Ordenada: ' + str(coef[1]) + ' +- ' + str(np.sqrt(np.diag(cov))[1]))
        print('R2: ' + str(np.corrcoef(x, y)[0,1]*np.corrcoef(x, y)[0,1]))
        print('==================================')
    else:
        print('==================================')
        print('Resultados de la regresión lineal:')
        for i in range(0,np.size(coef),1):
            print('Coef. orden ' + str(i) + ' : ' + str(coef[np.size(coef)-i-1]))
        print('==================================')

    plt.figure(10)
    plt.plot(x,y,'*',label='datos')
    plt.plot(x,np.polyval(coef,x),label='fit lineal')
    plt.xticks(rotation=50)
    plt.xlabel(labelX,fontweight='bold',fontsize=12)
    plt.ylabel(labelY,fontweight='bold',fontsize=12)
    plt.title('Proceso de linealización',fontweight='bold',fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return coef

def graficar_datos(x, y, error_x, error_y, label_x, label_y):
    
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(x,y,'*')
    plt.xlabel(label_x,fontweight='bold',fontsize=12)
    plt.ylabel(label_y,fontweight='bold',fontsize=12)
    
def graficar_datos_con_modelo(x, y, error_x, error_y, modelo_x, modelo_y, label_x, label_y):
    
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(x,y,'*',label='datos exp.')
    plt.plot(modelo_x, modelo_y, '--', label='modelo teórico')
    plt.xlabel(label_x,fontweight='bold',fontsize=12)
    plt.ylabel(label_y,fontweight='bold',fontsize=12)
    plt.legend()
    
def estadistica_datos(datos):
    
    import numpy as np
    from scipy.stats import skew
    
    # Análisis estadístico
    print('==============================================================')
    print('Análisis estadístico de los datos de período obtenidos')
    print('--------------------------------------------------------------')
    print('Cantidad de datos obtenidos: ' + str(np.size(datos)))
    print('Promedio de los datos obtenidos [s]: ' + str(np.mean(datos)))
    print('Desviación estándar de los datos obtenidos [s]: ' + str(np.std(datos)))
    print('Valor mínimo: ' + str(np.min(datos)))
    print('Valor máximo: ' + str(np.max(datos)))
    print('Skew de la muestra: ' + str(skew(datos)))
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
    
def graficar_histograma(datos, clases, label_x, label_y):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    # Se grafica histograma con la frecuencia absoluta
    counts, bins, _ = plt.hist(measurements, bins=clases, alpha=0.7, color='steelblue', edgecolor='black')

    # Ancho de bin
    bin_width = bins[1] - bins[0]

    # Estadística de datos para armar Gaussiana
    cant_medidas = len(datos)
    mu, sigma = norm.fit(datos)

    # Generate points on x-axis for Gaussian curve
    x = np.linspace(min(datos), max(datos), 100)
    y = norm.pdf(x, mu, std) * bin_width * total_measurements

    # Normalize the Gaussian curve to match the data
#    scaling_factor = counts.sum() / y.sum()
 #   y *= scaling_factor

    # Plot Gaussian curve
    plt.plot(x, y, 'r-', linewidth=2)

    # Set plot labels and title
    plt.xlabel('Measurement')
    plt.ylabel('Absolute Frequency')
    plt.title('Histogram with Normalized Gaussian Fit')

    # Display the plot
    plt.show()
    
# def graficar_boxplot:

    