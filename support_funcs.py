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
    """función que aplica el método de mínimos cuadrados para un set de datos
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
    """función que grafica un conjunto de datos, junto a sus incertidumbres
    
    input : x - eje X de los datos a graficar
            y - eje Y de los datos a graficar
            error_x - incertidumbre en el eje X
            error_y - incertidumbre en el eje Y
            label_x - etiqueta de eje X
            label_y - etiqueta de eje Y
    
    output : figura del conjunto de datos junto a sus correspondientes incertidumbres
    
    @author: mosorio
    @date: 202305
    @email: mosorio@fing.edu.uy """
    
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(x,y,'*')
    plt.xlabel(label_x,fontweight='bold',fontsize=12)
    plt.ylabel(label_y,fontweight='bold',fontsize=12)
    
def graficar_datos_con_modelo(x, y, error_x, error_y, modelo_x, modelo_y, label_x, label_y):
    """función que grafica un conjunto de datos, junto a sus incertidumbres y a un determinado modelo
    
    input : x - eje X de los datos a graficar
            y - eje Y de los datos a graficar
            error_x - incertidumbre en el eje X
            error_y - incertidumbre en el eje Y
            modelo_x - eje X del modelo
            modelo_Y - eje Y del modelo
            label_x - etiqueta de eje X
            label_y - etiqueta de eje Y
    
    output : figura del conjunto de datos junto a sus correspondientes incertidumbres
    
    @author: mosorio
    @date: 202305
    @email: mosorio@fing.edu.uy """
    
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(x,y,'*',label='datos exp.')
    plt.plot(modelo_x, modelo_y, '--', label='modelo teórico')
    plt.xlabel(label_x,fontweight='bold',fontsize=12)
    plt.ylabel(label_y,fontweight='bold',fontsize=12)
    plt.legend()
    
def estadistica_datos(datos):
    """función que despliega en pantalla información estadística básica de un conjunto de datos
      
    input : datos - datos bajo análisis
    
    output : parámetros estadísticos de los datos de interés
    
    @author: mosorio
    @date: 202305
    @email: mosorio@fing.edu.uy """
    
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
    """función que grafica un histograma de los datos que se pasan como parámetros
       junto a una curva Gaussiana generada a través de la estadística de los mismos
    
    input : datos - datos bajo análisis
            clases - cantidad de clases en que se quiere fraccionar el histograma
            orden - orden del polinomio
            label_x - etiqueta de eje X
            label_y - etiqueta de eje Y
    
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
    x = np.linspace(min(datos)*0.9, max(datos)*1.1, 200)
    y = norm.pdf(x, mu, sigma) * bin_width * cant_medidas

    # Grafica de histograma
    plt.plot(x, y, 'r-', linewidth=2, label='Curva Gauss.')
    plt.xlabel('Datos experimentales', fontweight='bold',fontsize=12)
    plt.ylabel('Frecuencia absoluta', fontweight='bold',fontsize=12)
    plt.legend()
    plt.show()
    
def seleccion_manual_datos(datos):
    """función que dado un conjunto de datos, selecciona algunos para su posterior procesamiento

    input : datos - datos bajo análisis

    output : datos_x - eje X de los datos seleccionados
             datos_y - eje Y de los datos seleccionados
             
    @author: mosorio
    @date: 202305
    @email: mosorio@fing.edu.uy """

    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(datos)
    plt.show()

    points = plt.ginput(n=-1, timeout=0)
    datos_x = [point[0] for point in points]
    datos_y = [point[1] for point in points]
    
    for x, y in zip(datos_x, datos_y):
        print(f"Dato seleccionado: x={x}, y={y}")

    return datos_x, datos_y
    

    