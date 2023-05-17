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

def graficar_datos(x, y, error_x, error_y, labelX, labelY):
    
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(x,y,'*',label='datos')
    plt.plot(T,np.exp(coef[1])*np.exp(coef[0]*x),label='modelo obtenido')
    plt.xlabel('T[K]',fontweight='bold',fontsize=12)
    plt.ylabel('R [ohm] ',fontweight='bold',fontsize=12)
    plt.legend()
    plt.title('Modelo obtenido',fontweight='bold',fontsize=14)
    
# def graficar_datos_con_modelo(x, y, error_x, error_y, labelX, labelY, x_modelo, y_modelo):
    
# def estadistica_datos():
    
# def graficar_histograma():
    
# def graficar_boxplot:

    