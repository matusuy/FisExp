# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 22:38:22 2022

@author: mosorio
"""
def criterio_descarte(x, d): 
    """función que aplica un criterio de descarte. todos los datos de x que estén
       a una distancia d mayor del promedio de los datos
    
    input : x - vector de datos
            d - distancia a los datos
    
    output : vector de datos descartados
    @author: mosorio
    @date: 2212
    @email: mosorio@fing.edu.uy """
    
    import numpy as np
    
    mask = abs(x-np.mean(x))<d
    return x[mask]