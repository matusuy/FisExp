#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 18:33:50 2023

@author: mosorio
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fit_function(t, A, tau, omega, phi):
    """
    Function: y(t) = A * exp(-t/tau) * sin(omega*t + phi)
    """
    return A * np.exp(-t/tau) * np.sin(omega*t + phi)

def fit_parameters(t, data):
    """
    Fit the function to the passed data and return the fitted parameters.

    Parameters:
    - t: Time values
    - noisy_data: Noisy data corresponding to time values

    Returns:
    - Fitted parameters (A, tau, omega, phi)
    """
    initial_guess = [1.0, 1.0, 1.0, 0.0]

    parameters, covariance = curve_fit(fit_function, t, data, p0=initial_guess)

    return parameters

# Generate time values
t = np.linspace(0, 10, 1000)

# Datos obtenidos
data = generate_noisy_data(t, A, tau, omega, phi)

# Ajuste por mínimos cuadrados de los datos
fitted_parameters = fit_parameters(t, data)

# Graficas
plt.figure(figsize=(10, 6))
plt.scatter(t, data, color='red', label='Datos adquiridos')
plt.plot(t, fit_function(t, *fitted_parameters), label='Ajuste', linestyle='--')
plt.xlabel('Tiempo [s]')
plt.ylabel('Posición [m]')
plt.legend()
plt.show()

# Display
print("Parámetros:")
print("A:", fitted_parameters[0])
print("tau:", fitted_parameters[1])
print("omega:", fitted_parameters[2])
print("phi:", fitted_parameters[3])