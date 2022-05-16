# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:45:12 2022

@author: lauta
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from bcch import BancoCentralDeChile
import os

bcch_user = os.environ['BCCH_USER']
bcch_pwd = os.environ['BCCH_PWD']

# Creación de la instancia
client = BancoCentralDeChile(bcch_user, bcch_pwd)

def cleaner(serie:str, resam:str=None, operations:list=None):
    """
    Limpiar la serie proveniente del la API y dejarla lista para ocupar
    Parameters
    ----------
    serie : str
        id de la serie macro a solicitar.
    resam : str
        frecuencia para el resampling.
    operation : list
        operación(es) para agregar los datos resampliandos.
    Returns
    -------
    pandas DataFrame
        serie lista para ocupar.
    """
    serie_ = pd.DataFrame(client.get_macro(serie=serie))
    serie_['value'] = pd.to_numeric(serie_['value'], errors='coerce')
    serie_['indexDateString'] = pd.to_datetime(serie_['indexDateString'], format='%d-%m-%Y')
    serie_.set_index('indexDateString', inplace=True)
    del serie_['statusCode']
    
    if resam is not None:
        if operations is not None:
            serie_ = serie_.resample(resam).agg(operations)
            # renombrar las columnas
            serie_.columns = ['_'.join(x) for x in serie_.columns]
            return serie_
        else:
            print('Ocupar ')
    else:
        return serie_
    
def sma(values, window):
    """
    Promedio movil simple de serie de datos
    Parameters
    ----------
    values : pd.Series o numpy array
        serie de datos tipo float.
    window : int
        cuantos valor a ocupar para el calculo.
    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    weights = np.repeat(1.0, window)/window
    return np.convolve(a=values, v=weights, mode='valid')

def ema(values, window):
    weights = np.exp(np.linspace(-1.0, 0.0, window))
    weights /= weights.sum()
    a = np.convolve(values, weights)[:len(values)]
    a[:window] = a[window]
    return a

def ma(values, window:int=4, mode:str='ema'):
    
    if mode == 'sma':
        return sma(values, window)
    elif mode == 'ema':
        return ema(values, window)
    
"""
	• Safe Heaven demand: Retornos de los últimos 20 días del del Dólar vs los 
    retornos del IPSA, donde se mida en un ratio.
    
		○ Tipo de cambio del dólar observado diario -> F073.TCO.PRE.Z.D
		○ Índice de Precios Selectivo de Acciones, IPSA- Valor nominal - Base: 
		30 de diciembre de 2002 =1000 -> F013.IBC.IND.N.7.LAC.CL.CLP.BLO.D
		
	• Junk Bond Demand: Spread bonos soberanos publicados en la API del BCCh
    
		○ Spread EMBI Chile (promedio, puntos base) -> F019.SPS.PBP.91.D
		
	• Momentum del mercado: Valor actual del IPSA vs su media móvil de 125 días
    (6 meses) o comprar la media móvil de 20 días vs la de 60 días
    (1 mes vs 1 trimestre). 
    
		○ Índice de Precios Selectivo de Acciones, IPSA- Valor nominal - Base: 
		30 de diciembre de 2002 =1000 -> F013.IBC.IND.N.7.LAC.CL.CLP.BLO.D
		
	• Volatilidad: Comparación de la variabilidad en los retornos del IPSA en 
    ratio de 20 vs 60 días (1 mes vs 1 trimestre).
    
		○ Índice de Precios Selectivo de Acciones, IPSA- Valor nominal - Base: 
		30 de diciembre de 2002 =1000 -> F013.IBC.IND.N.7.LAC.CL.CLP.BLO.D
		
	• Redes sociales: Indice de Twitter calculado por el BCCh. Podría ser un
    promedio junto al indice de incertidumbre política. Habría que conjugar 
    las diferentes temporalidades (diaria vs mensual).
    
    ○ Índice de Diario de Incertidumbre Económica -> F029.IDIE.IND.TOT.D
"""

#%% Safe Heaven demand

# https://stackoverflow.com/a/31287674/6509794

dolar = cleaner('F073.TCO.PRE.Z.D').fillna(method='ffill')
ipsa = cleaner('F013.IBC.IND.N.7.LAC.CL.CLP.BLO.D').fillna(method='ffill')

dolar_ = np.log(dolar) - np.log(dolar.shift(20))
ipsa_ = np.log(ipsa) - np.log(ipsa.shift(20))

fng = pd.DataFrame()
fng['safe_heaven'] = (dolar_ - ipsa_) * 100

# Graficar los datos
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(fng['safe_heaven'], color='tab:blue')
ax.grid(True, linestyle='--')
fig.suptitle('Dolar menos IPSA', fontweight='bold')
plt.title('Retornos mensuales')
ax.set_ylabel('')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Graph source
ax.text(0.2, -0.12,  
         "Fuente: Banco Central de Chile   Gráfico: Lautaro Parada", 
         horizontalalignment='center',
         verticalalignment='center', 
         transform=ax.transAxes, 
         fontsize=8, 
         color='black',
         bbox=dict(facecolor='tab:gray', alpha=0.5))

plt.show()

del dolar, ipsa, dolar_, ipsa_

#%% Junk Bond demand

fng['spreads_junk'] = cleaner('F019.SPS.PBP.91.D').fillna(method='ffill')

# Graficar los datos
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(fng['spreads_junk'], color='tab:blue')
ax.grid(True, linestyle='--')
fig.suptitle('Spread bonos soberanos', fontweight='bold')
plt.title('')
ax.set_ylabel('promedio, puntos base')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Graph source
ax.text(0.2, -0.12,  
         "Fuente: Banco Central de Chile   Gráfico: Lautaro Parada", 
         horizontalalignment='center',
         verticalalignment='center', 
         transform=ax.transAxes, 
         fontsize=8, 
         color='black',
         bbox=dict(facecolor='tab:gray', alpha=0.5))

plt.show()

#%% Momentum del mercado 

ipsa = cleaner('F013.IBC.IND.N.7.LAC.CL.CLP.BLO.D').fillna(method='ffill')

ipsa_125 = ipsa.rolling(window=125, min_periods=125).mean()

fng['momentum'] = ipsa - ipsa_125

# Graficar los datos
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(fng['momentum'], color='tab:blue')
ax.grid(True, linestyle='--')
fig.suptitle('Momentum', fontweight='bold')
plt.title('')
ax.set_ylabel('')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Graph source
ax.text(0.2, -0.12,  
         "Fuente: Banco Central de Chile   Gráfico: Lautaro Parada", 
         horizontalalignment='center',
         verticalalignment='center', 
         transform=ax.transAxes, 
         fontsize=8, 
         color='black',
         bbox=dict(facecolor='tab:gray', alpha=0.5))

plt.show()

del ipsa, ipsa_125

#%% Volatilidad

# https://stackoverflow.com/a/60669752/6509794
ipsa = cleaner('F013.IBC.IND.N.7.LAC.CL.CLP.BLO.D').fillna(method='ffill')

ipsa_vol_20 = ipsa.rolling(window=20, min_periods=20).std(ddof=0)
ipsa_vol_60 = ipsa.rolling(window=60, min_periods=60).std(ddof=0)

fng['volatilidad'] = ipsa_vol_20 - ipsa_vol_60

# Graficar los datos
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(fng['volatilidad'], color='tab:blue')
ax.grid(True, linestyle='--')
fig.suptitle('volatilidad', fontweight='bold')
plt.title('')
ax.set_ylabel('')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Graph source
ax.text(0.2, -0.12,  
         "Fuente: Banco Central de Chile   Gráfico: Lautaro Parada", 
         horizontalalignment='center',
         verticalalignment='center', 
         transform=ax.transAxes, 
         fontsize=8, 
         color='black',
         bbox=dict(facecolor='tab:gray', alpha=0.5))

plt.show()

del ipsa, ipsa_vol_20, ipsa_vol_60

#%% Redes sociales (sentimiento)

fng['sentimiento'] = cleaner('F029.IDIE.IND.TOT.D').fillna(method='ffill')

# Graficar los datos
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(fng['sentimiento'], color='tab:blue')
ax.grid(True, linestyle='--')
fig.suptitle('sentimiento', fontweight='bold')
plt.title('')
ax.set_ylabel('')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Graph source
ax.text(0.2, -0.12,  
         "Fuente: Banco Central de Chile   Gráfico: Lautaro Parada", 
         horizontalalignment='center',
         verticalalignment='center', 
         transform=ax.transAxes, 
         fontsize=8, 
         color='black',
         bbox=dict(facecolor='tab:gray', alpha=0.5))

plt.show()

#%% Rescalar las variables

fng.dropna(inplace=True)

def rescale(values, new_min:int=0, new_max:int=100):
    
    old_max, old_min = max(values), min(values)
    return [(new_max - new_min) / (old_max - old_min) * (x - old_min) + new_min for x in values]

fng['res_safe_heaven'] = rescale(fng['safe_heaven'])
fng['res_spreads_junk'] = rescale(fng['spreads_junk'])
fng['res_momentum'] = rescale(fng['momentum'])
fng['res_volatilidad'] = rescale(fng['volatilidad'])
fng['res_sentimiento'] = rescale(fng['sentimiento'])

fng['index'] = fng['res_safe_heaven']*0.13 + fng['res_spreads_junk']*0.13 + fng['res_momentum']*0.28 + fng['res_volatilidad']*0.28 + fng['res_sentimiento']*0.18

plt.plot(fng['index'])