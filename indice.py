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

#%% Safe Heaven demand -> Relación inversa

# https://stackoverflow.com/a/31287674/6509794

dolar = cleaner('F073.TCO.PRE.Z.D').fillna(method='ffill')
ipsa = cleaner('F013.IBC.IND.N.7.LAC.CL.CLP.BLO.D').fillna(method='ffill')
ann_factor = 20 * 250

dolar_ = np.log(dolar) - np.log(dolar.shift(20))
ipsa_ =  np.log(ipsa) - np.log(ipsa.shift(20))

fng = pd.DataFrame()
fng['safe_heaven'] = (dolar_ / ipsa_).rolling(window=20, min_periods=20).median()

# Graficar los datos
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(fng['safe_heaven'], color='tab:blue')
ax.grid(True, linestyle='--')
fig.suptitle('Retornos del Dólar respecto al IPSA', fontweight='bold')
plt.title('Mediana movil de los ultimos 20 dias habiles')
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

del dolar, ipsa, dolar_, ipsa_, ann_factor

#%% Junk Bond demand -> Relación inversa

fng['spreads_junk'] = cleaner('F019.SPS.PBP.91.D').fillna(method='ffill')

# Graficar los datos
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(fng['spreads_junk'], color='tab:blue')
ax.grid(True, linestyle='--')
fig.suptitle('Spread bonos soberanos EMBI Chile', fontweight='bold')
plt.title('Indicador de riesgo país, elaborado por JP Morgan')
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
fig.suptitle('Momentum mercado Renta Variable chileno', fontweight='bold')
plt.title('IPSA y su promedio movil de 125 dias (6 meses)')
ax.set_ylabel('Peso chilenos (CLP)')
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

#%% Volatilidad -> Relación inversa

# https://stackoverflow.com/a/60669752/6509794
ipsa = cleaner('F013.IBC.IND.N.7.LAC.CL.CLP.BLO.D').fillna(method='ffill')

ipsa_vol_20 = ipsa.rolling(window=20, min_periods=20).std(ddof=0)
ipsa_vol_60 = ipsa.rolling(window=60, min_periods=60).std(ddof=0)
ann_factor = 20 * 250

fng['volatilidad'] = ((ipsa_vol_20 * ann_factor)  / (ipsa_vol_60 * ann_factor)).rolling(window=60, min_periods=60).median()

# Graficar los datos
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(fng['volatilidad'], color='tab:blue')
ax.grid(True, linestyle='--')
fig.suptitle('Volatilidad de la Renta variable chilena', fontweight='bold')
plt.title('Comparación entre los 20 y 60 dias, mediana trimestral del ratio')
ax.set_ylabel('Ratio')
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

del ipsa, ipsa_vol_20, ipsa_vol_60, ann_factor

#%% Redes sociales (sentimiento) -> Relación inversa

fng['sentimiento'] = cleaner('F029.IDIE.IND.TOT.D').fillna(method='ffill')

# Graficar los datos
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(fng['sentimiento'], color='tab:blue')
ax.grid(True, linestyle='--')
fig.suptitle('Sentimiento del mercado de Renta Variable', fontweight='bold')
plt.title('Índice diario de incertidumbre económica propuesto por Becerra y Sagner (2020)')
ax.set_ylabel('Indice')
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

from sklearn.preprocessing import MinMaxScaler

def rescale(data:pd.Series, invert:bool=True):
    scaler_ = MinMaxScaler(feature_range=(0,100))
    data = data.values.reshape(-1, 1)
    scaler_ = scaler_.fit(data)
    
    if invert:
        return 100 - scaler_.transform(data)
    else:
        return scaler_.transform(data)


fng['res_safe_heaven'] = rescale(fng['safe_heaven'])
fng['res_spreads_junk'] = rescale(fng['spreads_junk'])
fng['res_momentum'] = rescale(fng['momentum'], invert=False)
fng['res_volatilidad'] = rescale(fng['volatilidad'])
fng['res_sentimiento'] = rescale(fng['sentimiento'])

fng['index'] = fng['res_safe_heaven']*0.13 + fng['res_spreads_junk']*0.13 + fng['res_momentum']*0.28 + fng['res_volatilidad']*0.28 + fng['res_sentimiento']*0.18

# Graficar los datos
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(fng['index'], color='tab:blue')
fig.suptitle('Indice de miedo y codicia para el IPSA', fontweight='bold')
plt.title('Promedio ponderado de variables instrumentales')
# Codicia extrema
ax.fill_between(fng['index'].index, fng['index'].shape[0]*[75], fng['index'].shape[0]*[100], color='darkgreen', alpha=0.35)
# Codicia
ax.fill_between(fng['index'].index, fng['index'].shape[0]*[60], fng['index'].shape[0]*[75], color='limegreen', alpha=0.25)
# Neutral
ax.fill_between(fng['index'].index, fng['index'].shape[0]*[40], fng['index'].shape[0]*[60], color='gold', alpha=0.25)
# Miedo
ax.fill_between(fng['index'].index, fng['index'].shape[0]*[25], fng['index'].shape[0]*[40], color='lightcoral', alpha=0.25)
# Miedo extremo
ax.fill_between(fng['index'].index, fng['index'].shape[0]*[0], fng['index'].shape[0]*[25], color='darkred', alpha=0.35)

ax.set_ylabel('Indice')
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

del fig, ax

#%% Estudiar la correlación con el IPSA

ipsa = cleaner('F013.IBC.IND.N.7.LAC.CL.CLP.BLO.D')
fng = fng.join(ipsa).fillna(method='ffill')

# Calculando la regresión
model = np.poly1d(np.polyfit(fng['index'], fng['value'], 2))
polyline = np.linspace(1, 100, fng.shape[0])

# R2
#define function to calculate r-squared
def polyfit(x, y, degree):
    results = {}
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    #calculate r-squared
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['r_squared'] = ssreg / sstot

    return results

polyfit(x=fng['index'], y=fng['value'], degree=1)

# Graficar los datos
fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(fng['index'], fng['value'], color='tab:blue')
ax.plot(polyline, model(polyline), color='tab:orange')
ax.grid(True, linestyle='--')
fig.suptitle('Relación entre el índice y el IPSA', fontweight='bold')
plt.title(f"Datos al {fng.index[-1].strftime('%Y-%m-%d')}")
ax.set_xlabel('Indice de miedo y codicia')
ax.set_ylabel('Valores nominales del IPSA')

# Graph source
ax.text(0.2, -0.17,  
         "Fuente: Banco Central de Chile   Gráfico: Lautaro Parada", 
         horizontalalignment='center',
         verticalalignment='center', 
         transform=ax.transAxes, 
         fontsize=8, 
         color='black',
         bbox=dict(facecolor='tab:gray', alpha=0.5))

plt.show()