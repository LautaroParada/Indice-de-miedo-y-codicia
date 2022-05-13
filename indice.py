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
	• Safe Heaven demand: Retornos de los últimos 20 días del bono BCP 10Y y 
    del Dólar (¿promediar ambos?) vs los retornos del IPSA, donde se mida en un 
    ratio. Idealmente que sea el indice de stress local vs el IPSA, pero tiene
    mucho desface en su publicación (15 días).
    
		○ Tipo de cambio del dólar observado diario -> F073.TCO.PRE.Z.D
		○ Índice de Precios Selectivo de Acciones, IPSA- Valor nominal - Base: 
		30 de diciembre de 2002 =1000 -> F013.IBC.IND.N.7.LAC.CL.CLP.BLO.D
		○ Tasa de interés mercado secundario de los bonos licitados por 
		el BCCh (BCP) a 10 años -> F022.BCLP.TIS.AN10.NO.Z.D
		
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