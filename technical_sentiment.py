# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:05:29 2022

@author: lauta
"""

import os

# Cargar mis claves para las APIs desde las variables de entorno.
api_key = os.environ['API_EOD']

from eod import EodHistoricalData
import pandas as pd
import numpy as np

# Creación de las instancias
client = EodHistoricalData(api_key)

from matplotlib import cm
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, Wedge, Rectangle
def degree_range(n): 
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points

def rot_text(ang): 
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation

def gauge(labels=['LOW','MEDIUM','HIGH','VERY HIGH','EXTREME'], \
          colors='jet_r', arrow=1, title='', fname=False): 
    
    """
    some sanity checks first
    
    """
    
    N = len(labels)
    
    if arrow > N: 
        raise Exception("\n\nThe category ({}) is greated than \
        the length\nof the labels ({})".format(arrow, N))
 
    
    """
    if colors is a string, we assume it's a matplotlib colormap
    and we discretize in N discrete colors 
    """
    
    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, N)
        cmap = cmap(np.arange(N))
        colors = cmap[::-1,:].tolist()
    if isinstance(colors, list): 
        if len(colors) == N:
            colors = colors[::-1]
        else: 
            raise Exception("\n\nnumber of colors {} not equal \
            to number of categories{}\n".format(len(colors), N))

    """
    begins the plotting
    """
    
    fig, ax = plt.subplots()

    ang_range, mid_points = degree_range(N)

    labels = labels[::-1]
    
    """
    plots the sectors and the arcs
    """
    patches = []
    for ang, c in zip(ang_range, colors): 
        # sectors
        patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
        # arcs
        patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))
    
    [ax.add_patch(p) for p in patches]

    
    """
    set the labels (e.g. 'LOW','MEDIUM',...)
    """

    for mid, lab in zip(mid_points, labels): 

        ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
            horizontalalignment='center', verticalalignment='center', fontsize=14, \
            fontweight='bold', rotation = rot_text(mid))

    """
    set the bottom banner and the title
    """
    r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
    ax.add_patch(r)
    
    ax.text(0, -0.05, title, horizontalalignment='center', \
         verticalalignment='center', fontsize=16, fontweight='bold')

    """
    plots the arrow now
    """
    
    pos = mid_points[abs(arrow - N)]
    
    ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                 width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
    
    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    """
    removes frame and ticks, and makes axis equal and tight
    """
    
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    plt.tight_layout()
    if fname:
        fig.savefig(fname, dpi=200)
    
    
    # Graph source
    ax.text(0.3, 0.03,  
             "Fuente: EOD Historical Data   Gráfico: Lautaro Parada", 
             horizontalalignment='center',
             verticalalignment='center', 
             transform=ax.transAxes, 
             fontsize=8, 
             color='black',
             bbox=dict(facecolor='tab:gray', alpha=0.5))
    

#%% CASO DEL IPSA

# Llamar a los datos
resp = pd.DataFrame(
    client.get_bulk_markets(
        exchange='SN', filter_='extended'
        )
    )

resp.set_index('code', inplace=True)

resp_ = client.get_fundamental_equity('SPIPSA.INDX')['Components']
# Extraer solo los nombres del ipsa
ipsa = []
for key in resp_.keys():
    ipsa.append( resp_[key]['Code'])
    
ipsa = resp.loc[ipsa]
ipsa['death_cross'] = ipsa['ema_50d'] < ipsa['ema_200d']
print(f"Empresas en el IPSA con un DEATH-CROSS: {ipsa.death_cross.sum()}")
ipsa['above_200'] = ipsa['close'] > ipsa['ema_200d']
print(f"Empresas en el IPSA sobre la media de 200 periodos: {ipsa.above_200.sum()}")
ipsa['above_50'] = ipsa['close'] > ipsa['ema_50d']
print(f"Empresas en el IPSA sobre la media de 50 periodos: {ipsa.above_50.sum()}")

#%% Death cross
ratio_ipsa = ipsa.death_cross.sum() / ipsa.shape[0]
print(f"Porcentaje de acciones del #IPSA con un DEATH-CROSS confirmado (200ma > 50ma): {round(ratio_ipsa*100, 2)}%")

if ratio_ipsa > 0 and ratio_ipsa < 0.25:
    arrow_ = 1
elif ratio_ipsa >= 0.25 and ratio_ipsa < 0.5:
    arrow_ = 2
elif ratio_ipsa >= 0.5 and ratio_ipsa < 0.75:
    arrow_ = 3
elif ratio_ipsa >= 0.75:
    arrow_ = 4
# graficar 
gauge(labels=['BAJO','MEDIO','ALTO','EXTREMO'], 
  colors=['#007A00','#0063BF','#FFCC00','#ED1C24'], 
  arrow=arrow_, title=f"Porcentaje del IPSA\ncon un DEATH-CROSS: {round(ratio_ipsa*100, 2)}%") 

#%% Sobre el 200 ma
ratio_ipsa = ipsa.above_200.sum() / ipsa.shape[0]
print(f"Porcentaje de acciones del #IPSA sobre la media de 200 periodos: {round(ratio_ipsa*100, 2)}%")

if ratio_ipsa > 0 and ratio_ipsa < 0.25:
    arrow_ = 1
elif ratio_ipsa >= 0.25 and ratio_ipsa < 0.5:
    arrow_ = 2
elif ratio_ipsa >= 0.5 and ratio_ipsa < 0.75:
    arrow_ = 3
elif ratio_ipsa >= 0.75:
    arrow_ = 4
# graficar 
gauge(labels=['BAJO','MEDIO','ALTO','EXTREMO'], 
  colors=['#007A00','#0063BF','#FFCC00','#ED1C24'], 
  arrow=arrow_, title=f"Porcentaje del IPSA sobre la\nmedia de 200 periodos: {round(ratio_ipsa*100, 2)}%") 
