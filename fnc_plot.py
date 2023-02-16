#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 09:46:49 2021
@author: S.K.J. Falkena (s.k.j.falkena@pgr.reading.ac.uk)

Functions to plot the regimes, their frequencies and more.
"""

#%% IMPORT FUNCTIONS
"Import the necessary functions, both from python as well as my own."

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


#%% PLOT REGIMES
"""
Plot the regimes (theta) for one or two sets of regimes with possibly
different domains. Also give the plot title, whether the domains need to be
indicated and the name for saving the plot in the folder 'Regimes'.
"""

def plot_regimes( theta0, title=None, theta1=None, boxlines=None, savename=None ):
    
    # Get number of clusters
    k_nr, lat_nr, lon_nr = theta0.shape

    if theta1 is not None and theta0.shape != theta1.shape:
        lons, lats = np.arange(-90,42.5,2.5), np.arange(87.5,17.5,-2.5)
        clon, clat = -25, 60
        theta0 = app0_reg(np.abs(theta0.shape[1]-25), theta0)
        theta1 = app0_reg(np.abs(theta1.shape[1]-25), theta1)
    elif theta0.shape[1] == 25:        # 20-80N, 90W-30E
        lons, lats = np.arange(-90,32.5,2.5), np.arange(80,17.5,-2.5)
        clon, clat = -30, 55
    elif theta0.shape[1] == 24:      # 30-90N, 80W-40E
        lons, lats = np.arange(-80,42.5,2.5), np.arange(87.5,27.5,-2.5)
        clon, clat = -20, 60
        
    # Earth's gravitational acceleration to get geopotential height
    g0 = 9.80665    # Divide by g0 to go from geopotential to geopotential height
    weights = np.sqrt( np.cos(lats * np.pi/180) )
    th0 = theta0 /weights[np.newaxis, :, np.newaxis] /g0
    if theta1 is not None:
        th1 = theta1 /weights[np.newaxis, :, np.newaxis] /g0
    
    # Get the settings for the colorbar
    cbar_max = 250;      cbar_min = -cbar_max;       cbar_step = 50
    cbar_nr = 2*cbar_max /cbar_step
    cbar_tick = range( cbar_min, cbar_max+cbar_step, cbar_step )
    cbar_label = "Geopotential Height Anomaly (gpm)"
    
    name_list = ["NAO+", "NAO-", "AR+", "SB+", "AR-", "SB-"]
    
    fig = plt.figure(figsize=(6.5,6))
    # Set the title if given
    if title:
        plt.suptitle(title, fontsize=16)
    
    for k in range(k_nr):
        ax = fig.add_subplot(3, 2, k+1, projection=ccrs.Orthographic( central_longitude=clon, central_latitude=clat))
        ax.coastlines(color='tab:gray', linewidth=0.5)
        ax.gridlines(color='tab:gray', linestyle=':')
        
        ax.set_title(name_list[k])
        sub = ax.contourf(lons, lats, th0[k], cbar_tick, transform=ccrs.PlateCarree(), \
                    cmap = plt.cm.get_cmap('bwr',cbar_nr), vmin=cbar_min, vmax=cbar_max)
        if theta1 is not None:
            if theta0[k,0,0] == theta1[k,-1,-1]:
                ax.contour(lons[5::], lats[0:-4], th1[k,0:-4,5::], cbar_tick, transform=ccrs.PlateCarree(), \
                           colors = 'k', linewidths=1, vmin=cbar_min, vmax=cbar_max)
            else:
                ax.contour(lons, lats, th1[k], cbar_tick, transform=ccrs.PlateCarree(), \
                           colors = 'k', linewidths=1, vmin=cbar_min, vmax=cbar_max)
        else:
            ax.contour( lons, lats, th0[k], cbar_tick, transform=ccrs.PlateCarree(), \
                        colors = 'k', linewidths=1, vmin=cbar_min, vmax=cbar_max )
    
        if boxlines:
            # Linestyle
            linc = 'k';     linw = 1.5;     lins = '--'
            # Domain A
            plt.plot(np.arange(-90, 31,1), 20*np.ones(121), c=linc, lw=linw, ls=lins,\
                     transform=ccrs.PlateCarree())
            plt.plot(30*np.ones(61), np.arange(20,81,1), c=linc, lw=linw, ls=lins, \
                     transform=ccrs.PlateCarree())
            plt.plot(np.arange(-90, 31,1), 80*np.ones(121), c=linc, lw=linw, ls=lins, \
                     transform=ccrs.PlateCarree())
            plt.plot(-90*np.ones(61), np.arange(20,81,1), c=linc, lw=linw, ls=lins, \
                     transform=ccrs.PlateCarree())
            # Domain B
            plt.plot(np.arange(-80, 41,1), 30*np.ones(121), c=linc, lw=linw, ls=lins, \
                     transform=ccrs.PlateCarree())
            plt.plot(40*np.ones(61), np.arange(30,91,1), c=linc, lw=linw, ls=lins, \
                     transform=ccrs.PlateCarree())
            plt.plot(np.arange(-80, 41,1), 90*np.ones(121), c=linc, lw=linw, ls=lins, \
                     transform=ccrs.PlateCarree())
            plt.plot(-80*np.ones(61), np.arange(30,91,1), c=linc, lw=linw, ls=lins, \
                     transform=ccrs.PlateCarree())
            
    # Get the colorbar
    fig.subplots_adjust( hspace=.25, right=0.8 )
    cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
    # fig.colorbar( sub, cax=cbar_ax, ticks=cbar_tick, label=cbar_label )
    cbar = fig.colorbar( sub, cax=cbar_ax, ticks=cbar_tick )
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_label, fontsize=14) 
    
    # Save if a name is given
    if savename:
        plt.savefig("Plots/Regimes/"+savename+".eps", format='eps')
        
    return fig

# Add dimensions to plot two different domains
def app0_reg(dom, theta):
    # Get number of regimes
    k_nr = theta.shape[0]
    # Initialize regimes
    th = -10**10 * np.ones((k_nr, 28,53))
    if dom == 0:
        th[:,3::,0:-4] = theta
    elif dom == 1:
        th[:,0:-4,4::] = theta
    return th