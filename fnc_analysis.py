#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:49:31 2021
@author: S.K.J. Falkena (s.k.j.falkena@pgr.reading.ac.uk)

Functions for analysing the regimes, occurrence rates and more.
"""

#%% IMPORT FUNCTIONS
"Import the necessary functions, both from python as well as my own."

import numpy as np
import random as rnd

from fnc_bayescluster import yearbreak_ens

#%% REARRANGE CLUSTER CENTRES
"""
Rearrange the cluster centres such that they are in line with previous order.
"""

def rearrange_training(theta, seq_old, year):

    if year == 10:
        # Re-order the cluster centres
        temp = np.copy( theta[1] );  theta[1] = theta[2];  theta[2] = temp
        temp = np.copy( theta[2] );  theta[2] = theta[4];  theta[4] = temp
        temp = np.copy( theta[3] );  theta[3] = theta[4];  theta[4] = temp
        temp = np.copy( theta[4] );  theta[4] = theta[5];  theta[5] = temp
        
        # Change the corresponding numbers in the switching sequence
        seq = np.copy( seq_old )
        seq[ seq_old == 1 ] = 3;   seq[ seq_old == 3 ] = 5
        seq[ seq_old == 5 ] = 4;   seq[ seq_old == 4 ] = 2
        seq[ seq_old == 2 ] = 1
    
    elif year == 36:
        # Re-order the cluster centres
        temp = np.copy( theta[0] );  theta[0] = theta[3];  theta[3] = temp
        temp = np.copy( theta[1] );  theta[1] = theta[2];  theta[2] = temp
        temp = np.copy( theta[2] );  theta[2] = theta[5];  theta[5] = temp
        temp = np.copy( theta[4] );  theta[4] = theta[5];  theta[5] = temp
        
        # Change the corresponding numbers in the switching sequence
        seq = np.copy( seq_old )
        seq[ seq_old == 0 ] = 3;   seq[ seq_old == 1 ] = 4
        seq[ seq_old == 2 ] = 1;   seq[ seq_old == 3 ] = 0
        seq[ seq_old == 4 ] = 5;   seq[ seq_old == 5 ] = 2
    
    elif year == 0: # Constrained regimes
        # Re-order the cluster centres
        temp = np.copy( theta[0] );  theta[0] = theta[4];  theta[4] = temp
        temp = np.copy( theta[2] );  theta[2] = theta[5];  theta[5] = temp
        temp = np.copy( theta[3] );  theta[3] = theta[5];  theta[5] = temp
        
        # Change the corresponding numbers in the switching sequence
        seq = np.copy( seq_old )
        seq[ seq_old == 0 ] = 4;   seq[ seq_old == 4 ] = 0
        seq[ seq_old == 2 ] = 3;   seq[ seq_old == 3 ] = 5
        seq[ seq_old == 5 ] = 2;
    
    else:
        print("Give the length of the training dataset.")

    return theta, seq


#%% OCCURRENCE RATE
"""
Compute the occurrence rate of each of the k regimes over all ensemble members
and the whole timeseries.
"""

def occurrence( seq, k_nr ):
    occ = np.zeros(k_nr)
    for k in range(k_nr):
        occ[k] = np.sum(seq==k, axis=(0,1)) / (seq.shape[0]*seq.shape[1])    
    return occ

def occurrence_prob( gam, data ):
    if data == 0: # ERA-Interim
        occ = np.sum(gam, axis=0) / gam.shape[0]
    elif data == 1: # SEAS5
        occ = np.sum(gam, axis=(0,1)) / (gam.shape[0]*gam.shape[1])    
    return occ

def occurrence_erai( seq, k_nr ):
    occ = np.zeros(k_nr)
    for k in range(k_nr):
        occ[k] = np.sum(seq==k, axis=0) / seq.shape[0]   
    return occ

#%% TRANSITION PROBABILITIES
"""
Compute the transition probabilities between each combination of the k regimes
over all ensemble members and the full timeseries.
"""

def transition( seq, yb, k_nr ):
    # Get the number of regimes and number of years
    y_nr = len(yb[1::])

    # For each year
    tynr = np.zeros(( y_nr, k_nr, k_nr ))
    for y in range(y_nr):
        # Set arrays for the regime on day one, and the day after
        seqy0 = seq[yb[y]:yb[y+1]-1];   seqy1 = seq[yb[y]+1:yb[y+1]]
        
        # Loop over k to get all transitions for that year
        for i in range(0,k_nr):
            for j in range(0,k_nr):
                tynr[y,i,j] = np.sum( seqy1[seqy0==i]==j )
    
    # Remove the final elements of each year
    seq0 = np.delete(seq, yb[1::]-1, axis=0)
    # seq1 = np.delete(seq, yb[0:-1], axis=0)
    
    # Compute the average over the transition probabilities over all years
    tp = np.zeros(( k_nr, k_nr )); 
    tnr = np.sum( tynr, axis=0)
    for i in range(0,k_nr):
        tp[i,:] = tnr[i,:] /np.sum( seq0==i )
    # for j in range(0,k_nr):
    #     tp[:,j] = tnr[:,j] /np.sum( seq1==j )
    
    # tnr = tnr.T
    # tp = tp.T

    return tnr, tp

def transition_prob_old( gam, yb ):
    # Get the number of regimes and number of years
    time, ensnr, k_nr = gam.shape
    y_nr = len(yb[1::])
    
    # For each year
    tpy = np.zeros((y_nr,ensnr,k_nr,k_nr))
    for y in range(y_nr):
        # Set arrays for the regime probabilities on day one, and the day after
        gam0 = gam[yb[y]:yb[y+1]-1];  gam1 = gam[yb[y]+1:yb[y+1]]
        
        # Loop over k to get all transitions for that year
        for i in range(k_nr):
            for j in range(k_nr):
                tpy[y,:,i,j] = np.sum(np.multiply(gam0[:,:,i],gam1[:,:,j]), 
                                      axis=0 )
    
    # Remove the last year to use for normalization
    gam01 = np.delete(gam, yb[1::]-1, axis=0)
    
    # Compute the transition probabilities
    tp = np.zeros((k_nr,k_nr))
    for k in range(k_nr):
        tp[k] = np.sum( tpy, axis=(0,1))[k,:] / np.sum(gam01[:,:,k])
    
    # Transpose
    tp = tp.T
    
    return tp

def transition_prob( gam, yb ):
    # Get the number of regimes and number of years
    time, ensnr, k_nr = gam.shape
    y_nr = len(yb[1::])
    
    # For each year
    tpy = np.zeros((y_nr,ensnr,k_nr,k_nr))
    for y in range(y_nr):
        # Set arrays for the regime probabilities on day one, and the day after
        gam0 = gam[yb[y]:yb[y+1]-1];  gam1 = gam[yb[y]+1:yb[y+1]]
        
        # Loop over k to get all transitions for that year
        for i in range(k_nr):
            for j in range(k_nr):
                tpy[y,:,i,j] = np.sum(np.multiply(gam0[:,:,i],gam1[:,:,j]), 
                                      axis=0 )
    
    # Remove the last year to use for normalization
    gam01 = np.delete(gam, yb[1::]-1, axis=0)
    
    # Compute the transition probabilities
    tp = np.zeros((k_nr,k_nr))
    for k in range(k_nr):
        tp[k] = np.sum( tpy, axis=(0,1))[k,:] / np.sum(gam01[:,:,k])
    
    # Transpose
    tp = tp.T
    
    return tp

#%% AUTOCORRELATION
"""
Compute the 1-day and lagged autocorrelation of the regime probability time
series.
"""

def autocor(gam, yb, data):
    if data == 0: # ERA-Interim
         # Get the number of regimes and number of years
        time, k_nr = gam.shape
        
        # Get the gamma arrays removing either the first or last element of each year
        gam0 = np.delete(gam, yb[1::]-1, axis=0)
        gam1 = np.delete(gam, yb[0:-1], axis=0)
        # For each regime
        ac = np.zeros(k_nr)
        for k in range(k_nr):
            ac[k] = np.corrcoef(gam0[:,k], gam1[:,k])[0,1]
    
    if data == 1: # SEAS5
         # Get the number of regimes and number of years
        time, ensnr, k_nr = gam.shape
        y_nr = len(yb[1::])
        
        # Get the gamma arrays removign either the first or last element of each year
        gam0 = np.reshape(np.delete(gam, yb[1::]-1, axis=0), ((time-y_nr)*ensnr, k_nr))
        gam1 = np.reshape(np.delete(gam, yb[0:-1], axis=0), ((time-y_nr)*ensnr, k_nr))
        # For each regime
        ac = np.zeros(k_nr)
        for k in range(k_nr):
            ac[k] = np.corrcoef(gam0[:,k], gam1[:,k])[0,1]
    
    return ac

def autocor_lag(gam, yb, lagmax, data):
    if data == 0: # ERA-Interim
         # Get the number of regimes and number of years
        time, k_nr = gam.shape
        y_nr = len(yb)-1
        
        ac = np.zeros((k_nr, lagmax))
        for lag in np.arange(lagmax):
            # Get the gamma arrays removing the first/last element(s) of each year
            if lag == 0:
                gam0 = np.delete(gam, yb[1::]-1, axis=0)
                gam1 = np.delete(gam, yb[0:-1], axis=0)
            else:
                gam0 = gam[0:yb[1]-lag-1]
                gam1 = gam[lag+1:yb[1]]
                for y in range(1,y_nr):
                    gam0 = np.append(gam0, gam[yb[y]:yb[y+1]-lag-1], axis=0)
                    gam1 = np.append(gam1, gam[yb[y]+lag+1:yb[y+1]], axis=0)
            # For each regime
            for k in range(k_nr):
                ac[k,lag] = np.corrcoef(gam0[:,k], gam1[:,k])[0,1]
    
    if data == 1: # SEAS5
         # Get the number of regimes and number of years
        time, ensnr, k_nr = gam.shape
        y_nr = len(yb[1::])
            
        ac = np.zeros((k_nr, lagmax))
        for lag in np.arange(lagmax):
            # Get the gamma arrays removing the first/last element(s) of each year
            if lag == 0:
                gam0 = np.delete(gam, yb[1::]-1, axis=0)
                gam1 = np.delete(gam, yb[0:-1], axis=0)
            else:
                gam0 = gam[0:yb[1]-lag-1]
                gam1 = gam[lag+1:yb[1]]
                for y in range(1,y_nr):
                    gam0 = np.append(gam0, gam[yb[y]:yb[y+1]-lag-1], axis=0)
                    gam1 = np.append(gam1, gam[yb[y]+lag+1:yb[y+1]], axis=0)
            # Reshape
            gam0 = np.reshape(gam0, ((time-(y_nr*(lag+1)))*ensnr, k_nr))
            gam1 = np.reshape(gam1, ((time-(y_nr*(lag+1)))*ensnr, k_nr))
            # For each regime
            for k in range(k_nr):
                ac[k,lag] = np.corrcoef(gam0[:,k], gam1[:,k])[0,1]
    
    return ac

#%% BOOTSTRAPPING INTERANNUAL OCCURRENCE RATE
"""
Bootstrapping of the full SEAS5 switching sequence to get an error on the
seasonal occurrence rate values. the first argument (gam) is the data input 
being the regime probabilities from which we obtain the number of clusters and 
number of ensemble members. The second set of arguments give the number of 
members (m_nr) and the number of years (y_nr) to be considered for 
bootstrapping, as well as the total number of tests (btot) to be done. 
"""

def bootstrap_occ_year(gam, m_nr, btot):

    # Get dimensions from gam
    (tmax, enr, k_nr) = gam.shape    

    # Get an array indicating the start/end of the years
    yb, feb = yearbreak_ens( 0 );   ytot = len(yb) - 1
    
    # Initialize bootstrapping array for the seasonal occurrence
    occY_b          = np.zeros((btot,ytot,k_nr))
    occY_b_noise    = np.zeros((btot,ytot,k_nr))
    
    # Loop over the number of bootstrap years
    for b in range(btot):
        rnd.seed(b)
        
        # Create a random list of members to consider for each year
        y_b = np.zeros((ytot, m_nr))
        for y in range(ytot):
            y_b[y] = rnd.sample(range(0,enr), m_nr)
        
        # If using only one member a year
        if m_nr==1:
            # Loop over all years to select the sequences of bnr members each
            gam_b = np.zeros((yb[-1],k_nr))
            for y in range(ytot):
                gam_b[yb[y]:yb[y+1]] = gam[yb[y]:yb[y+1],int(y_b[y])]
        
            # Compute the yearly occurrence
            for y in range(ytot):
                occY_b[b,y] = occurrence_prob( gam_b[yb[y]:yb[y+1]], 0)
        
        # If using multiple members a year
        else:
            # Loop over all years to select the sequences of bnr members each
            gam_b = np.zeros((yb[-1], m_nr, k_nr))
            for y in range(ytot):
                for i in range(m_nr):
                    gam_b[yb[y]:yb[y+1],i] = gam[yb[y]:yb[y+1],int(y_b[y,i])]
        
            # Compute the yearly occurrence
            for y in range(ytot):
                occY_b[b,y] = occurrence_prob( gam_b[yb[y]:yb[y+1]], 1)
                
        # Randomly select rm_day days to get a noise level assuming stationarity
        for y in range(ytot):
            gam_b_noise = []
            rnd_year = rnd.sample(range(0,ytot), m_nr)
            for i in range(m_nr):
                gam_b_noise = np.append(gam_b_noise, gam[yb[rnd_year[i]]:yb[rnd_year[i]+1],rnd.randint(0,enr-1)])
            occY_b_noise[b,y] = occurrence_prob( gam_b_noise, 0)
    
    return occY_b, occY_b_noise
