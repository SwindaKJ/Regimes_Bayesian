#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:05:04 2021
@author: S.K.J. Falkena (s.k.j.falkena@pgr.reading.ac.uk)

Functions to use for different steps of the sequential Bayesian clustering.
"""

#%% IMPORT FUNCTIONS
"Import the necessary functions, both from python as well as my own."

import numpy as np
import random as rnd
import time as tm
from netCDF4 import Dataset

#%% LOAD DATA
"""
Load the data of the SEAS5 hindcast ensemble for the input domain, months and
background climatology:
    racc:       0   if running on mac
                1   if running on racc
    domain:     0   20-80N, 90W-30E
                1   30-87.5N, 80W-40E
    months:     0   DJFM
                1   DJF
                2   NDJFMA
    clim:       0   Constant climatology
                1   Seasonally varying climatology
"""

def loaddata(racc = 0, domain = 0, months = 0, clim = 0):
    # Load the raw data file
    if racc == 0:
        # data = Dataset('/Users/skf1018/OneDrive - University of Reading/MPE_CDT'
        #                '/PhD/Code/Code_BayesianClustering/Data'
        #                '/seas5_z500_NDJFMA_domall.nc', mode = 'r')
        data = Dataset('/Users/3753808/Library/CloudStorage/'
                       'OneDrive-UniversiteitUtrecht/PhD_Reading/Code/'
                       'PhD_Code/Code_BayesianClustering/Data/'
                       'seas5_z500_NDJFMA_domall.nc', mode = 'r')
    elif racc == 1:
        data = Dataset('Data/ens_geo500_NDJFMA_domall.nc', mode = 'r')

    # Get z500 data for the selected domain
    if domain == 0:    # 20-80N, 90W-30E
        # Extract latitude
        lats = data.variables['latitude'][4::]
        # Select z500 for the desired months
        z500d = filter_ens(data.variables['z'][:,:,4::,0:-4], months)
    elif domain == 1:    # 30-87.5N, 80W-40E
        # Extract latitude
        lats = data.variables['latitude'][1:-4]
        # Select z500 for the desired months
        z500d = filter_ens(data.variables['z'][:,:,1:-4,4::], months)
    # Close the dataset
    data.close()
    
    # Create weights, i.e. square-root of the cos of latitude
    weights = np.sqrt( np.cos(lats * np.pi/180) )
    # Create weighted data
    z500w = z500d * weights[:, np.newaxis]
    del z500d                   # Clear memory
    
    # Set the yearbreak depending on the months considered
    yb, lind = yearbreak_ens( months );      ynr = len(yb)-1;   lnr = int(ynr/4)
    ly0 = [[0,0,1,0] for i in range(lnr)];   ly = [i for l in ly0 for i in l]
    del ynr, lnr, ly0           # Clear memory
    
    # Compute anomalies wrt the selected climatology
    if clim == 0:   # Constant
        # Remove the average background state for all years
        dev = z500w - np.average( z500w, axis=(0,1) )
    elif clim == 1: # Seasonally varying
        dev, climbg = climatology_ens(z500w, yb, lind, ly)
        del climbg
    del lind, ly, z500w     # Clear memory
    
    return dev, weights, yb


# Select the desired months from the ensemble data
def filter_ens( geo, filter_index = 0 ):
    # Back-up to avoid crash!?##£!&*&^%$
    if filter_index not in [0,1,2,3]:
        filter_index = 0
    
    # Create an array with equidistant points for time and the year breaks
    nov=30;   dec=31;   jan=31;   feb=28;   mar=31;   apr=30;   may=10;
    feb_leap = feb+1;
    days = nov+dec+jan+feb+mar+apr+may
    winter_diff = np.arange(0, 36*days, days)

    # Append the desired months for each year
    winter_ind = []
    if filter_index == 0:       # DJFM
        for i in range(0, len(winter_diff)):
            if int((i+2)/4) == (i+2)/4:    # Leap years
                winter_ind = np.append( winter_ind, range(winter_diff[i]+nov, \
                                winter_diff[i]+nov+dec+jan+feb_leap+mar) )
            else:
                winter_ind = np.append( winter_ind, range(winter_diff[i]+nov, \
                                winter_diff[i]+nov+dec+jan+feb+mar) )
            dimlist = list(geo.shape);      dimlist[0] = len(winter_ind)
            dim = tuple(dimlist)
            # Select geopotential data
            geo_winter = np.zeros( dim )
            for i in range(0, len(winter_ind) ):
                geo_winter[i] = geo[int(winter_ind[i])]
    elif filter_index == 1:     # DJF
        for i in range(0, len(winter_diff)):
            if int((i+2)/4) == (i+2)/4:    # Leap years
                winter_ind = np.append( winter_ind, range(winter_diff[i]+nov, \
                                winter_diff[i]+nov+dec+jan+feb_leap) )
            else:
                winter_ind = np.append( winter_ind, range(winter_diff[i]+nov, \
                                winter_diff[i]+nov+dec+jan+feb) )
            dimlist = list(geo.shape);      dimlist[0] = len(winter_ind)
            dim = tuple(dimlist)
            # Select geopotential data
            geo_winter = np.zeros( dim )
            for i in range(0, len(winter_ind) ):
                geo_winter[i] = geo[int(winter_ind[i])]
    elif filter_index == 2:       # NDJFMA (start at Nov 16 to reduce bias, reassign later)
        for i in range(0, len(winter_diff)):
            if int((i+2)/4) == (i+2)/4:    # Leap years
                winter_ind = np.append( winter_ind, range(winter_diff[i]+15, \
                                winter_diff[i]+nov+dec+jan+feb_leap+mar+apr) )
            else:
                winter_ind = np.append( winter_ind, range(winter_diff[i]+15, \
                                winter_diff[i]+nov+dec+jan+feb+mar+apr) )
            dimlist = list(geo.shape);      dimlist[0] = len(winter_ind)
            dim = tuple(dimlist)
            # Select geopotential data
            geo_winter = np.zeros( dim )
            for i in range(0, len(winter_ind) ):
                geo_winter[i] = geo[int(winter_ind[i])]
    elif filter_index == 3:       # NDJFMA (all)
        for i in range(0, len(winter_diff)):
            if int((i+2)/4) == (i+2)/4:    # Leap years
                winter_ind = np.append( winter_ind, range(winter_diff[i], \
                                winter_diff[i]+nov+dec+jan+feb_leap+mar+apr) )
            else:
                winter_ind = np.append( winter_ind, range(winter_diff[i], \
                                winter_diff[i]+nov+dec+jan+feb+mar+apr) )
            dimlist = list(geo.shape);      dimlist[0] = len(winter_ind)
            dim = tuple(dimlist)
            # Select geopotential data
            geo_winter = np.zeros( dim )
            for i in range(0, len(winter_ind) ):
                geo_winter[i] = geo[int(winter_ind[i])]

    return geo_winter

# Make a list of the start/end indices of each year
def yearbreak_ens( filter_index ):
    # Set the time length and winter length
    nov = 30;   dec = 31;   jan = 31;   feb = 28;   mar = 31;   apr = 30
    if filter_index == 0:   # DJFM
        days = dec+jan+feb+mar;             leap_min = mar + 1
    elif filter_index == 1: # DJF
        days = dec+jan+feb;                 leap_min = 1
    elif filter_index == 2: # NDJFMA from Nov 16th
        days = 15+dec+jan+feb+mar+apr;     leap_min = mar+apr + 1
    elif filter_index == 3: # All NDJFMA
        days = nov+dec+jan+feb+mar+apr;     leap_min = mar+apr + 1
    leap = days + 1

    # Create an array with equidistant points for time and the year breaks
    winter_diff = [0,days];     leap_index = [0]
    for y in range(1983,2018):
        if int(y/4) == y/4 :
            leap_index = np.append( leap_index, winter_diff[-1] + leap - leap_min )
            winter_diff = np.append( winter_diff, winter_diff[-1] + leap )
        else:
            winter_diff = np.append( winter_diff, winter_diff[-1] + days )

    return winter_diff, leap_index[1::]

# Compute the seasonal climatology and anomalies with respect to it
def climatology_ens(geo_sc, yb, lind, ly):
    # Get the lat-lon numbers
    time, ens_nr, lat_nr, lon_nr = geo_sc.shape
    
    # Get the length of a leap year (# month dependent)
    if lind[0]<yb[1]:
        leap_ind = int(lind[0])
    else:
        leap_ind = int(lind[0] - yb[1]*(ly.index(1)))
    
    # Get the average state for every day of the winter
    clim_raw = np.zeros((yb[1]-ly[0],lat_nr,lon_nr))
    for i in range(yb[1]-ly[0]):
        il = yb[0:-1]+i
        if il[0] >= leap_ind:    # Account for leap years
            il = il + ly
        clim_raw[i] = np.average( geo_sc[il], axis=(0,1) )
    # Add Feb 29th
    clim_raw = np.insert(clim_raw, leap_ind, np.average( geo_sc[lind], axis=(0,1)), axis=0)
    
    # Fit a 4th order polynomial
    clim = np.zeros(clim_raw.shape);    time_clim = np.arange(0,len(clim_raw))
    for it in range(lat_nr):
        for il in range(lon_nr):
            pf = np.polyfit(time_clim, clim_raw[:,it,il], deg=4)
            pol = np.poly1d(pf)
            clim[:,it,il] = pol(time_clim)
            
    # Compute deviations wrt the computed climatology
    dev = np.zeros(geo_sc.shape)
    # February 29th
    dev[lind] = geo_sc[lind] - clim[leap_ind]
    for i in range(yb[1]-ly[0]):
        il = yb[0:-1]+i
        if il[0] >= leap_ind:    # Account for leap years
            il = il + ly
        dev[il] = geo_sc[il] - clim[i]
    
    return dev, clim


#%% K-MEANS CLUSTERING FOR TRAINING
"""
Implement k-means clustering. Input is the full dataset, the desired number
of clusters k_nr and the number of runs. In every step the distance between
the data and the different clusters is computed. Based on this distance each
datapoint is assigned to a cluster. The new clusters are computed as the
average of all data belonging to that cluster. This iteration is continued
until the result is not improving sufficiently anymore. The output are the 
different clusters, the switching sequence and the distance of the data to 
the clusters.
"""

def kmeans( data_ens, k_nr, test_nr ):
    # Get the data size
    (time_nr, ens_nr, lat_nr, lon_nr) = data_ens.shape

    # Set maximum number of steps
    step_max = 400

    # Create initial arrays
    dist0 = np.zeros(( k_nr, time_nr, ens_nr ))         # Distance vector
    seq0 = np.ones((time_nr, ens_nr))                   # Transition sequence
    # Set the arrays for the results to save
    L_hist = np.zeros(test_nr)
    seq_hist = np.zeros((test_nr, time_nr, ens_nr))
    # Initialize the optimal results
    L_sum_opt = 1000
    theta_opt = np.zeros((k_nr, lat_nr, lon_nr))
    seq_opt = np.zeros((time_nr, ens_nr))
    dist_opt = np.zeros((k_nr, time_nr, ens_nr))

    # Run the algorithm multiple times with different initial conditions
    for test in range(test_nr):
        print("Test Nr.:" +repr(test))
        # Set the initial distance, L and clusters
        dist = dist0;   seq = 0*seq0
        theta = np.random.randn( k_nr, lat_nr, lon_nr ) * np.amax( data_ens )
            # The initial random clusters, scaled by the data
        # Set the counter
        j = 0

        start_loop = tm.time()
        # As long as the desired tolerance, or maximum step number are not reached
        while( j < step_max and len(np.ones((time_nr, ens_nr))[seq0!=seq]) > 0 ):
            print("The nr of mismatching clusters is: "+repr(len(np.ones((time_nr, ens_nr))[seq0!=seq])))
            # Set the previous sequence
            seq0 = seq
            # Get distance between the data and the clusters
            for k in range(0,k_nr):
                dist[k] = distance( data_ens, theta[k] )
            # Create a sequence of which cluster is closest
            seq = np.argmin( dist, axis=0 )

            # Implement a back up if a cluster is not present
            for k in range(0, k_nr):
                if not any( s==k for s in seq.flatten()):
                    rand_int0 = np.random.randint(0,time_nr)
                    rand_int1 = np.random.randint(0,ens_nr)
                    seq[ rand_int0, rand_int1 ] = k
                    print( "Cluster " + repr(k) + " was not present in the data,"
                    "therefore a perturbation has been added at time and ensemble "
                    + repr([rand_int0, rand_int1]) )

            # Get the list of minimal distances to find L
            L_list = np.min( dist, axis=0 )
            L_sum = sum( L_list.flatten() ) /(time_nr*ens_nr)
            # Update the clusters as the average of all data assigned to that cluster
            for k in range(0,k_nr):
                theta[k] = np.average( data_ens[seq==k], axis=0 )
            # Update the counter
            j = j+1

        # Print information about the success of the optimalization
        if j == step_max:
            print( "The maximum stepnumber (" + repr(step_max) + ") was reached." )
        else:
            # Compute the final distances, sequence and L
            for k in range(0,k_nr):
                dist[k] = distance( data_ens, theta[k] )
            seq = np.argmin( dist, axis=0 )
            L_list = np.min( dist, axis=0 )
            L_sum = sum( L_list.flatten() ) /(time_nr*ens_nr)
            print( "The result is not improving sufficiently anymore, after "
            + repr(j) + " steps and the distance is " + repr( L_sum ) )
        print( "The time taken to compute the distance is " + repr( tm.time() - start_loop) )

        # Save the L and sequence for this test
        L_hist[test] = L_sum
        seq_hist[test] = seq
        # If the current L is smaller than the so far best one, replace it
        if L_sum < L_sum_opt:
            L_sum_opt = L_sum
            theta_opt = theta
            seq_opt = seq
            dist_opt = dist

    return L_hist, seq_hist, L_sum_opt, theta_opt, seq_opt, dist_opt

# Compute the distance between the data and the clusters
def distance( data_ens, cluster ):
    diff = data_ens - cluster                           # Compute difference
    dist = np.linalg.norm( diff, axis=(2,3) )           # Compute norm of difference
    dist = dist /(data_ens.shape[2]*data_ens.shape[3])  # Normalise
    return dist

    
#%% SEQUENTIAL BAYESIAN CLUSTERING
"""
The functions required for sequential Bayesian clustering. Specifically, the
computation of the probability of the data given each of the regimes following
a multivariate normal distribution with mean and covariance determined from
the training dataset.
"""

# Compute the distance between 51 Z500 fields and a cluster centre
def distance_obs(data, theta):
    diff  = data - theta                            # Compute difference
    dist  = np.linalg.norm( diff, axis=(1,2) )      # Compute norm of difference
    distn = dist /(data.shape[1]*data.shape[2])     # Normalise
    return distn

# A function for a multivariate normal distribution
def normal_distr(mu, cov_inv, cov_det, k_nr, x):
    nd = np.exp(-1/2 * np.dot( (x-mu).T, np.dot(cov_inv, x-mu)) ) \
        / np.sqrt((2*np.pi)**k_nr * cov_det)
    return nd
    
def prob_obs_mvn( data, theta, mean, covinv, covdet, dist_return):
    
    # Get the number of ensemble members and clusters
    ensnr = data.shape[0]
    k_nr = theta.shape[0]
    
    # Compute the distance between the data and the cluster centres
    dist_obs = np.zeros((ensnr, k_nr))
    for k in range(k_nr):
        dist_obs[:,k] = distance_obs(data, theta[k])
    
    # Compute the probability of being in each of the regimes following MVN
    prob_reg_obs = np.zeros((ensnr, k_nr))
    # For each ensemble member
    for i in range(ensnr):
        prob_dist_cond_mv = np.zeros(k_nr)
        # And conditional on each regime
        for k in range(k_nr):
            # Compute the PDF-value corresponding to the observations
            prob_dist_cond_mv[k] = normal_distr(mean[k], covinv[k], covdet[k], 
                                                k_nr, dist_obs[i])
    
        # Convert this to a probability of being in each of the regimes (normalisation)
        prob_reg_obs[i] = prob_dist_cond_mv / np.sum(prob_dist_cond_mv)
    
    if dist_return == 0:
        return prob_reg_obs
    elif dist_return == 1:
        return prob_reg_obs, dist_obs

def distance_obs_erai(data, theta):
    diff  = data - theta                            # Compute difference
    dist  = np.linalg.norm( diff, axis=(1,2) )      # Compute norm of difference
    distn = dist /(data.shape[0]*data.shape[1])     # Normalise
    return distn

def prob_obs_erai_mvn( data, theta, mean, covinv, covdet):
    
    # Get the number of ensemble members and clusters
    k_nr = theta.shape[0]
    
    # Compute the distance between the data and the cluster centres
    dist_obs = distance_obs_erai(data, theta)
    
    # Compute the probability of being in each of the regimes following MVN
    prob_dist_cond_mv = np.zeros(k_nr)
    # And conditional on each regime
    for k in range(k_nr):
        # Compute the PDF-value corresponding to the observations
        prob_dist_cond_mv[k] = normal_distr(mean[k], covinv[k], covdet[k], 
                                            k_nr, dist_obs)

    # Convert this to a probability of being in each of the regimes (normalisation)
    prob_reg_obs = prob_dist_cond_mv / np.sum(prob_dist_cond_mv)
    
    return prob_reg_obs
    
#%% ERA-INTERIM

def loaddata_erai(domain = 0, months = 0, clim = 0):
    # Load the raw data file
    data = Dataset('Data/erai_z500_NDJFMA.nc', mode = 'r')

    # Get z500 data for the selected domain
    if domain == 0:    # 20-80N, 90W-30E
        # Extract latitude
        lats = data.variables['latitude'][4::]
        # Select z500 for the desired months
        z500d = filter_erai(data.variables['z'][:,4::,0:-4], months)
    elif domain == 1:    # 30-87.5N, 80W-40E
        # Extract latitude
        lats = data.variables['latitude'][1:-4]
        # Select z500 for the desired months
        z500d = filter_erai(data.variables['z'][:,1:-4,4::], months)
    # Close the dataset
    data.close()
    
    # Create weights, i.e. square-root of the cos of latitude
    weights = np.sqrt( np.cos(lats * np.pi/180) )
    # Create weighted data
    z500w = z500d * weights[:, np.newaxis]
    del z500d                   # Clear memory
    
    # Set the yearbreak depending on the months considered
    yb, lind = yearbreak_erai( months );      ynr = len(yb)-1;   lnr = int(ynr/4)
    ly0 = [[0,0,1,0] for i in range(lnr)];   ly = [i for l in ly0 for i in l]
    del ynr, lnr, ly0           # Clear memory
    
    # Compute anomalies wrt the selected climatology
    if clim == 0:   # Constant
        # Remove the average background state for all years
        dev = z500w - np.average( z500w, axis=0 )
    elif clim == 1: # Seasonally varying
        dev, climbg = climatology_erai(z500w, yb, lind, ly)
        del climbg
    del lind, ly, z500w     # Clear memory
    
    return dev, weights, yb

def filter_erai( geo, filter_index ):

    # Create an array with equidistant points for time and the year breaks
    nov = 30;   dec = 31;   jan = 31;   feb = 28;   mar = 31;   apr = 30
    feb_leap = feb+1
    days = nov+dec+jan+feb+mar+apr
    leap = days+1
    winter_diff = [0,leap]
    for y in range(1981,2020):
        if int(y/4) == y/4 :
            winter_diff = np.append( winter_diff, winter_diff[-1] + leap )
        else:
            winter_diff = np.append( winter_diff, winter_diff[-1] + days )

    if filter_index == 0: # DJFM
        winter_ind = []
        for i in range(0, len(winter_diff)-1):
            if winter_diff[i+1] - winter_diff[i] == 182:    # Leap years
                winter_ind = np.append( winter_ind, range(winter_diff[i]+nov, \
                                winter_diff[i]+nov+dec+jan+feb_leap+mar) )
            else:
                winter_ind = np.append( winter_ind, range(winter_diff[i]+nov, \
                                winter_diff[i]+nov+dec+jan+feb+mar) )
        
    elif filter_index == 1: #DJF
        winter_ind = []
        for i in range(0, len(winter_diff)-1):
            if winter_diff[i+1] - winter_diff[i] == 182:    # Leap years
                winter_ind = np.append( winter_ind, range(winter_diff[i]+nov, \
                                winter_diff[i]+nov+dec+jan+feb_leap) )
            else:
                winter_ind = np.append( winter_ind, range(winter_diff[i]+nov, \
                                winter_diff[i]+nov+dec+jan+feb) )
    elif filter_index == 2: # NDJFMA
        geo_winter = geo
    else:
        print("Give a correct filter entry.")
    
    # If filtering, select geopotential data
    if filter_index != 2:
        geo_one = np.zeros(geo.shape[1::])
        geo_winter = np.repeat(geo_one[np.newaxis], len(winter_ind), axis=0)
        for i in range(0, len(winter_ind) ):
            geo_winter[i] = geo[int(winter_ind[i])]

    return geo_winter

def yearbreak_erai( filter_index ):

    # Set the time length and winter length
    nov = 30;   dec = 31;   jan = 31;   feb = 28;   mar = 31;   apr = 30
    if filter_index == 0: # DJFM
        days = dec+jan+feb+mar;             leap_min = mar + 1
    elif filter_index == 1: # DJF
        days = dec+jan+feb;                 leap_min = 1
    elif filter_index == 2: # NDJFMA
        days = nov+dec+jan+feb+mar+apr;     leap_min = mar+apr + 1
    leap = days + 1

    # Create an array with equidistant points for time and the year breaks
    winter_diff = [0,leap];     leap_index = [leap - leap_min]
    for y in range(1981,2020):
        if int(y/4) == y/4 :
            leap_index = np.append( leap_index, winter_diff[-1] + leap - leap_min )
            winter_diff = np.append( winter_diff, winter_diff[-1] + leap )
        else:
            winter_diff = np.append( winter_diff, winter_diff[-1] + days )

    return winter_diff, leap_index

def climatology_erai(geo_sc, yb, lind, ly):
    
    # Get the lat-lon numbers
    time, lat_nr, lon_nr = geo_sc.shape
    
    # Get the length of a leap year (# month dependent)
    if lind[0]<yb[1]:
        leap_ind = int(lind[0])
    else:
        leap_ind = int(lind[0] - yb[1]*(ly.index(1)))
    
    # Get the average state for every day of the winter
    clim_raw = np.zeros((yb[1]-ly[0],lat_nr,lon_nr))
    for i in range(yb[1]-ly[0]):
        il = yb[0:-1]+i
        if il[0] >= leap_ind:    # Account for leap years
            il = il + ly
        clim_raw[i] = np.average( geo_sc[il], axis=0 )
    # Add Feb 29th
    clim_raw = np.insert(clim_raw, leap_ind, np.average( geo_sc[lind], axis=0), axis=0)
    
    # Fit a 4th order polynomial
    clim = np.zeros(clim_raw.shape);    time_clim = np.arange(0,len(clim_raw))
    for it in range(lat_nr):
        for il in range(lon_nr):
            pf = np.polyfit(time_clim, clim_raw[:,it,il], deg=4)
            pol = np.poly1d(pf)
            clim[:,it,il] = pol(time_clim)
            
    # Compute deviations wrt the computed climatology
    dev = np.zeros(geo_sc.shape)
    # February 29th
    dev[lind] = geo_sc[lind] - clim[leap_ind]
    for i in range(yb[1]-ly[0]):
        il = yb[0:-1]+i
        if il[0] >= leap_ind:    # Account for leap years
            il = il + ly
        dev[il] = geo_sc[il] - clim[i]
    
    return dev, clim

# Rearrange the ERA-Interim data
def rearrange_erai(theta, seqd):

    temp = np.copy( theta[0] );  theta[0] = theta[4];  theta[4] = temp
    temp = np.copy( theta[1] );  theta[1] = theta[2];  theta[2] = temp
    temp = np.copy( theta[4] );  theta[4] = theta[5];  theta[5] = temp

    seq = np.copy( seqd )
    seq[ seqd == 0 ] = 5;   seq[ seqd == 1 ] = 2;   seq[ seqd == 2 ] = 1
    seq[ seqd == 4 ] = 0;   seq[ seqd == 5 ] = 4

    return theta, seq

def rearrange_match(theta0, theta1, seq1):
    # Get the number of regimes
    k_nr = theta0.shape[0]

    # Initialize rearranged regimes/sequences
    thetam = np.zeros(theta1.shape);    seqm = np.copy( seq1 );

    # Create a matrix with the distances between all regimes
    mat = np.zeros((k_nr, k_nr));
    for k in range(k_nr):
        mat[k] = distance_obs( theta0, theta1[k] )

    # Go through the distance matrix for each ensemble, starting at the smallest
    # entry and working up. Both the sum off all distances, as well as the matching
    # regimes are computed
    diff = 0; # Set the clustervariation to zero
    while np.min( mat ) < 100:
        diff += np.min(mat);    diff_max = np.min(mat)  # Add + store the minimal matrix value
        ind_min = np.argwhere( mat == np.min(mat) )[0]
        mat[ind_min[0],:] = 100*np.ones(k_nr)   # Replace corresp. rows and
        mat[:,ind_min[1]] = 100*np.ones(k_nr)   # columns
        # Rearrange the regimes and sequences
        thetam[ind_min[1]] = theta1[ind_min[0]]
        seqm[seq1 == ind_min[0]] = ind_min[1]

    # Store the quality of the match
    dist = diff/k_nr     # Add the result to list and normalize by k
    dist_max = diff_max  # Store the final (largest considered) distance

    return thetam, seqm, mat, dist, dist_max
    