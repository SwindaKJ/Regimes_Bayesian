#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:54:12 2021
@author: S.K.J. Falkena (s.k.j.falkena@pgr.reading.ac.uk)

Apply a sequential Bayesian clustering method to identify the circulation
regimes and their evolution in time.
"""

#%% IMPORT FUNCTIONS
"Import the necessary functions, both from python as well as my own."

import numpy as np
import random as rnd
import matplotlib.pyplot as plt

from fnc_bayescluster import loaddata, loaddata_erai, yearbreak_ens, \
                                yearbreak_erai, prob_obs_mvn, \
                                prob_obs_erai_mvn
from fnc_analysis import occurrence_prob, transition_prob, autocor, autocor_lag, \
                            bootstrap_occ_year
from fnc_plot import plot_regimes

#%% INPUT ARGUMENTS
"Set the number of clusters and the number of years used for training"

train_year = 36     # Number of years for the training dataset
k_nr = 6            # Number of clusters

# Set print options
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

#%% LOAD SEAS5 DATA
"Load the SEAS5 z500 anomalies for the desired domain, months and climatology."

# Load the z500 anomalies, weights and indices for the start/end of each year
dev, weights, yb = loaddata()
# Get the dimensions of the data
(timenr, ensnr, latnr, lonnr) = dev.shape

# Separate in a training part and a part for sequential Bayesian clustering
dev_train = dev[0:yb[train_year]]   # Training dataset
# Get the list of the start/end of each year in the data
yb, feb = yearbreak_ens( 0 )
if train_year < 36:
    dev_bayes = dev[yb[train_year]::]   # Bayesian clustering dataset
    # Select the bayes years
    yb_bayes = yb[train_year::] - yb[train_year]
    feb_bayes = feb[feb>yb[train_year]]- yb[train_year]
else:
    dev_bayes = dev_train
    # Select all years
    yb_bayes = yb
    feb_bayes = feb
# get the number of years
ynr_bayes = len(yb_bayes) - 1
del dev, weights, dev_train

#%% LOAD ERA-INTERIM DATA
"Load the ERA-Interim z500 anomalies for the desired domain, months and climatology."

# Load the z500 anomalies, weights and indices for the start/end of each year
dev_erai, weights, yb = loaddata_erai( )
del weights, yb
# Get yearbreaks
yb_erai, feb_erai = yearbreak_erai( 0 )


#%% IMPORT ERA-INTERIM REGIMES
"Import the ERA-Interim regimes without constraint. We obtain the cluster \
centres, average frequencies and transition probabilities, as well as the \
distance distributions conditional on each of the regimes."

# Load the data
training = np.load("Training/ERAInterim_k"+repr(k_nr)+".npz")
# Retrieve the training outcome
th_erai = training["theta"]
q_erai = training["occurrence"]
T_erai = training["transition"]
dmean_erai = training["mean_distance"]
dcov_erai = training["covariance_distance"]
dcovinv_erai = training["covariance_inverse_distance"]
dcovdet_erai = training["covariance_determinant_distance"]
del training

# Plot the regimes
# fig = plot_regimes(th_erai, title="ERA-Interim")

#%% IMPORT TIME CONSTRAINED ERA-INTERIM REGIMES
"Import the ERA-Interim regimes with time constraint. We obtain the cluster \
centres, average frequencies and transition probabilities, as well as the \
distance distributions conditional on each of the regimes."

# Load the data
training = np.load("Training/ERAInterim_timeconstraint_k"+repr(k_nr)+".npz")
# Retrieve the training outcome
th_eraiC = training["theta"]
sq_eraiC = training["sequence"]
q_eraiC = training["occurrence"]
T_eraiC = training["transition"]
dmean_eraiC = training["mean_distance"]
dcov_eraiC = training["covariance_distance"]
dcovinv_eraiC = training["covariance_inverse_distance"]
dcovdet_eraiC = training["covariance_determinant_distance"]
del training

# Plot the regimes
# fig = plot_regimes(th_eraiC, title="Constrained ERA-Interim")

# Corresponding gamma for hard assignment
gam_eraiC = np.zeros((4729, k_nr))
for t in range(4729):
    gam_eraiC[t,int(sq_eraiC[t])] = 1

#%% IMPORT SEAS5 REGIMES
"Load the standard SEAS5 regimes. We obtain the cluster centres, average \
frequencies and transition probabilities, as well as the distance \
distributions conditional on each of the regimes."

# Load the data
training = np.load("Training/Seas5prior_"+repr(train_year)+"years_k"
                   +repr(k_nr)+".npz")
# Retrieve the training outcome
th_train = training["theta"]
q_train = training["occurrence"]
T_train = training["transition"]
dmean_train = training["mean_distance"]
dcov_train = training["covariance_distance"]
dcovinv_train = training["covariance_inverse_distance"]
dcovdet_train = training["covariance_determinant_distance"]
del training

# Plot the regimes
# fig = plot_regimes(th_train, title="SEAS5 Training")

#%% IMPORT CONSTRAINED SEAS5 REGIMES
"Load the constrained SEAS5 regimes. We obtain the cluster centres, average \
frequencies and transition probabilities, as well as the distance \
distributions conditional on each of the regimes."

# Load the data
training = np.load("Training/Seas5prior_constrained_"+repr(train_year)+
                   "years_k"+repr(k_nr)+".npz")
# Retrieve the training outcome
th_trainC = training["theta"]
q_trainC_7 = training["occurrence"]
T_trainC_7 = training["transition"]
dmean_trainC = training["mean_distance"]
dcov_trainC = training["covariance_distance"]
dcovinv_trainC = training["covariance_inverse_distance"]
dcovdet_trainC = training["covariance_determinant_distance"]
del training

# Correct q and T to six regimes
"CHECK WITH OTHERS"
q_trainC = q_trainC_7[0:k_nr] / np.sum(q_trainC_7[0:k_nr])
T_trainC = T_trainC_7[0:k_nr,0:k_nr] / np.sum(T_trainC_7[0:k_nr,0:k_nr], axis=1)

# Plot the regimes
# fig = plot_regimes(th_trainC, title="SEAS5 Constrained")

#%% LOAD SEAS5 ENSO DISTANCE DISTRIBUTIONS
"Load the distance distributions corresponding to very strong El Nino, strong \
El Nino and strong La Nina events (in that order)."

# Make a list of the year data (ERA-Interim)
years = np.arange(1979, 2019)
# Lists of El Nino/La Nina years
ens_y = np.array([1982, 1997, 2015])
en_y = np.array([1982, 1987, 1991, 1997, 2015])
ln_y = np.array([1988, 1998, 1999, 2007, 2010])
# Get the indices of these years
ind_ens = np.intersect1d(years, ens_y, return_indices=True)[1]
ind_en = np.intersect1d(years, en_y, return_indices=True)[1]
ind_ln = np.intersect1d(years, ln_y, return_indices=True)[1]

# Get list of time indices during those years
yb_ind_ens = np.arange(yb_erai[ind_ens[0]],yb_erai[ind_ens[0]+1])
yb_ind_en = np.arange(yb_erai[ind_en[0]],yb_erai[ind_en[0]+1])
yb_ind_ln = np.arange(yb_erai[ind_ln[0]],yb_erai[ind_ln[0]+1])
for i in range(1,5):
    if i < 3:
        yb_ind_ens = np.append(yb_ind_ens, np.arange(yb_erai[ind_ens[i]],yb_erai[ind_ens[i]+1]))
    yb_ind_en = np.append(yb_ind_en, np.arange(yb_erai[ind_en[i]],yb_erai[ind_en[i]+1]))
    yb_ind_ln = np.append(yb_ind_ln, np.arange(yb_erai[ind_ln[i]],yb_erai[ind_ln[i]+1]))
# Combine
yb_ind = [yb_ind_ens, yb_ind_en, yb_ind_ln]

# Load the data
ensodata = np.load("Training/Seas5_enso_distancedistr_"+repr(train_year)+
                   "years_k"+repr(k_nr)+".npz")
# Retrieve the distance distributions
dmean_enso = ensodata["mean_distance"]
dcov_enso = ensodata["covariance_distance"]
dcovinv_enso = ensodata["covariance_inverse_distance"]
dcovdet_enso = ensodata["covariance_determinant_distance"]
del ensodata

#%% BAYESIAN CLUSTERING ERA-INTERIM
"Basic Bayes applied to the ERA-Interim time series. The prior at the start of \
each year is given by the climatological values of ERA-Interim, after that the \
posterior from the previous timestep is propagated in time using the transition \
matrix. We consider two different cases: \
    1. Standard \
    2. Using the SEAS5 distance distr. during strong El Nino and La Nina years"

# Length of testing data
time_test = dev_erai.shape[0]

# Very strong (0) or strong (1) El Nino
en_ind = 0

# Initialize posterior regime probabilities
gam_erai_post = np.zeros((time_test, k_nr))
gam_erai_seas5_post = np.zeros((time_test, k_nr))
gam_erai_enso_post = np.zeros((time_test, k_nr))

# Keep track of prior and observed regime probabilities
gam_erai_prior = np.zeros((time_test, k_nr));       gam_erai_prior[0] = q_erai
gam_erai_seas5_prior = np.zeros((time_test, k_nr)); gam_erai_seas5_prior[0] = q_erai 
gam_erai_enso_prior = np.zeros((time_test, k_nr));  gam_erai_enso_prior[0] = q_erai 
gam_erai_obs = np.zeros((time_test, k_nr))
gam_erai_seas5_obs = np.zeros((time_test, k_nr))
gam_erai_enso_obs = np.zeros((time_test, k_nr))

# Loop through time
tmax = time_test
for t in range(tmax):
    # Compute observed probabilities
    gam_erai_obs[t] = prob_obs_erai_mvn(dev_erai[t], th_erai, dmean_erai, 
                                        dcovinv_erai, dcovdet_erai)
    # Using SEAS5 distance distributions
    gam_erai_seas5_obs[t] = prob_obs_erai_mvn(dev_erai[t], th_erai, dmean_train, 
                                        dcovinv_train, dcovdet_train)
    # Using the ENSO distance distributions
    if t in yb_ind[en_ind]: # El Nino
        gam_erai_enso_obs[t] = prob_obs_erai_mvn(dev_erai[t], th_erai, \
                                dmean_enso[en_ind], dcovinv_enso[en_ind], dcovdet_enso[en_ind])
    elif t in yb_ind[2]: # La Nina
        gam_erai_enso_obs[t] = prob_obs_erai_mvn(dev_erai[t], th_erai, \
                                dmean_enso[2], dcovinv_enso[2], dcovdet_enso[2])
    else: # Other years
        gam_erai_enso_obs[t] = gam_erai_seas5_obs[t]
        
    # Compute the posterior
    gam_post_temp = gam_erai_obs[t] * gam_erai_prior[t]
    gam_erai_post[t] = gam_post_temp / np.sum(gam_post_temp)
    # Using SEAS5 likelihood
    gam_post_seas5_temp = gam_erai_seas5_obs[t] * gam_erai_seas5_prior[t]
    gam_erai_seas5_post[t] = gam_post_seas5_temp / np.sum(gam_post_seas5_temp)
    # Using ENSO likelihood
    gam_post_enso_temp = gam_erai_enso_obs[t] * gam_erai_enso_prior[t]
    gam_erai_enso_post[t] = gam_post_enso_temp / np.sum(gam_post_enso_temp)
    
    # Get the new prior
    if t+1 < tmax:
        if t+1 in yb_erai: # Start new winter with climatological prior
            gam_erai_prior[t+1] = q_erai
            gam_erai_enso_prior[t+1] = q_erai
            gam_erai_seas5_prior[t+1] = q_erai
        else:
            gam_erai_prior[t+1] = np.dot(gam_erai_post[t], T_erai)
            gam_erai_seas5_prior[t+1] = np.dot(gam_erai_seas5_post[t], T_erai)
            gam_erai_enso_prior[t+1] = np.dot(gam_erai_enso_post[t], T_erai)

# Compute the hard assignment corresponding to observations and posterior
gam_erai_hard = np.zeros((time_test, k_nr))
seq_erai_hard = np.argmax(gam_erai_obs, axis=1)
for t in range(tmax):
    gam_erai_hard[t,seq_erai_hard[t]] = 1

# Clear memory
del gam_post_temp, tmax, time_test, t

# Collate the hard, observed and posterior gamma's
gam_erai_all = np.array([gam_erai_hard, gam_erai_obs, gam_erai_post, \
                         gam_erai_seas5_post, gam_erai_enso_post])

#%% BAYESIAN CLUSTERING FOR SEAS5
"Basic Bayes applied to the SEAS5 time series. As for ERA-Interim for each of \
the ensemble members."

# Length of testing data
time_test = dev_bayes.shape[0]

# Initialize obtained regime probabilities
gam_bayes_post = np.zeros((time_test, ensnr, k_nr))
gam_bayesC_post = np.zeros((time_test, ensnr, k_nr))

# Keep track of prior and observed regime probabilities
gam_bayes_prior = np.zeros((time_test, ensnr, k_nr))
gam_bayesC_prior = np.zeros((time_test, ensnr, k_nr))
gam_bayes_prior[0] = np.tile(q_train, (ensnr,1))      # Initial prior
gam_bayesC_prior[0] = np.tile(q_trainC, (ensnr,1))    # Initial prior
gam_bayes_obs = np.zeros((time_test, ensnr, k_nr))
gam_bayesC_obs = np.zeros((time_test, ensnr, k_nr))

# Loop through time
tmax = time_test
for t in range(tmax):
    # Compute observed probabilities
    gam_bayes_obs[t] = prob_obs_mvn(dev_bayes[t], th_train, dmean_train, 
                                    dcovinv_train, dcovdet_train, 0)
    gam_bayesC_obs[t] = prob_obs_mvn(dev_bayes[t], th_trainC, dmean_trainC, 
                                     dcovinv_trainC, dcovdet_trainC, 0)
    
    # Compute the posterior
    gam_post_temp = gam_bayes_obs[t] * gam_bayes_prior[t]
    gam_bayes_post[t] = gam_post_temp / np.sum(gam_post_temp, axis=1)[:,np.newaxis]
    gam_postC_temp = gam_bayesC_obs[t] * gam_bayesC_prior[t]
    gam_bayesC_post[t] = gam_postC_temp / np.sum(gam_postC_temp, axis=1)[:,np.newaxis]
    
    # Get the new prior
    if t+1 < tmax:
        if t+1 in yb_bayes: # Start new winter with climatological prior
            gam_bayes_prior[t+1] = np.tile(q_train, (ensnr,1))
            gam_bayesC_prior[t+1] = np.tile(q_trainC, (ensnr,1))
        else:
            gam_bayes_prior[t+1] = np.dot(gam_bayes_post[t], T_train)
            gam_bayesC_prior[t+1] = np.dot(gam_bayesC_post[t], T_trainC)

# Compute the hard assignment corresponding to observations and posterior
gam_bayes_hard = np.zeros((time_test, ensnr, k_nr))
seq_bayes_hard = np.argmax(gam_bayes_obs, axis=2)
for t in range(tmax):
    for i in range(ensnr):
        gam_bayes_hard[t,i,seq_bayes_hard[t,i]] = 1

# Clear memory
del gam_post_temp, time_test, t, i

#%% BAYESIAN CLUSTERING INCLUDING ENSEMBLE COUPLING FOR SEAS5
"Bayes with addapting transition probabilities following the ensemble"

# Length of testing data
time_test = dev_bayes.shape[0]

# Initialize obtained regime probabilities
gam_bayes_ens_post = np.zeros((time_test, ensnr, k_nr))
gam_erai_ens_post = np.zeros((time_test, k_nr))

# Keep track of prior and observed regime probabilities
gam_bayes_ens_prior = np.zeros((time_test, ensnr, k_nr))
gam_bayes_ens_prior[0] = np.tile(q_train, (ensnr,1))      # Initial prior
gam_bayes_ens_obs = np.zeros((time_test, ensnr, k_nr))
# Similarly for ERA-Interim
gam_erai_ens_prior = np.zeros((time_test, k_nr));   gam_erai_ens_prior[0] = q_erai 
gam_erai_ens_obs = np.zeros((time_test, k_nr))

# Save history of temperature perturbations
Tpert_hist = np.zeros((time_test, k_nr))
Tpert_erai_hist = np.zeros((time_test, k_nr))
# Tpert_erai_hist_new = np.zeros((time_test, k_nr))

# Set the number of days to average the ensemble over
# time_ens = 3

# Loop through time
tmax = time_test
for t in range(tmax):
    # print(t)
    # Compute observed probabilities
    gam_bayes_ens_obs[t] = prob_obs_mvn(dev_bayes[t], th_train, dmean_train, 
                                        dcovinv_train, dcovdet_train, 0)
    gam_erai_ens_obs[t] = prob_obs_erai_mvn(dev_erai[t+yb_erai[2]], th_erai, dmean_erai, 
                                            dcovinv_erai, dcovdet_erai)

    # Compute the posterior
    gam_post_ens_temp = gam_bayes_ens_obs[t] * gam_bayes_ens_prior[t]
    gam_bayes_ens_post[t] = gam_post_ens_temp / np.sum(gam_post_ens_temp, axis=1)[:,np.newaxis]
    # For ERA-Interim
    gam_post_erai_temp = gam_erai_ens_obs[t] * gam_erai_ens_prior[t]
    gam_erai_ens_post[t] = gam_post_erai_temp / np.sum(gam_post_erai_temp)
    
    # Get the new prior
    if t+1 < tmax:
        if t+1 in yb_bayes: # Start new winter with climatological prior
            gam_bayes_ens_prior[t+1] = np.tile(q_train, (ensnr,1))
            gam_erai_ens_prior[t+1] = q_erai
        else:
            # Mean of ensemble
            gam_bayes_ensmean = np.average( gam_bayes_ens_post[t], axis=0 )
            # Corrected for ERA-Interim - SEAS5 bias in regime frequencies
            gam_bayes_erai_ensmean = gam_bayes_ensmean * q_erai/q_train \
                / np.sum(gam_bayes_ensmean * q_erai/q_train)

            # Update the transition probabilities
            Tens = np.zeros((k_nr,k_nr));   Tens_erai = np.zeros((k_nr,k_nr))
            Tpert = np.zeros(k_nr);         Tpert_erai = np.zeros(k_nr)
            for j in range(k_nr):
                # Justified using Tc + T', focussing on persistence
                Tpert[j] = 1 - np.sum(T_train[:,j]*gam_bayes_ensmean) /gam_bayes_ensmean[j]
                # Tpert_erai[j] = 1 - np.sum(T_erai[:,j]*gam_bayes_ensmean) /gam_bayes_ensmean[j]
                Tpert_erai[j] = 1 - np.sum(T_erai[:,j]*gam_bayes_erai_ensmean) /gam_bayes_erai_ensmean[j]
                # Check for negative
                if T_train[j,j] + Tpert[j] > 0:
                    Tens[j,j] = T_train[j,j] + Tpert[j]
                else: # Back-up when negative
                    Tens[j,j] = 0
                    Tpert[j] = -T_train[j,j]
                # Same for ERA-Interm
                if T_erai[j,j] + Tpert_erai[j] > 0:
                    Tens_erai[j,j] = T_erai[j,j] + Tpert_erai[j]
                else: # Back-up when negative
                    Tens_erai[j,j] = 0
                    Tpert_erai[j] = -T_erai[j,j]
            # Store the history of the perturbations to the diagonal
            Tpert_hist[t] = Tpert
            Tpert_erai_hist[t] = Tpert_erai
            # Tpert_erai_hist_new[t] = Tpert_erai
            
            # Compute off-diagonal elements, which cannot be negative
            for j in range(k_nr):
                for i in range(k_nr):
                    if i != j:
                        Tens[i,j] = T_train[i,j] - Tpert[j] / (k_nr-1)  
                if np.amin(Tens[:,j]) < 0:
                    i01 = np.argmin(Tens[:,j])
                    Tens[i01,j] = 0
                    for i in range(k_nr):
                        if i != j and i != i01:
                            Tens[i,j] = T_train[i,j] - (Tpert[j]-T_train[i01,j]) \
                                        / (k_nr-2)
                    if np.amin(Tens[:,j]) < 0:
                        i02 = np.argmin(Tens[:,j])
                        Tens[i02,j] = 0
                        for i in range(k_nr):
                            if i != j and i != i01 and i != i02:
                                Tens[i,j] = T_train[i,j] - (Tpert[j]-T_train[i01,j] \
                                                            -T_train[i02,j]) \
                                            / (k_nr-3)
                        if np.amin(Tens[:,j]) < 0:
                            i03 = np.argmin(Tens[:,j])
                            Tens[i03,j] = 0
                            for i in range(k_nr):
                                if i != j and i != i01 and i != i02 and i != i03:
                                    Tens[i,j] = T_train[i,j] - (Tpert[j]-T_train[i01,j] \
                                                                -T_train[i02,j]-T_train[i03,j]) \
                                                / (k_nr-4)
                            if np.amin(Tens[:,j]) < 0:
                                i04 = np.argmin(Tens[:,j])
                                Tens[i04,j] = 0
                                for i in range(k_nr):
                                    if i != j and i != i01 and i != i02 and i != i03 and i != i04:
                                        Tens[i,j] = T_train[i,j] - (Tpert[j]-T_train[i01,j] \
                                                                    -T_train[i02,j]-T_train[i03,j] \
                                                                    -T_train[i04,j]) \
                                                    / (k_nr-5)
            # For ERA-Interim
            for j in range(k_nr):
                for i in range(k_nr):
                    if i != j:
                        Tens_erai[i,j] = T_erai[i,j] - Tpert_erai[j] / (k_nr-1)  
                if np.amin(Tens_erai[:,j]) < 0:
                    i01 = np.argmin(Tens_erai[:,j])
                    Tens_erai[i01,j] = 0
                    for i in range(k_nr):
                        if i != j and i != i01:
                            Tens_erai[i,j] = T_erai[i,j] - (Tpert_erai[j]-T_erai[i01,j]) \
                                        / (k_nr-2)
                    if np.amin(Tens_erai[:,j]) < 0:
                        i02 = np.argmin(Tens_erai[:,j])
                        Tens_erai[i02,j] = 0
                        for i in range(k_nr):
                            if i != j and i != i01 and i != i02:
                                Tens_erai[i,j] = T_erai[i,j] - (Tpert_erai[j]-T_erai[i01,j] \
                                                            -T_erai[i02,j]) \
                                            / (k_nr-3)
                        if np.amin(Tens_erai[:,j]) < 0:
                            i03 = np.argmin(Tens_erai[:,j])
                            Tens_erai[i03,j] = 0
                            for i in range(k_nr):
                                if i != j and i != i01 and i != i02 and i != i03:
                                    Tens_erai[i,j] = T_erai[i,j] - (Tpert_erai[j]-T_erai[i01,j] \
                                                                -T_erai[i02,j]-T_erai[i03,j]) \
                                                / (k_nr-4)
                            if np.amin(Tens_erai[:,j]) < 0:
                                i04 = np.argmin(Tens_erai[:,j])
                                Tens_erai[i04,j] = 0
                                for i in range(k_nr):
                                    if i != j and i != i01 and i != i02 and i != i03 and i != i04:
                                        Tens_erai[i,j] = T_erai[i,j] - (Tpert_erai[j]-T_erai[i01,j] \
                                                                    -T_erai[i02,j]-T_erai[i03,j] \
                                                                    -T_erai[i04,j]) \
                                                    / (k_nr-5)
            # Tens[Tens <= 0] = 0
            
            # Which then is used to compute the new prior
            gam_bayes_ens_prior[t+1] = np.dot(gam_bayes_ens_post[t], Tens.T)
            gam_erai_ens_prior[t+1] = np.dot(gam_erai_ens_post[t], Tens_erai.T)

            # Break if NaN's occur
            if np.isnan(gam_bayes_ens_prior[t+1,0,0]) == True:
                break

gam_erai_ens_post_new = gam_erai_ens_post

#%%
# Collate the hard, observed and posterior (for both standard and coupled) gamma's
gam_bayes_all = np.array([gam_bayes_hard, gam_bayes_obs, gam_bayes_post, 
                          gam_bayes_ens_post, gam_bayesC_post])
gam_erai_all = np.array([gam_erai_hard[yb_erai[2]:yb_erai[-3]], gam_erai_obs[yb_erai[2]:yb_erai[-3]], 
                         gam_erai_post[yb_erai[2]:yb_erai[-3]],
                         gam_erai_seas5_post[yb_erai[2]:yb_erai[-3]], gam_erai_ens_post])

#%% ANALYSE PERTURBATION DYNAMICS
"Compute the average and yearly perturbations to each of the diagonal elements \
of the transition matrix."

# Average
Tpert_av = np.mean(Tpert_hist, axis=0)
Tpert_erai_av = np.mean(Tpert_erai_hist, axis=0)
# Tpert_erai_av_new = np.mean(Tpert_erai_hist_new, axis=0)

# Yearly average
Tpert_year = np.zeros((ynr_bayes, k_nr))
Tpert_erai_year = np.zeros((ynr_bayes, k_nr))
Tpert_erai_year_new = np.zeros((ynr_bayes, k_nr))
for y in range(ynr_bayes):
    Tpert_year[y] = np.mean(Tpert_hist[yb_bayes[y]:yb_bayes[y+1]], axis=0)
    Tpert_erai_year[y] = np.mean(Tpert_erai_hist[yb_bayes[y]:yb_bayes[y+1]], axis=0)
    # Tpert_erai_year_new[y] = np.mean(Tpert_erai_hist_new[yb_bayes[y]:yb_bayes[y+1]], axis=0)

#%% OCCURRENCE AND AUTOCORRELATION
"Compute the occurrence rates and autocorrelation for both the ERA-Interim \
and SEAS5 results."

# Occurrence rate and 1-day autocorrelation
occ_erai = np.zeros((5,k_nr));      ac_erai = np.zeros((5,k_nr))
occ_bayes = np.zeros((5,k_nr));     ac_bayes = np.zeros((5,k_nr))
for i in range(5):
    occ_erai[i] = occurrence_prob(gam_erai_all[i], 0)
    # ac_erai[i] = autocor( gam_erai_all[i], yb_erai, 0)
    ac_erai[i] = autocor( gam_erai_all[i], yb_bayes, 0)
for i in range(5):
    occ_bayes[i] = occurrence_prob(gam_bayes_all[i], 1)
    ac_bayes[i] = autocor( gam_bayes_all[i], yb_bayes, 1)

# For constrained ERA-Interim
occ_eraiC = occurrence_prob( gam_eraiC, 0)
ac_eraiC = autocor( gam_eraiC, yb_erai[0:-1], 0)


#%% BOOTSTRAPPING
"Bootstrapping for occurrence rates."

# Set number of bootstrapping
btot = 500

# Initialize bootstrapping array for occurrence
occ_bayes_bs = np.zeros((5,btot, k_nr))
ac_bayes_bs = np.zeros((5,btot, k_nr))

for i in range(5):
    # Loop over the number of bootstrap years
    for b in range(btot):
        rnd.seed(b)
        
        # Create a random list of members to consider for each (random) year
        y_b = np.zeros((ynr_bayes, 1))
        for y in range(ynr_bayes):
            y_b[y] = rnd.sample(range(0,ensnr), 1)
        
        # Loop over the y_nr random years to select the sequences of m_nr members
        gam_b = np.array([[] for k in range(k_nr)]).T
        for y in range(ynr_bayes):
            gam_b = np.append(gam_b, gam_bayes_all[i][yb_bayes[y]:yb_bayes[y+1],int(y_b[y])],axis=0)
    
        # Compute the occurrence
        occ_bayes_bs[i,b] = occurrence_prob(gam_b, 0)
        ac_bayes_bs[i,b] = autocor( gam_b, yb_bayes, 0)

#%%
# Set lag up to which compute autocorrelation
lagmax = 20
# Lagged autocorrelation for ERA-I and SEAS5
acl_erai = np.zeros((5,k_nr,lagmax))
acl_bayes = np.zeros((5,k_nr,lagmax))
for i in range(5):
    # acl_erai[i] = autocor_lag(gam_erai_all[i], yb_erai, lagmax, 0)
    acl_erai[i] = autocor_lag(gam_erai_all[i], yb_bayes, lagmax, 0)
for i in range(5):
    acl_bayes[i] = autocor_lag(gam_bayes_all[i], yb_bayes, lagmax, 1)

# For constrained ERA-Interim
acl_eraiC = autocor_lag( gam_eraiC, yb_erai[0:-1], lagmax, 0)

#%%PLOT SETTINGS
"Set the regime labels and colors"

lab_reg = ["NAO+", "NAO-", "AR+", "SB+", "AR-", "SB-"]
col_reg = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
           'tab:brown']

#%% PLOT TIMESERIES
"Plot timeseries to compare observed probabilities and Bayesian probabilities \
with the prior and hard assignment respectively. For ERA-Interim also a \
comparison with result using persistent k-means is included"

# To save or not to save
saveplot = False

# ERA-Interim: Observed, Bayesian, Hard, Persistent (ORIGINAL)
tplot = 240     # Number of timesteps to show
# Create figure
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(4, 1, 1)
ax2 = fig.add_subplot(4, 1, 2)
ax3 = fig.add_subplot(4, 1, 3)
ax4 = fig.add_subplot(4, 1, 4)
for k in range(k_nr):
    ax1.plot(range(tplot), gam_erai_obs[0:tplot,k], ls='-', c=col_reg[k], label=lab_reg[k])
    ax2.plot(range(tplot), gam_erai_post[0:tplot,k], ls='-', c=col_reg[k])
    ax3.plot(range(tplot), gam_erai_hard[0:tplot,k], ls='-', c=col_reg[k])
    ax4.plot(range(tplot), gam_eraiC[0:tplot,k], ls='-', c=col_reg[k])
ax1.set_ylabel("Observation", fontsize=14)
ax2.set_ylabel("Bayesian", fontsize=14)
ax3.set_ylabel("Hard", fontsize=14)
ax4.set_ylabel("Persistent", fontsize=14)
ax1.set_xlim(0,tplot)
ax2.set_xlim(0,tplot)
ax3.set_xlim(0,tplot)
ax4.set_xlim(0,tplot)
ax1.set_ylim(0,1.05)
ax2.set_ylim(0,1.05)
ax3.set_ylim(0,1.05)
ax4.set_ylim(0,1.05)
ax1.set_xticklabels( () )
ax2.set_xticklabels( () )
ax3.set_xticklabels( () )
ax4.tick_params(axis='x', labelsize=12)
ax4.set_xlabel("Time (days)", fontsize=14)
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
fig.legend(loc = 'upper right',  fontsize=14)
fig.suptitle("ERA-Interim", fontsize=14)
plt.tight_layout()
if saveplot == True:
    plt.savefig("Plots/Occurrence/Occrate_2year_erai_compare.eps", format='eps')

#%%
saveplot = False

# ERA-Interim: Observed, Bayesian, Hard, Persistent (NEW)
# tplot = 121     # Number of timesteps to show

years_erai = np.arange(1979,2020)
year = 1993      # Year to consider
# List of indices corresponding to selected year
nr_year = np.where(years_erai == year)[0]
ind_year = np.arange(yb_erai[nr_year],yb_erai[nr_year+1])


gam_erai_hard_plot = np.copy(gam_erai_hard)
gam_erai_hard_plot[gam_erai_hard_plot == 0] = np.nan
gam_eraiC_plot = np.copy(gam_eraiC)
gam_eraiC_plot[gam_eraiC_plot == 0] = np.nan

# Create figure
fig, axs = plt.subplots(3,1,figsize=(12,6), gridspec_kw={'height_ratios': [5,5,1]})
for k in range(k_nr):
    axs[0].plot(ind_year-yb_erai[nr_year], gam_erai_obs[ind_year,k], ls='-', c=col_reg[k], label=lab_reg[k])
    axs[1].plot(ind_year-yb_erai[nr_year], gam_erai_post[ind_year,k], ls='-', c=col_reg[k])
    axs[2].scatter(ind_year-yb_erai[nr_year], gam_erai_hard_plot[ind_year,k], marker = '|', s=350, lw=5.8, c=col_reg[k])
    axs[2].scatter(ind_year-yb_erai[nr_year], gam_eraiC_plot[ind_year,k]*0.9, marker = '|', s=350, lw=5.8, c=col_reg[k])
axs[0].set_ylabel("Likelihood", fontsize=14)
axs[1].set_ylabel("Bayesian", fontsize=14)
axs[0].set_xlim(0,tplot-1)
axs[1].set_xlim(0,tplot-1)
axs[2].set_xlim(0,tplot-1)
axs[0].set_ylim(0,1.05)
axs[1].set_ylim(0,1.05)
axs[0].set_xticklabels( () )
axs[1].set_xticklabels( () )
axs[2].tick_params(axis='x', labelsize=12)
axs[0].tick_params(axis='y', labelsize=12)
axs[1].tick_params(axis='y', labelsize=12)
axs[2].tick_params(axis='y', labelsize=12)
axs[2].set_yticks([0.915,0.985])
axs[2].set_yticklabels(["Persistent", "Hard"], fontsize=14)
axs[2].set_xlabel("Time (days)", fontsize=14)
axs[0].grid()
axs[1].grid()
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
axs[2].spines['left'].set_visible(False)
fig.legend(loc = 'center right',  fontsize=14)
fig.suptitle("ERA-Interim, winter "+repr(year)+"-"+repr(year+1), fontsize=14)
plt.tight_layout(rect=[0, 0, 0.89, 1])
if saveplot == True:
    plt.savefig("Plots/Occurrence/Occrate_1year_erai_winter"+repr(year)+"_compareall.eps", format='eps')

#%%
# SEAS5: Prior, Observed and Posterior for both standard and coupled
nr_check = 42    # Ensemble number
tmax=120        # Number of timesteps to show
# Create figure
fig, axs = plt.subplots(3,1,figsize=(12,7), gridspec_kw={'height_ratios': [5,5,5]})
for k in range(k_nr):
    axs[0].plot(range(tmax), gam_bayes_prior[0:tmax,nr_check,k], ls='-', c=col_reg[k], label=lab_reg[k])
    axs[1].plot(range(tmax), gam_bayes_obs[0:tmax,nr_check,k], ls='-', c=col_reg[k])
    axs[2].plot(range(tmax), gam_bayes_post[0:tmax,nr_check,k], ls='-', c=col_reg[k])
    # axs[0].plot(range(tmax), gam_bayes_ens_prior[0:tmax,nr_check,k], ls='--', c=col_reg[k], label=lab_reg[k])
    # axs[2].plot(range(tmax), gam_bayes_ens_post[0:tmax,nr_check,k], ls='--', c=col_reg[k])
axs[0].vlines(yb_bayes, 0, 1.05, colors='grey', linestyles=':')
axs[1].vlines(yb_bayes, 0, 1.05, colors='grey', linestyles=':')
axs[2].vlines(yb_bayes, 0, 1.05, colors='grey', linestyles=':')
axs[0].set_ylabel("Prior", fontsize=14)
axs[1].set_ylabel("Likelihood", fontsize=14)
axs[2].set_ylabel("Bayesian", fontsize=14)
axs[2].set_xlabel("Time (days)", fontsize=14)
axs[0].set_xlim(0,tmax)
axs[1].set_xlim(0,tmax)
axs[2].set_xlim(0,tmax)
axs[0].set_ylim(0,1.05)
axs[1].set_ylim(0,1.05)
axs[2].set_ylim(0,1.05)
axs[0].set_xticklabels( () )
axs[1].set_xticklabels( () )
axs[2].tick_params(axis='x', labelsize=12)
axs[0].tick_params(axis='y', labelsize=12)
axs[1].tick_params(axis='y', labelsize=12)
axs[2].tick_params(axis='y', labelsize=12)
axs[0].grid()
axs[1].grid()
axs[2].grid()
fig.legend(loc = 'center right',  fontsize=14)
fig.suptitle("SEAS5 Ensemble Member "+repr(nr_check), fontsize=14)
plt.tight_layout(rect=[0, 0, 0.9, 1])
if saveplot == True:
    plt.savefig("Plots/Occurrence/Occrate_1year_ensnr"+repr(nr_check)+"_mvn_compens.eps", 
                format='eps')

#%%

# SEAS5: Observed, Bayesian, Hard (ORIGINAL)
nr_check = 42   # Ensemble number
tmax=120        # Number of timesteps to show
# Create figure
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
for k in range(k_nr):
    ax1.plot(range(tmax), gam_bayes_obs[0:tmax,nr_check,k], ls='-', c=col_reg[k])
    ax2.plot(range(tmax), gam_bayes_post[0:tmax,nr_check,k], ls='-', c=col_reg[k])
    ax3.plot(range(tmax), gam_bayes_hard[0:tmax,nr_check,k], ls='-', c=col_reg[k], label=lab_reg[k])
ax1.set_ylabel("Likelihood")
ax2.set_ylabel("Bayesian")
ax3.set_ylabel("Hard assignment")
ax1.set_ylim(0,1.05)
ax2.set_ylim(0,1.05)
ax3.set_ylim(0,1.05)
ax3.legend()
ax1.grid()
ax2.grid()
ax3.grid()
fig.suptitle("Ensemble Member "+repr(nr_check))
plt.tight_layout()
if saveplot == True:
    plt.savefig("Plots/Occurrence/Occrate_1year_ensnr"+repr(nr_check)+"_comparehard.eps", 
                format='eps')

#%%

saveplot = False

# SEAS5: Prior, Observed, Bayesian, Hard (NEW)
nr_check = 42   # Ensemble number
tplot = 121     # Number of timesteps to show
gam_bayes_hard_plot = np.copy(gam_bayes_hard[:,nr_check])
gam_bayes_hard_plot[gam_bayes_hard_plot == 0] = np.nan

# Create figure
fig, axs = plt.subplots(4,1,figsize=(12,8), gridspec_kw={'height_ratios': [6,6,6,1]})
for k in range(k_nr):
    axs[0].plot(range(tplot), gam_bayes_prior[0:tplot,nr_check,k], ls='-', c=col_reg[k], label=lab_reg[k])
    axs[1].plot(range(tmax), gam_bayes_obs[0:tmax,nr_check,k], ls='-', c=col_reg[k])
    axs[2].plot(range(tplot), gam_bayes_post[0:tplot,nr_check,k], ls='-', c=col_reg[k])
    axs[3].scatter(range(tplot), gam_bayes_hard_plot[0:tplot,k], marker = '|', s=200, lw=6.1, c=col_reg[k])
axs[0].set_ylabel("Prior", fontsize=14)
axs[1].set_ylabel("Observation", fontsize=14)
axs[2].set_ylabel("Bayesian", fontsize=14)
axs[0].set_xlim(0,tplot-1)
axs[1].set_xlim(0,tplot-1)
axs[2].set_xlim(0,tplot-1)
axs[3].set_xlim(0,tplot-1)
axs[0].set_ylim(0,1.05)
axs[1].set_ylim(0,1.05)
axs[2].set_ylim(0,1.05)
axs[0].set_xticklabels( () )
axs[1].set_xticklabels( () )
axs[2].set_xticklabels( () )
axs[2].tick_params(axis='x', labelsize=12)
axs[0].tick_params(axis='y', labelsize=12)
axs[1].tick_params(axis='y', labelsize=12)
axs[2].tick_params(axis='y', labelsize=12)
axs[3].tick_params(axis='y', labelsize=12)
axs[3].set_yticks([1])
axs[3].set_yticklabels(["Hard"], fontsize=14)
axs[3].set_xlabel("Time (days)", fontsize=14)
axs[0].grid()
axs[1].grid()
axs[2].grid()
axs[3].spines['top'].set_visible(False)
axs[3].spines['right'].set_visible(False)
axs[3].spines['left'].set_visible(False)
fig.legend(loc = 'center right',  fontsize=14)
fig.suptitle("SEAS5 Ensemble Member "+repr(nr_check), fontsize=14)
plt.tight_layout(rect=[0, 0, 0.89, 1])
if saveplot == True:
    plt.savefig("Plots/Occurrence/Occrate_1year_seas5_ensnr"+repr(nr_check)+"_compareall.eps", format='eps')

#%%
# SEAS5: Observed, Bayesian, Hard (NEW)
nr_check = 42   # Ensemble number
tplot = 120     # Number of timesteps to show
gam_bayes_hard_plot = np.copy(gam_bayes_hard[:,nr_check])
gam_bayes_hard_plot[gam_bayes_hard_plot == 0] = np.nan

# Create figure
fig, axs = plt.subplots(3,1,figsize=(12,6), gridspec_kw={'height_ratios': [6,6,1]})
for k in range(k_nr):
    axs[0].plot(range(tplot), gam_bayes_obs[0:tplot,nr_check,k], ls='-', c=col_reg[k], label=lab_reg[k])
    axs[1].plot(range(tplot), gam_bayes_post[0:tplot,nr_check,k], ls='-', c=col_reg[k])
    axs[2].scatter(range(tplot), gam_bayes_hard_plot[0:tplot,k], marker = '|', s=200, lw=6.1, c=col_reg[k])
axs[0].set_ylabel("Observation", fontsize=14)
axs[1].set_ylabel("Bayesian", fontsize=14)
axs[0].set_xlim(0,tplot-1)
axs[1].set_xlim(0,tplot-1)
axs[2].set_xlim(0,tplot-1)
axs[0].set_ylim(0,1.05)
axs[1].set_ylim(0,1.05)
axs[0].set_xticklabels( () )
axs[1].set_xticklabels( () )
axs[2].tick_params(axis='x', labelsize=12)
axs[0].tick_params(axis='y', labelsize=12)
axs[1].tick_params(axis='y', labelsize=12)
axs[2].tick_params(axis='y', labelsize=12)
axs[2].set_yticks([1])
axs[2].set_yticklabels(["Hard"], fontsize=14)
axs[2].set_xlabel("Time (days)", fontsize=14)
axs[0].grid()
axs[1].grid()
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
axs[2].spines['left'].set_visible(False)
fig.legend(loc = 'center right',  fontsize=14)
fig.suptitle("SEAS5 Ensemble Member "+repr(nr_check), fontsize=14)
plt.tight_layout(rect=[0, 0, 0.89, 1])
if saveplot == True:
    plt.savefig("Plots/Occurrence/Occrate_1year_seas5_ensnr"+repr(nr_check)+"_comparehard.eps", format='eps')


#%% PLOT OCCURRENCE RATE AND AUTOCORRELATION
"Plot the regime frequency and autocorrelation averaged over the whole time \
series for both ERA-Interim and SEAS5. Plotted are the values for a hard \
assignment, observational probability, bayesian probability, coupled bayesian \
for SEAS5 and time constrained for ERA-Interim. "

saveplot = False

data_label = ["Hard Assignment", "Likelihood", "Sequential Bayes", "Ensemble Bayes"]
dataset = [" ERA-I", " SEAS5"]
mrkrs = ['s', 'v', 'o', '^', 'P']

fig = plt.figure(figsize=(6,8))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
for i in range(4):
    if i < 3:
        ax1.scatter(np.arange(k_nr)-0.1, occ_erai[i], marker=mrkrs[i], 
                    facecolors='none', edgecolors=col_reg, s=40)
        ax2.scatter(np.arange(k_nr)-0.1, ac_erai[i], marker=mrkrs[i], 
                    facecolors='none', edgecolors=col_reg, s=40)
    ax1.scatter(np.arange(k_nr)+0.1, occ_bayes[i], marker=mrkrs[i], 
                color=col_reg, s=40, label=data_label[i])
    ax2.scatter(np.arange(k_nr)+0.1, ac_bayes[i], marker=mrkrs[i], 
                color=col_reg, s=40)
ax1.scatter(np.arange(k_nr)-0.1, q_eraiC, marker='X',facecolors='none', 
            edgecolors=col_reg, s=40, label="Time Constrained")
ax2.scatter(np.arange(k_nr)-0.1, ac_eraiC, marker='X',facecolors='none', 
            edgecolors=col_reg, s=40)
ax1.set_xticks(range(k_nr))
ax1.set_xticklabels(lab_reg, fontsize=14)
ax2.set_xticks(range(k_nr))
ax2.set_xticklabels(lab_reg, fontsize=14)
ax1.set_ylabel("Occurrence Rate", fontsize=14)
ax2.set_ylabel("Autocorrelation", fontsize=14)
ax1.tick_params(axis='y', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)
ax1.grid()
ax2.grid()
fig.legend(loc='lower center', fontsize=14, ncol=2)
plt.tight_layout(rect=[0, 0.12, 1, 1])
if saveplot == True:
   plt.savefig("Plots/Occurrence/Occrate_autocor_compare.eps", format='eps')

#%% BOXPLOT OF OCCURRENCE RATES
"Plot the occurrence rates with and without constraint, for both domains and \
for ERA-Interim with ERA-I or SEAS5 regimes, all in one plot. "

# Set labels and colors
data_label = ["Hard Assignment", "Likelihood", "Sequential Bayes", "Ensemble Bayes"]

# Compute mean, standard deviation and confidence interval
occ_bayes_mean = np.mean(occ_bayes_bs, axis=1)
occ_bayes_std = np.std(occ_bayes_bs, axis=1)

from scipy.stats import t
confidence=0.95
dof=500-1
t_crit = np.abs(t.ppf((1-confidence)/2,dof))

# Figure
fig = plt.figure(figsize=(12,5))

# Boxplot of bootstrapping
# bp0 = plt.boxplot(occ_bayes_bs[0], positions=np.arange(1,k_nr*4+1,4), patch_artist=True, \
#                   widths=0.8 )
# for patch, color in zip(bp0['boxes'], col_reg):
#     patch.set(facecolor=color)
# for median in bp0['medians']: 
#     median.set(color ='k')
bp0 = plt.errorbar(np.arange(1.25,k_nr*4+1,4), occ_bayes_mean[0], xerr=None, 
                    yerr=occ_bayes_std[0], fmt='.k', ecolor=col_reg, elinewidth=6, markersize=17)
bp1 = plt.errorbar(np.arange(1.25,k_nr*4+1,4), occ_bayes_mean[0], xerr=None, 
                   yerr=occ_bayes_std[0]*t_crit, fmt='.k',ecolor=col_reg, elinewidth=2)
    
# bp2 = plt.boxplot(occ_bayes_bs[2], positions=np.arange(2,k_nr*4+1,4), \
#                    patch_artist=True, widths=0.8 )
# for patch, color in zip(bp2['boxes'], col_reg):
#     patch.set(facecolor=color, hatch='///')
# for median in bp2['medians']: 
#     median.set(color ='k')
bp2 = plt.errorbar(np.arange(2,k_nr*4+1,4), occ_bayes_mean[2], xerr=None, 
                    yerr=occ_bayes_std[2], fmt='*k', ecolor=col_reg, elinewidth=6, markersize=11)
bp3 = plt.errorbar(np.arange(2,k_nr*4+1,4), occ_bayes_mean[2], xerr=None, 
                   yerr=occ_bayes_std[2]*t_crit, fmt='*k', ecolor=col_reg, elinewidth=2)

# Plot Era-Interim values
p00 = plt.plot(np.arange(2.75,k_nr*4+1,4), occ_erai[0], '.', c='grey', markersize=17)
p0C = plt.plot(np.arange(3.1,k_nr*4+1,4), occ_eraiC, 's', c='grey', markersize=9)
p02 = plt.plot(np.arange(3.45,k_nr*4+1,4), occ_erai[2], '*', c='grey', markersize=11)

# Get lines for equal distribution
plt.hlines(1/6, 0, k_nr*7, linestyles=':', color='k')

# Plot settings
plt.xlim(0, k_nr*4)     # Use to see value of no regime occurrence
# plt.ylim(0.1, 0.25)
plt.ylim(0.12, 0.23)
plt.xticks(np.arange(2,25,4), lab_reg, fontsize=14)
plt.yticks(fontsize=12)
plt.ylabel("Regime Frequency", fontsize=14)
plt.grid(axis='y')

# leg1 = plt.legend([p00[0], p0C[0], p02[0], bp0['boxes'][0], bp2['boxes'][0]], 
#                   ["Hard Assignment ERA-Interim", "Persistent Assignment ERA-Interim", 
#                    "Sequential Bayes ERA-Interim", "Hard Assignment SEAS5", "Sequential Bayes SEAS5"], \
#                   fontsize=14, title_fontsize=14, ncol=2, loc='center', bbox_to_anchor=(.5, 1.15))
leg1 = plt.legend([p00[0], p0C[0], p02[0], bp0[0], bp2[0]], 
                  ["Hard Assignment ERA-Interim", "Persistent Assignment ERA-Interim", 
                    "Sequential Bayes ERA-Interim", "Hard Assignment SEAS5", "Sequential Bayes SEAS5"], \
                  fontsize=14, title_fontsize=14, ncol=2, loc='center', bbox_to_anchor=(.5, 1.15))
# plt.legend([p00[0], p0C[0], p02[0]], ["Hard Assignment", "Persistent Assignment","Sequential Bayes"], \
#             title="ERA-Interim", fontsize=14, title_fontsize=14, ncol=1, loc='center', bbox_to_anchor=(0.94, 0.9))
# plt.gca().add_artist(leg1)
    
plt.tight_layout()
# plt.savefig("Plots/Statistics_analysis/Occ_boxplot_k"+repr(k_nr)+".eps", format='eps')
# plt.savefig("Plots/Statistics_analysis/Occ_confintplot_k"+repr(k_nr)+".pdf", format='pdf')

#%% BOXPLOT OF AUTOCORRELATION RATES
"Plot the occurrence rates with and without constraint, for both domains and \
for ERA-Interim with ERA-I or SEAS5 regimes, all in one plot. "

# Set labels and colors
data_label = ["Hard Assignment", "Likelihood", "Sequential Bayes", "Ensemble Bayes"]

# Compute mean, standard deviation and confidence interval
ac_bayes_mean = np.mean(ac_bayes_bs, axis=1)
ac_bayes_std = np.std(ac_bayes_bs, axis=1)

from scipy.stats import t
confidence=0.95
dof=500-1
t_crit = np.abs(t.ppf((1-confidence)/2,dof))

# Figure
fig = plt.figure(figsize=(12,4))

# Boxplot of bootstrapping
# bp0 = plt.boxplot(ac_bayes_bs[0], positions=np.arange(1,k_nr*4+1,4), patch_artist=True, \
#                   widths=0.8 )
# for patch, color in zip(bp0['boxes'], col_reg):
#     patch.set(facecolor=color)
# for median in bp0['medians']: 
#     median.set(color ='k')
bp0 = plt.errorbar(np.arange(1.25,k_nr*4+1,4), ac_bayes_mean[0], xerr=None, 
                    yerr=ac_bayes_std[0], fmt='.k', ecolor=col_reg, elinewidth=6, markersize=17)
bp1 = plt.errorbar(np.arange(1.25,k_nr*4+1,4), ac_bayes_mean[0], xerr=None, 
                   yerr=ac_bayes_std[0]*t_crit, fmt='.k',ecolor=col_reg, elinewidth=2)
    
# bp2 = plt.boxplot(ac_bayes_bs[2], positions=np.arange(2,k_nr*4+1,4), \
#                    patch_artist=True, widths=0.8 )
# for patch, color in zip(bp2['boxes'], col_reg):
#     patch.set(facecolor=color, hatch='///')
# for median in bp2['medians']: 
#     median.set(color ='k')
bp2 = plt.errorbar(np.arange(2,k_nr*4+1,4), ac_bayes_mean[2], xerr=None, 
                    yerr=ac_bayes_std[2], fmt='*k', ecolor=col_reg, elinewidth=6, markersize=11)
bp3 = plt.errorbar(np.arange(2,k_nr*4+1,4), ac_bayes_mean[2], xerr=None, 
                   yerr=ac_bayes_std[2]*t_crit, fmt='*k', ecolor=col_reg, elinewidth=2)

# Plot Era-Interim values
p00 = plt.plot(np.arange(2.75,k_nr*4+1,4), ac_erai[0], '.', c='grey', markersize=17)
p0C = plt.plot(np.arange(3.1,k_nr*4+1,4), ac_eraiC, 's', c='grey', markersize=9)
p02 = plt.plot(np.arange(3.45,k_nr*4+1,4), ac_erai[2], '*', c='grey', markersize=11)

# Plot settings
plt.xlim(0, k_nr*4)     # Use to see value of no regime occurrence
plt.ylim(0.55, .95)
plt.xticks(np.arange(2,25,4), lab_reg, fontsize=14)
plt.yticks(fontsize=12)
plt.ylabel("Autocorrelation", fontsize=14)
plt.grid(axis='y')

# leg1 = plt.legend([bp0['boxes'][0], bp2['boxes'][0]], ["Hard Assignment", "Sequential Bayes"], \
#                   title="SEAS5", fontsize=14, title_fontsize=14, ncol=1, loc='center', bbox_to_anchor=(.58, 0.12))
# plt.legend([p00[0], p0C[0], p02[0]], ["Hard Assignment", "Persistent Assignment","Sequential Bayes"], \
#             title="ERA-Interim", fontsize=14, title_fontsize=14, ncol=3, loc='center', bbox_to_anchor=(0.5, 0.95))
# plt.gca().add_artist(leg1)
    
plt.tight_layout()
# plt.savefig("Plots/Statistics_analysis/Autocor_boxplot_k"+repr(k_nr)+".eps", format='eps')
# plt.savefig("Plots/Statistics_analysis/Autocor_confintplot_k"+repr(k_nr)+".pdf", format='pdf')


#%% LAGGED AUTOCORRELATION
"Plot the autocorrelation as a function of lag for all regimes."

data_label = ["Hard Assignment", "Observation", "Bayesian", "Coupled Bayesian"]
linest = [':','--','-','-.']

laglist = np.arange(1, lagmax+1)

fig = plt.figure(figsize=(8,8))
for k in range(k_nr):
    ax = fig.add_subplot(3, 2, k+1)
    ax.set_title(lab_reg[k], fontsize=16)
    for i in range(4):
        if i < 3:
            ax.plot(laglist, acl_erai[i,k], ls=linest[i], \
                    color='grey')
        ax.plot(laglist, acl_bayes[i,k], ls=linest[i], \
                color=col_reg[k], label=data_label[i])
    ax.plot(laglist, acl_eraiC[k], ls='-.', color='grey', label="Time Constrained")     
    
    # Plot settings
    ax.set_xlim([1,13])
    ax.set_ylim([-0.1,0.95])
    
    if k >= 4:
        ax.set_xlabel("Lag (days)", fontsize=14)
    if k in [0,2,4]:
        ax.set_ylabel("Autocorrelation", fontsize=14)
    ax.set_yticks(np.arange(0,1,0.2))
    ax.tick_params(labelsize=12)
    ax.grid()
    if k == 0:
        fig.legend(loc='lower center', fontsize=14, ncol=3)
plt.tight_layout(rect=[0, 0.08, 1, 1])
if saveplot == True:
    plt.savefig("Plots/Occurrence/Autocorrelation_lag_compare.eps", format='eps')

del data_label, linest, i, k


#%% OCCURRENCE RATE FOR EACH YEAR
"Compute the occurrence rate for each year for ERA-Interim, using either \
ERA-I or SEAS5 regimes, and SEAS5, using either the constrained or \
unconstrained results, for both domains. "

ynr_erai = len(yb_erai)-1

# Initialize arrays
occ_bayes_year = np.zeros((5,ynr_bayes,k_nr))
occ_erai_year = np.zeros((5,ynr_bayes,k_nr))
# occ_erai_year = np.zeros((5,ynr_erai,k_nr))

occ_erai_year_new = np.zeros((ynr_bayes,k_nr))

# Get occurrence rates for each year
for i in range(5):
    for y in range(ynr_bayes):
        occ_bayes_year[i,y] = occurrence_prob(gam_bayes_all[i,yb_bayes[y]:yb_bayes[y+1]],1)
        occ_erai_year[i,y] = occurrence_prob(gam_erai_all[i,yb_bayes[y]:yb_bayes[y+1]],0)
        occ_erai_year_new[y] = occurrence_prob(gam_erai_ens_post_new[yb_bayes[y]:yb_bayes[y+1]],0)
    # for y in range(ynr_erai):
    #     occ_erai_year[i,y] = occurrence_prob(gam_erai_all[i,yb_erai[y]:yb_erai[y+1]],0)

#%% BOOTSTRAPPING OF OCCURRENCE RATES
"Use bootstrapping with 25 members a year (500 times) to get an error on the \
yearly SEAS5 occurrence rates. The settings for the bootstrapping can be \
changed. "

# Bootstrapping of SEAS5
btot = 500      # The total number of bootstrap years
m_nr = 25       # The number of members for bootstrapping

# Initialize bootstrap arrays
occ_bayes_y_b = np.zeros((5,btot,36,k_nr))
occ_bayes_y_noise_b = np.zeros((5,btot,36,k_nr));

# Yearly occurrence rates
print("Bootstrapping for yearly occurrence rates.")

for i in range(5):
    occ_bayes_y_b[i], occ_bayes_y_noise_b[i] = bootstrap_occ_year(gam_bayes_all[i], m_nr, btot)

del m_nr, i

#%% ENSO-YEARS
"Get the average occurrence rates and error bounds for strong El Nino/La Nina \
years."

# Make a list of the year data
years = np.arange(1981, 2017)

# Lists of El Nino/La Nina years
ens_y = np.array([1982, 1997, 2015])
en_y = np.array([1982, 1987, 1991, 1997, 2015])
ln_y = np.array([1988, 1998, 1999, 2007, 2010])

# Get the indices of these years
ind_ens = np.intersect1d(years, ens_y, return_indices=True)[1]
ind_en = np.intersect1d(years, en_y, return_indices=True)[1]
ind_ln = np.intersect1d(years, ln_y, return_indices=True)[1]

# Collate the bootstrappped data for those years
occ_bayes_ensy = np.zeros((5,1500,k_nr))
occ_bayes_eny = np.zeros((5,2500,k_nr))
occ_bayes_lny = np.zeros((5,2500,k_nr))
for i in range(5):
    occ_bayes_ensy[i] = np.reshape(occ_bayes_y_b[i,:,ind_ens], (len(ind_ens)*btot, k_nr))
    occ_bayes_eny[i] = np.reshape(occ_bayes_y_b[i,:,ind_en], (len(ind_en)*btot, k_nr))
    occ_bayes_lny[i] = np.reshape(occ_bayes_y_b[i,:,ind_ln], (len(ind_ln)*btot, k_nr))

#%% PLOT INTER-ANNUAL VARIABILITY IN OCCURRENCE RATES
"Plot the inter-annual variability in the occurrence rates for SEAS5 (with \
and without constraint, both domains), using the bootstrapping arrays to \
provide the error bounds. Also plot the ERA-I variability (ERA-I and SEAS5 \
regimes, both domains). It gets quite busy, so turn of certain things to make \
more clear. "

saveplot = False

# Make a list of the year data
years = np.arange(1981, 2017)

# Set labels, colors and linestyles
data_label = ["Hard Assignment", "Observation", "Bayesian", "Coupled Bayesian"]
linest = [':','--','-','-.']

# Plot
fig = plt.figure(figsize=(12,8))
for k in range(k_nr):
    ax = fig.add_subplot(3, 2, k+1)
    ax.set_title(lab_reg[k], fontsize=16)
    
    # Horizontal lines for strong El Nino years
    ax.vlines(ens_y, 0, 0.4, ls='-', color='indianred')
    # And La Nino years
    ax.vlines(ln_y, 0, 0.4, ls='-.', color='cornflowerblue')
    
    # For the Bayesian approach
    # Compute the two standard deviation lines
    sd1 = np.percentile(occ_bayes_y_b[3], 2.5, axis=0)
    sd2 = np.percentile(occ_bayes_y_b[3], 97.5, axis=0)
        
    # Plot the SEAS5 variability
    ax.fill_between(years, sd1[:,k], sd2[:,k], color=col_reg[k], alpha=0.5)
    ax.plot(years, sd1[:,k], ls='-', color=col_reg[k], alpha=0.5)
    ax.plot(years, sd2[:,k], ls='-', color=col_reg[k], alpha=0.5)
    Pcoupled, = ax.plot(years, occ_bayes_year[3,:,k], ls='-', color=col_reg[k])
    Pstandard, = ax.plot(years, occ_bayes_year[2,:,k], ls='--', color='k')
    # Pconstrained, = ax.plot(years, occ_bayes_year[4,:,k], ls='-.', color='k')
    
    # Plot the ERA-Interim variability
    Perai, = ax.plot(years, occ_erai_year[2,:,k], ls=(0,(1,1)), 
                     color='k', label="ERA-Interim")
    # Perai_seas5, = ax.plot(years, occ_erai_year[3,:,k], ls=(0,(1,1)), 
    #                   color='tab:pink', label="ERA-Interim SEAS5")
    Perai_seas5, = ax.plot(years, occ_erai_year_new[:,k], ls=(0,(1,1)), 
                      color='tab:pink', label="ERA-Interim SEAS5 T Bias")
    Perai_enso, = ax.plot(years, occ_erai_year[4,:,k], ls=(0,(1,1)), 
                     color='gray', label="ERA-Interim SEAS5 T")
        
    # # Get the noise level
    # sdN1 = np.percentile(occ_bayes_y_noise_b[i], 2.5, axis=0)
    # sdN2 = np.percentile(occ_bayes_y_noise_b[i], 97.5, axis=0)
    # sdN3 = np.average(sdN1,axis=0) * np.ones((36,k_nr))
    # sdN4 = np.average(sdN2,axis=0) * np.ones((36,k_nr))
    # ax.fill_between(years, sdN3[:,k], sdN4[:,k], \
    #             color='grey', alpha=0.25)
    # ax.plot(years, sdN3[:,k], ls=':', color='grey')
    # ax.plot(years, sdN4[:,k], ls=':', color='grey')

    # Add boxplots for strong El Nino/La Nina years
    # ENSO boxes coupled
    bp0 = plt.boxplot([occ_bayes_ensy[3,:,k], occ_bayes_lny[3,:,k]], positions=[2018, 2021], \
                      patch_artist=True, widths=0.8, whiskerprops=dict(linewidth=1.5), \
                      capprops=dict(linewidth=1.5), flierprops=dict(linewidth=1.5), \
                      showfliers=False, labels = [r"El Ni\~no", r"La Ni\~na"])
    for patch, color, htch in zip(bp0['boxes'], ['indianred', 'cornflowerblue'], [None, '/////']):
        patch.set(facecolor=color, hatch=htch, linewidth=1.5)
    for median in bp0['medians']: 
        median.set(color ='k', linewidth=1.5)
        
    # ENSO boxes standard
    bp1 = plt.boxplot([occ_bayes_ensy[2,:,k], occ_bayes_lny[2,:,k]], positions=[2019.5, 2022.5], \
                      patch_artist=True, widths=0.8, whiskerprops=dict(linewidth=1.5), \
                      capprops=dict(linewidth=1.5), flierprops=dict(linewidth=1.5), \
                      showfliers=False, labels = [r"El Ni\~no", r"La Ni\~na"])
    for patch, color, htch in zip(bp1['boxes'], ['lightgrey', 'lightgrey'], [None, '/////']):
        patch.set(facecolor=color, hatch=htch, linewidth=1.5)
    for median in bp1['medians']: 
        median.set(color ='k', linewidth=1.5)
            
    # Plot settings
    ax.set_ylim([0.0,0.38])
    ax.set_xlim([1979,2023.5])
    if k in [0,2,4]:
        ax.set_ylabel("Occurrence Rate", fontsize=14)
    if k in [4,5]:
        ax.set_xlabel("Year", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_yticks([0,0.1,0.2,0.3])
    # ax.set_xticks(np.append(np.arange(1980,2020,5),[2018.75, 2021.75]))
    # ax.set_xticklabels(np.append(np.arange(1980,2020,5),[r"El Ni\~no", r"La Ni\~na"]))
    ax.set_xticks(np.arange(1980,2020,5))
    ax.set_xticklabels(np.arange(1980,2020,5))
    ax.grid()
fig.legend([Pcoupled, Pstandard, Perai, bp0["boxes"][0], bp1["boxes"][0], \
            bp0["boxes"][1], bp1["boxes"][1], Perai_enso, Perai_seas5], \
           [r"Coupled Bayesian", "Standard Bayesian", "ERA-Interim", \
            "El Niño Coupled", "El Niño Standard", "La Niña Coupled", "La Niña Standard", \
            "ERA-Interim SEAS5 T" , "ERA-Interim SEAS5 T Bias"], \
            loc='lower center', ncol=3, fontsize=14)
plt.tight_layout(rect=[0, 0.115, 1, 1])
# plt.savefig("Plots/Constraint_Ensemble/Occurrence/Occ_year_stdev_phi"+phi+"_k"+repr(k_nr)+"_inclENSO_str.pdf", format='pdf')
if saveplot == True:
   plt.savefig("Plots/Occurrence/Occrate_year_compare_bayesiancoupled_transerai.pdf", format='pdf')

del data_label, linest, sd1, sd2, bp0, bp1, Pcoupled, Pstandard, Perai, k

#%% PLOT 

saveplot = False

# Make a list of the year data
years = np.arange(2017-ynr_bayes, 2017)

# Lists of El Nino/La Nina years
ens_y = np.array([1982, 1997, 2015])
en_y = np.array([1982, 1987, 1991, 1997, 2015])
ln_y = np.array([1988, 1998, 1999, 2007, 2010])

# Set labels and linestyles
data_label = ["Hard Assignment", "Observation", "Bayesian", "Coupled Bayesian"]
linest = [':','--','-','-.']

fig = plt.figure(figsize=(12,8))
for k in range(k_nr):
    ax = fig.add_subplot(3, 2, k+1)
    ax.set_title(lab_reg[k], fontsize=16)
    for i in range(4): # SEAS5
        ax.plot(years, occ_bayes_year[i,:,k], ls=linest[i], \
                color=col_reg[k], label=data_label[i])
    # for i in range(3): # ERA-Interim
    #     ax.plot(years, occ_erai_year[i,2:-2,k], ls=linest[i], \
    #             color='grey', label=data_label[i])
            
    # Plot settings
    ax.set_ylim([0.07,0.33])
    if k >= 4:
        ax.set_xlabel("Year", fontsize=14)
    if k in [0,2,4]:
        ax.set_ylabel("Occurrence Rate", fontsize=14)
    ax.set_yticks(np.arange(0.1,0.35,0.05))
    ax.tick_params(labelsize=12)
    ax.grid()
    if k == 0:
        fig.legend(loc='lower center', fontsize=14, ncol=4)
        
    # Horizontal lines for strong El Nino years
    ax.vlines(ens_y, 0, 0.4, ls='-', color='indianred')
    # And La Nino years
    ax.vlines(ln_y, 0, 0.4, ls='-.', color='cornflowerblue')

plt.tight_layout(rect=[0, 0.04, 1, 1])
if saveplot == True:
   plt.savefig("Plots/Occurrence/Occrate_year_compare_markov.eps", format='eps')

del data_label, linest, i, k

#%% 
"Plot the difference between the standard and coupled approach. \
Plot the updates transition probabilities with respect to the climatological ones."

saveplot = False

# Make a list of the year data
years = np.arange(2017-ynr_bayes, 2017)

# Set persistent elements, i.e. diagonal
T_pers = np.diag(T_train)

fig = plt.figure(figsize=(12,8))
for k in range(k_nr):
    ax = fig.add_subplot(3, 2, k+1)
    ax.set_title(lab_reg[k], fontsize=16)
    ax.plot(years, occ_bayes_year[3,:,k] - occ_bayes_year[2,:,k] - occ_bayes[3,k] + occ_bayes[2,k], ls='-', color=col_reg[k])
    # ax.plot(years, occ_erai_year[3,:,k] - occ_erai_year[2,:,k], ls='--', color='tab:pink')
    ax.plot(years, occ_erai_year[4,:,k] - occ_erai_year[2,:,k], ls='-', color='gray')
    ax.plot(years, occ_erai_year_new[:,k] - occ_erai_year[2,:,k], ls='-', color='tab:pink')

    axT = ax.twinx()
    axT.plot(years, Tpert_year[:,k], ls='--', color='k')
    axT.plot(years, Tpert_erai_year[:,k], ls='--', color='gray')
    axT.plot(years, Tpert_erai_year_new[:,k], ls='--', color='tab:pink')
    # axT.plot(years, (T_pers[k] + Tpert_year[:,k])/T_pers[k], ls='--', color='gray')
            
    # Plot settings
    ax.set_ylim([-0.05,0.05])
    axT.set_ylim([-0.5,0.5])
    # axT.set_ylim([0.3,1.2])
    axT.hlines(0,years[0]-1, years[-1]+1, ls='-', lw=1, color='k')
    ax.set_xlim([years[0]-1, years[-1]+1])
    ax.tick_params(labelsize=12)
    axT.tick_params(labelsize=12)
    ax.grid()
    axT.grid()
    
    if k >= 4:
        ax.set_xlabel("Year", fontsize=14)
    if k in [0,2,4]:
        ax.set_ylabel("Difference Occ. Rate", fontsize=14)
        axT.set_yticks(())
    if k in [1,3,5]:
        axT.set_ylabel("Transition Perturbation", fontsize=14)
        ax.set_yticks(())
    
    # Horizontal lines for strong El Nino years
    ax.vlines(ens_y, -0.4, 0.4, ls='-', color='indianred')
    # And La Nino years
    ax.vlines(ln_y, -0.4, 0.4, ls='-.', color='cornflowerblue')

plt.tight_layout(rect=[0, 0, 1, 1])
if saveplot == True:
   plt.savefig("Plots/Occurrence/Occrate_year_compare_diff_transitionperturbation_eraibias.eps", format='eps')

# del data_label, linest, k

#%%

Tpert_lr = np.zeros(k_nr)
for k in range(k_nr):
    Tpert_lr[k] = np.polyfit(years, (T_pers[k] + Tpert_year[:,k])/T_pers[k], deg=1)[0]
print(Tpert_lr)


#%%
for i in range(4):
    print(np.std(occ_bayes_year[i], axis=0))