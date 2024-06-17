# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 11:22:54 2024

@author: zpb4
"""
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))
import numpy as np
import matplotlib.pyplot as plt
import syn_util
import ensemble_verification_functions as verify
from time import localtime, strftime
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import seaborn as sns

col_cb = sns.color_palette('colorblind')
chefs = col_cb[0]  #hefs is colorblind blue

#date range of consideration (inclusive of all sites and HEFS avail dates)
sd = '1990-10-01' 
ed = '2019-08-15'

loc = 'YRS'
site = 'ORDC1'

ld = 1
lds = (1,3,5,10)
pcnt_rnk = (0.95,1)
pcnt_bse = (0,1)
low_pcnt_ecrps = (0,0.99)
upr_pcnt_ecrps = (0.99,1)

Q_hefs,Qf_hefs_inp,dowy_hefs,tocs,df_idx_hefs = syn_util.extract(sd,ed,forecast_type='hefs',syn_sample='',Rsyn_path='./',syn_vers='',forecast_param='',loc=loc,site=site,opt_pcnt=0.99,gen_setup='')
Qf_hefs = verify.onesamp_forecast_rearrange(Qf_hefs_inp)
ne = np.shape(Qf_hefs)[1]

wy_vec = df_idx_hefs.year.values
wy_vec[np.isin(df_idx_hefs.month,[10,11,12])] = wy_vec[np.isin(df_idx_hefs.month,[10,11,12])]+1

#1. plot rank histograms for various leads and percentiles
Qf_hefs = Qf_hefs[:,:,:]
Q_hefs = Q_hefs
dowy_hefs = dowy_hefs
fsort = False

sset_idx = np.arange(len(Q_hefs))
ne = np.shape(Qf_hefs)[1]

fig = plt.figure(layout='constrained',figsize=(15,3))
gs0 = fig.add_gridspec(1,5)

ax0 = fig.add_subplot(gs0[0])

rnk_hist_hefs = verify.onesamp_rnk_hist(Qf_hefs[sset_idx,:,ld-1], Q_hefs[sset_idx], pcnt_rnk, forc_sort=fsort)
x=np.linspace(1,ne+1,ne+1)-0.5
unif_dens = rnk_hist_hefs[3]
freqs = rnk_hist_hefs[0]

ax0.bar(x=x,height=freqs/np.sum(freqs),color=chefs)
ax0.set_xlabel('Ensemble Rank')
ax0.set_ylabel('Density')
ax0.set_title('Rank Histogram')
plt.legend(['HEFS'])
ax0.hlines(y=unif_dens,xmin=0,xmax=ne,colors='gray',linestyles='-',label='uniform')
ax0.set_ylim([0,2*unif_dens])
ax0.set_xlim([0,ne+2])
ax0.text(5,1.75*unif_dens,str(pcnt_rnk[0])+'-'+str(pcnt_rnk[1])+' percentile')
ax0.text(5,1.5*unif_dens,'entropy: '+str(round(rnk_hist_hefs[2],4)))
ax0.text(ne/2,1.05*unif_dens,'uniform',color='gray')

#2. plot cumulative rank histograms for various leads and percentiles
crnk_hist_hefs = np.zeros(ne+2)
crnk_hist_hefs[1:] = rnk_hist_hefs[1]

sns.set_theme()
sns.set_style('ticks')
sns.set_context('paper')

x = np.arange(ne+2)

#plt.subplot(4,4,1)
ax1 = fig.add_subplot(gs0[1])
ax1.plot(x,crnk_hist_hefs,color=chefs,linewidth=2)
ax1.set_xlabel('Ensemble Rank')
ax1.set_ylabel('Cumulative Density')
ax1.set_title('Cumulative Rank Histogram')
ax1.legend(['HEFS'],fontsize='large')
ax1.axline((0,0),slope=1/(ne+1),color='gray',linestyle='--')
ax1.plot(x,crnk_hist_hefs,color=chefs,linewidth=2)
ax1.set_xlim([0,ne+1])
ax1.set_ylim([0,1])
ax1.text(2,0.7,str(pcnt_rnk[0])+'-'+str(pcnt_rnk[1])+' percentile')
ax1.text(2,0.6,'Lead ' + str(ld))
ax1.text((ne-4),0.1,'a)',fontsize='xx-large',fontweight='bold')
ax1.xaxis.set_ticklabels([])


#3. plot binned spread error diagrams for various leads and percentiles
bse_hefs = verify.onesamp_bse(Qf_hefs[sset_idx,:,ld-1], Q_hefs[sset_idx], bins=20, pcntile=pcnt_bse)


axlim = np.max(bse_hefs)*1.5

ax2 = fig.add_subplot(gs0[2])
ax2.plot(bse_hefs[0,:],bse_hefs[1,:],color=chefs,linewidth=2)
ax2.set_xlabel('Spread ($\sqrt{\sigma^2}$)')
ax2.set_ylabel('Error ($\sqrt{mse}$)')
ax2.set_title('BSE Diagram')
ax2.legend(['HEFS'],fontsize='large')
ax2.axline((0,0),slope=1,color='gray',linestyle='--')
ax2.plot(bse_hefs[0,:],bse_hefs[1,:],color=chefs,linewidth=2)
ax2.set_xlim([0,axlim*1.05])
ax2.set_ylim([0,axlim*1.05])
ax2.text(0.05*axlim,0.75*axlim,str(pcnt_bse[0])+'-'+str(pcnt_bse[1])+' percentile')
ax2.text(0.05*axlim,0.65*axlim,'Lead ' + str(ld))
ax2.yaxis.get_major_locator().set_params(integer=True)
ax2.xaxis.set_ticklabels([])


#----------------------------------------------------------------------------
#4. eCRPS diagrams

ref_st = '1985-10-15'
ref_end = '2019-08-15'


Q_ref,dowy_ref = syn_util.extract_obs(ref_st, ref_end, Rsyn_path='./',loc=loc,site=site)
Qf_ref_ens_inp = verify.create_obs_ref_ens(Q_ref, dowy_ref, sd, ed)
Qf_ref_ens = Qf_ref_ens_inp[:,:]

ecrps_ss_hefs = np.empty((2,len(lds)))


for i in range(len(lds)):
    ecrps_hefs = verify.onesamp_ecrps(ensemble = Qf_hefs[sset_idx,:,lds[i]], tgt = Q_hefs[sset_idx], pcntile = upr_pcnt_ecrps)
    ecrps_ref = verify.onesamp_ecrps(ensemble = Qf_ref_ens[sset_idx,:], tgt = Q_hefs[sset_idx], pcntile = upr_pcnt_ecrps)
    
    ecrps_ss_hefs[0,i] = 1 - (ecrps_hefs[1] / ecrps_ref[1])
    
    ecrps_hefs = verify.onesamp_ecrps(ensemble = Qf_hefs[sset_idx,:,lds[i]], tgt = Q_hefs[sset_idx], pcntile = low_pcnt_ecrps)
    ecrps_ref = verify.onesamp_ecrps(ensemble = Qf_ref_ens[sset_idx,:], tgt = Q_hefs[sset_idx], pcntile = low_pcnt_ecrps)

    ecrps_ss_hefs[1,i] = 1 - (ecrps_hefs[1] / ecrps_ref[1])


ax3 = fig.add_subplot(gs0[3])
ax3.scatter((0.75,1.75,2.75,3.75),(ecrps_ss_hefs[0,0],ecrps_ss_hefs[0,1],ecrps_ss_hefs[0,2],ecrps_ss_hefs[0,3]),s=50,marker='^',color=chefs)
ax3.legend(['HEFS'],fontsize='large')
ax3.set_xlim([0,4.5])
ax3.set_ylim([0,1])
ax3.set_title('Ensemble CRPS Skill Score')
ax3.set_ylabel('eCRPS skill score')
ax3.text(2.75,0.75,str(low_pcnt_ecrps[0])+'-'+str(low_pcnt_ecrps[1])+' percentile')
ax3.set_xticks([0.75,1.75,2.75,3.75],['Lead %s' %(lds[0]),'Lead %s' %(lds[1]),'Lead %s' %(lds[2]),'Lead %s' %(lds[3])])


ax4 = fig.add_subplot(gs0[4])
ax4.scatter((0.75,1.75,2.75,3.75),(ecrps_ss_hefs[1,0],ecrps_ss_hefs[1,1],ecrps_ss_hefs[1,2],ecrps_ss_hefs[1,3]),s=50,marker='^',color=chefs)
ax4.legend(['HEFS'],fontsize='large')
ax4.set_xlim([0,4.5])
ax4.set_ylim([0,1])
ax4.set_title('Ensemble CRPS Skill Score')
ax4.set_ylabel('eCRPS skill score')
ax4.text(2.75,0.75,str(upr_pcnt_ecrps[0])+'-'+str(upr_pcnt_ecrps[1])+' percentile')
ax4.set_xticks([0.75,1.75,2.75,3.75],['Lead %s' %(lds[0]),'Lead %s' %(lds[1]),'Lead %s' %(lds[2]),'Lead %s' %(lds[3])])


fig.savefig('./figs/ld=%s_rnk-hist_pcnt=%s_bse_pcnt=%s_ecrps-ss_lds=%s_lowpcnt=%s_uprpcnt=%s.png' %(ld,pcnt_rnk,pcnt_bse,lds,low_pcnt_ecrps,upr_pcnt_ecrps),dpi=300,bbox_inches='tight')


#---------------------------------------------------------------------end------------------------------------------------------------------
