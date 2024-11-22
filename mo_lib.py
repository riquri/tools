import math
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm.notebook import tqdm

import deeptime
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.plots import plot_implied_timescales
from deeptime.util.validation import implied_timescales

import MDAnalysis as mda
import MDAnalysis.transformations
import MDAnalysis.lib.distances



def plot_nice_pmf(xall, weights=None, ax=None, **kwargs):
    plt.style.use("tableau-colorblind10")
    H = np.histogram(xall, bins=kwargs.get("nbins", 60), weights=weights)
    datanumber = np.sum(H[0])
    with np.errstate(divide='ignore'):
        logedH = -np.log(H[0].T/datanumber)
    minimalenagy = np.min(logedH)
    PMF = logedH - minimalenagy
    
    if ax == None:
        plt.figure(figsize=(83/24.5,3))
        ax = plt.gca()
        no_ax = True
    else:
        no_ax = False
    ax.plot(H[1][:-1], PMF, color=kwargs.get("color", "#333333"), label=kwargs.get("label", None), linewidth=kwargs.get("linewidth",1), alpha=kwargs.get("alpha",1))   

    if type(kwargs.get("nbins", 60)) is not int and len(kwargs.get("nbins", 60)) > 1:
        ax.set_xlim((kwargs.get("nbins", None)[0],kwargs.get("nbins", None)[-1]))
    else:
        ax.set_xlim(kwargs.get("xlim",None))
    ax.set_ylim(kwargs.get("ylim",None))
    ax.set_xlabel(kwargs.get("xlabel",None))
    ax.set_ylabel(kwargs.get("ylabel",None))
    if no_ax == True:
        plt.tight_layout()
        plt.show()
    #plt.gca().set_aspect("equal")

    return PMF


def plot_nice_free_energy(xall, yall, weights=None, ax=None, **kwargs):
    
    H = np.histogram2d(xall,yall, bins=kwargs.get("nbins", 100), weights=weights)
    datanumber = np.sum(H[0])
    with np.errstate(divide='ignore'):
        logedH = -np.log(H[0].T/datanumber)
    minimalenagy = np.min(logedH)
    PMF = logedH - minimalenagy
    
    x = []
    for i in range(len(H[1])-1):
        x.append((H[1][i] + H[1][i+1]) /2)
    y = []
    for i in range(len(H[2])-1):
        y.append((H[2][i] + H[2][i+1]) /2)

    xx, yy = np.meshgrid(x, y) 
    
    if ax == None:
        plt.figure(figsize=(83/25.4,60/25.4))
        ax = plt.gca()
        no_ax = True
    else:
        no_ax = False
    contf = ax.contourf(xx, yy, PMF, cmap=kwargs.get("cmap","Greens_r"), levels=kwargs.get("levels",None))
    if kwargs.get("colorbar",True) == True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        cbar = plt.colorbar(contf, cax=cax)
        cbar.ax.set_xlabel("$k_\mathrm{B}T$")
    ax.contour(xx, yy, PMF, colors="white", linewidths=kwargs.get("linewidth",0.2), levels=kwargs.get("levels",None))
    ax.set_xlim(kwargs.get("xlim",None))
    ax.set_ylim(kwargs.get("ylim",None))
    ax.set_xlabel(kwargs.get("xlabel",None))
    ax.set_ylabel(kwargs.get("ylabel",None))
    
    if no_ax == True:
        plt.tight_layout()
        plt.show()
    #plt.gca().set_aspect("equal")

    return PMF


def do_MSM_1D(x, logging=False, **kwargs):
    matplotlib.rcParams['figure.dpi'] = 100
    orig_shape = np.shape(x)
    depptime_clustering = deeptime.clustering.RegularSpace(dmin=kwargs.get("dmin", 1), max_centers=kwargs.get("max_centers", 20))
    depptime_clustering.fit(np.hstack(x))
    clustering = depptime_clustering.fetch_model()

    dtrajs = list(np.reshape(clustering.transform(np.hstack(x)), (orig_shape[0], -1)))
    
    cc_x = clustering.cluster_centers.T[0]
    if logging == True:
        plt.figure(figsize=(83*2/25.4,60/25.4))
        ax=plt.subplot(1,2,1)
        PMF = plot_nice_pmf(np.hstack(x), ax=ax, **kwargs)
        for cc_x_i in cc_x:
            ax.axvline(cc_x_i, linewidth=0.1, color="gray", linestyle="dashed")
        
    
    traj_len = len(x[0])
    models = []
    lagtimes = np.linspace(1, traj_len-1, 40, dtype=int)
    for lagtime in lagtimes:
        try:
            models.append(MaximumLikelihoodMSM(use_lcc=True).fit_fetch(dtrajs, lagtime=lagtime, count_mode="sliding"))
        except:
            continue
    its_data = implied_timescales(models)
    
    if logging == True:
        ax=plt.subplot(1,2,2)
        plot_implied_timescales(its_data, n_its=10, ax=ax)
        ax.set_yscale("log")
        plt.show()
    
    msm_lag = kwargs.get("msm_lag", 500)
    M = MaximumLikelihoodMSM(use_lcc=True).fit_fetch(dtrajs, lagtime=msm_lag, count_mode="sliding")

    W = np.concatenate(M.compute_trajectory_weights(dtrajs))
    
    return np.array([np.hstack(x), W])


# Usage: do_MSM_2D(all_data, nbins=np.linspace(1,15,80), msm_lag=1500, max_centers=200, logging=True, dmin=1.5)
def do_MSM_2D(all_data,logging=False, ax=None, **kwargs):
    
    # Prepare figure
    if logging is True:
        plt.figure(figsize=(83*2/25.4,60/25.4))    

    # Discretization of trajectory using Regular Space method
    regspace = deeptime.clustering.RegularSpace(kwargs.get("dmin", 1.5), max_centers=kwargs.get("max_centers", 200))
    clustering = regspace.fit_fetch(np.vstack(all_data)[:, :2])
    orig_shape = np.shape(all_data)
    dtrajs = list(np.reshape(clustering.transform(np.vstack(all_data)[:, :2]), (orig_shape[0], -1)))

    # Get cluster centers
    cc_x = clustering.cluster_centers[:,0]
    cc_y = clustering.cluster_centers[:,1]

    # Plot discretized cluster centers
    if logging is True:
        ax=plt.subplot(1,2,1)
        plot_nice_free_energy(np.vstack(all_data)[:, 0], np.vstack(all_data)[:, 1], ax=ax, nbins=kwargs.get("nbins", 100))
        ax.scatter(cc_x,cc_y, color="black", s=1)
        
    
    # Calculate Implied Time Scales (ITS) 
    traj_len = np.shape(all_data)[1]
    models = []
    lagtimes = np.linspace(1, traj_len-1, 20, dtype=int)
    for lagtime in lagtimes:
        try:
            models.append(MaximumLikelihoodMSM(use_lcc=True).fit_fetch(dtrajs, lagtime=lagtime, count_mode="sliding"))
        except:
            continue
    its_data = implied_timescales(models)

    # Plot ITS data
    if logging is True:
        ax=plt.subplot(1,2,2)
        plot_implied_timescales(its_data, n_its=10, ax=ax)
        ax.set_yscale("log")
        plt.tight_layout()
    
    # Build MSM model
    msm_lag = kwargs.get("msm_lag", 500)
    M = MaximumLikelihoodMSM(use_lcc=True).fit_fetch(dtrajs, lagtime=msm_lag, count_mode="sliding")

    # Get weight for the trajectory
    W = np.concatenate(M.compute_trajectory_weights(dtrajs))
    
    # The weights are added on the last column of original
    return np.concatenate([np.vstack(all_data).T, W.reshape(-1,1).T]).T


def rolling(xx, size):
    b = np.ones(size)/size
    xx_mean = np.convolve(xx, b, mode="same")

    n_conv = math.ceil(size/2)

    # 補正部分
    xx_mean[0] *= size/n_conv
    for i in range(1, n_conv):
        xx_mean[i] *= size/(i+n_conv)
        xx_mean[-i] *= size/(i + n_conv - (size % 2)) 
	# size%2は奇数偶数での違いに対応するため

    return xx_mean
