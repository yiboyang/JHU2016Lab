import numpy as np
import jhu2016_labs.models as models
import jhu2016_labs.plotting as plotting
from jhu2016_labs.gs_gmm import GSBayesianGMM
import matplotlib.pyplot as pl

true_model = models.GMM([-4, 0, 2], [1, 2, .7], [0.2, 0.2, 0.6])
X = true_model.sampleData(50)
Y = np.zeros_like(X) # The random variable has a single dimension!


# (blocked) Gibbs sampling demo
# hyper-parameters [pi_1, pi_2, ...], m, kappa, a, b
# you may try to change this.
bgmm = GSBayesianGMM([1, 1, 1], 0, 1, 2, 1)

y_min = -0.01
y_max = .4
fig, ax = plotting.plotGMM(true_model, y_min=y_min, y_max=y_max, label='True model')

for i in range(5):
    # Sample the latent variables
    Z = bgmm.sampleLatentVariables(X)

    # Update the parameters
    bgmm.sampleMeansVariances(X, Z)
    bgmm.sampleWeights(Z)

    # Just for plotting, this is not part of the Gibbs Sampling algorithm.
    plotting.plotGMM(bgmm.gmm, fig=fig, ax=ax, color='b', lw=.5, label='sampled GMM')

gmm_avg = bgmm.averageGMM()
plotting.plotGMM(gmm_avg, fig=fig, ax=ax, color='r', lw=3, label='avg GMM')
ax.plot(X, Y, '+', color='b', label='data')
handles, labels = ax.get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
ax.legend(handles, labels, loc='upper left')

pl.show()
