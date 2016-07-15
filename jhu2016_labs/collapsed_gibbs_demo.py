'''Collapsed Gibbs Sampling for Finite GMM'''
import numpy as np
#import matplotlib
#matplotlib.use('Agg')   # for use on remote machine
import matplotlib.pyplot as pl
from .models import *

# calculates posterior predictive probability of an unseen data vector x
# based on existing data assignments z and posterior predictive distributions
# from Gaussian components
# essentially calculates sum over k=1:K p(x*|z_k,X)*p(z_k|X)
# where X is the data we used to derive p(x*|z_k,X) and p(z_k|X) from;
# p(x*|z_k,X) is the same as the predictive distribution of x* evidenced on
# X_k, the set of data points assigned to cluster k; p(z_k|X) is the
# p
def predictive(z, p, x):
    K=len(p)
    I=np.array([np.where(z==k) for k in range(K)])[:,0]    # inverse indices
    Counts=np.array([len(i) for i in I])    # counts of data points for each mixture
    W = 1. * Counts/sum(Counts)
    return np.sum(np.exp([ (p[k].logpdf(x) + np.log(W[k])) for k in \
        range(K)]))

def logPredictive(z, p, x):
    return np.log(predictive(z, p, x))

if __name__=='__main__':
    np.random.seed(0)

    # true/target distribution
    trueDist=MVGMM(np.array([[0,0],[2,2]]),np.array([np.eye(2), np.eye(2)]),[0.7,0.3])
    N=100  # number of data points
    X=trueDist.sampleData(N)
    K=3     # set the number of mixtures in advance (here we're cheating a bit)
    Z=np.random.randint(0,K,N)  # randomly initialize mixture ids

    # initialize prior Bayesian GMM with hyperparameters
    alpha=np.ones(K)   # assuming 3 components here
    m0 = [1, 1]
    k0 = 2
    v0 = 3
    S0 = np.eye(2)
    BGMM = BayesianMVGMM(alpha, m0, k0, v0, S0)

    T=51   # number of sampling iterations
    Zs, Ps = BGMM.collapsedSampleLatentVars(X, Z, T)



    # plot the clustering progress
    fig = pl.figure()
    step = 5
    numplots = int((T+1)/step)

    colors = 'brcgmykw' # currently can have 8 colors/clusters max
    assert K <= len(colors)

    # for plotting contours
    dx=0.1
    x = np.arange(np.min(X), np.max(X), dx)
    csize = len(x)
    y = x.copy()
    XX, YY = np.meshgrid(x, y)
    points = np.dstack((XX,YY))
    P = np.empty((csize,csize))   # to hold posterior predictive densities

    for t in range(0, T, step):
        clr = 0
        z = Zs[t, :]    # labels from the t th iteration

        fig = pl.figure()

        p = Ps[t, :]
        for i in range(csize):
            for j in range(csize):
                P[i,j] = logPredictive(z, p, points[i,j])
        C = pl.contour(XX, YY, P, label="Predictive Distribution")

        for k in range(K):
            Xk = X[np.where(z==k)[0], :]
            pl.scatter(Xk[:,0],Xk[:,1],alpha=0.7,color=colors[clr])
            clr+=1

        pl.title("Component Assignments from Collapsed Gibbs Sampling,\
                Iteration "+str(t)+"log pdf contour")
        pl.savefig('Collapsed_Gibbs_demo_iter'+str(t)+'.png')


#    m = 2   # number of subplots per figure
#
#    # iteration numbers to be plotted
#    toPlot = np.arange(0, numplots, step=step)

#            # meshgrid for plotting true GMM
#            dx = 0.01
#            x = np.arange(np.min(Zs), np.max(Zs), dx)
#            X, Y = np.meshgrid(x, x)
#            locations = np.dstack((X, Y))
#            BGMM.mvgmm.gaussians[k].pdf(locations)




#    for i in range(m):
#        for j in range(n):
#            z = Zs[toPlot[i,j], :]
#            for k in range(K):
#                axarr[i, j].scatter(z[:,0], z[:,1], alpha=0.8,
#                        color=colors[clr++])
#            axarr[i, j].set_title("Gibbs Sampling Iteration "+str(toPlot[i,j]))
#





