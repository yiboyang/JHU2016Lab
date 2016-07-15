'''Collapsed Gibbs Sampling for Finite GMM'''
import matplotlib.pyplot as pl
from .models import *

if __name__=='__main__':
    np.random.seed(0)

    # true/target distribution
    trueDist=MVGMM(np.array([[0,0],[2,2]]),np.array([np.eye(2), np.eye(2)]),[0.7,0.3])
    N=100  # number of data points
    X=trueDist.sampleData(N)
    K=2     # set the number of mixtures in advance (here we're cheating a bit)
    Z=np.random.randint(0,K,N)  # randomly initialize mixture ids

    # initialize prior Bayesian GMM with hyperparameters
    alpha=[0.6, 0.4]
    assert K==len(alpha)    # number of mixture components
    m0 = [1, 1]
    k0 = 2
    v0 = 3
    S0 = np.eye(2)
    BGMM = BayesianMVGMM(alpha, m0, k0, v0, S0)

    T=20   # number of sampling iterations
    Zs = BGMM.collapsedSampleLatentVars(X, Z, T)



    # plot the clustering progress
    fig = pl.figure()
    step = 5
    numplots = int((T+1)/step)

    colors = 'brcgmykw' # currently can have 8 colors/clusters max
    assert K <= len(colors)

    for i in range(0, (T+1), step):
        clr = 0
        z = Zs[i, :]
        fig = pl.figure()
        for k in range(K):
            Xk = X[np.where(z==k)[0], :]
            pl.scatter(Xk[:,0],Xk[:,1],alpha=0.7,color=colors[clr])
            clr+=1

        pl.title("Component Assignments from Collapsed Gibbs Sampling, Iteration "+str(i))
        pl.savefig('Collapsed_Gibbs_demo_iter'+str(i)+'.png')


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




