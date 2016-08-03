import numpy as np
import matplotlib.pyplot as pl
from jhu2016_labs.mv_models import MVGaussian
# demonstrates Gibbs sampling algorithm for bivariate Gaussian


# proposal distribution here consists of the m conditional distributions
# it can be shown that the conditional distributions of multivariate Gaussian
# are again Gaussian
# here we only deal with the 2D case for now
def sampleProposal(BVG, lastState):
    # we have 2 vars, so we sample z_(i+1) conditioned on current z_i
    mu=BVG.mean
    sig=BVG.cov
    assert BVG.dim==2
    # initialize current state with the end state from last iteration
    state=lastState.copy()
    # each iteration will produce m samples (each m-dimensional)
    # here m=2, i.e. given [x(t), y(t)], this function call gives
    # smps=
    # [x(t+1), y(t)]
    # [x(t+1), y(t+1)]
    # state=[x(t+1), y(t+1)] # as initialization for next iteration
    smps=np.empty_like(sig)
    for i in range(2):  # loop over dimensions
        o = 1 if i ==0 else 0   # get index of the other dimension
        condMu = mu[i]  + sig[o,i]*(state[o]-mu[o])
        condSig = np.sqrt(1-sig[o,i]**2)
        state[i] = np.random.normal(condMu,condSig)
        smps[i,:] = state
    return smps, state

# main sampling routine; note we do not reject any samples
# size is the nxm matrix of samples; since there're m variables (corresponding
# to m-dimensional data) jointly sampled in each iteration, the actual number
# of iterations (calls to sampleProposal) is only n/m
def sample(BVG, size=(1000,2)):
    z=np.random.rand(size[1])     # initialze first state (random guess)
    niters=int(size[0]/size[1])  # actual number of Gibbs iterations
    samples=np.empty(size)
    for i in range(niters):
        sam, z = sampleProposal(BVG, z)
        samples[i*size[1]:(i+1)*size[1],:]=sam
    return samples


if __name__ == "__main__":
    bvg = MVGaussian(mean=np.array([0.,0.]),cov=np.array([[1,0.8],[0.8,1]]))
    sams = sample(bvg)
    # plot results
    fig=pl.figure()
    pl.title('Gibbs sampling for bivariate Gaussian')
    pl.scatter(sams[:,0], sams[:,1], alpha=0.2, label="Samples")
    pl.plot(sams[:20,0], sams[:20,1], color='g', label="First 20 samples path")

    # plot target
    dx=0.01
    x = np.arange(np.min(sams), np.max(sams), dx)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    Z=bvg.pdf(np.dstack((X,Y)))
    C = pl.contour(X, Y, Z, 10, label="True Distribution")
    pl.clabel(C, inline=1, fontsize=10)

    pl.legend(loc='upper left')
    pl.show()
    #pl.savefig('plots/Gibbs_demo.png')

