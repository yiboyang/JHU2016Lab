import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import multivariate_normal
# demonstrates Gibbs sampling algorithm for bivariate Gaussian

# target distribution is of the form p(z1,z2,...zm); here we use a joint Gaussian
# dist (m=2, bivariate)

class MVGaussian(object):

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.mvg = multivariate_normal(mean, cov)

    def pdf(self, x):
        return self.mvg.pdf(x)

    def sample(self, x):
        return np.random.multivariate_normal(self.mean, self.cov)

# proposal distribution here consists of the m conditional distributions
# it can be shown that the conditional distributions of multivariate Gaussian
# are again Gaussian
# here we only deal with the 2D case for now
def sampleProposal(BVG, lastState):
    # we have 2 vars, so we sample z_(i+1) conditioned on current z_i
    mu=BVG.mean
    sig=BVG.cov
    m=len(mu)   # m is the number of dimensions
    assert m==2
    # initialize current state with the end state from last iteration
    state=lastState.copy()
    # each iteration will produce m samples (each m-dimensional)
    # here m=2, i.e. given [x(t), y(t)], this function call gives
    # smps=
    # [x(t+1), y(t)]
    # [x(t+1), y(t+1)]
    # state=[x(t+1), y(t+1)] # as initialization for next iteration
    smps=np.empty_like(sig)
    for i in range(m):  # loop over dimensions
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
    z=np.random.rand(2)     # initialze first states (random guess)
    niters=size[0]/size[1]  # actual number of Gibbs iterations
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
    pl.plot(sams[:,0], sams[:,1],'ro', label="Samples")
    pl.plot(sams[:50,0], sams[:50,1], label="First 50 samples path")
#
#    gx=np.arange(-4,4,0.1)
#    gy=targetPdf(gx)
#    pl.hist(sam,normed=True,label="Sampled Distribution")
#    pl.plot(gx,gy,'ro',label="True Distribution")
    pl.legend(loc='upper left')
    pl.savefig('Gibbs_demo.pdf')

