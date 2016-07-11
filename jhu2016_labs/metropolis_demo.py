import numpy as np
import matplotlib.pyplot as pl

# demonstrates Metropolis algorithm
# this is a specific case of Metropolis-Hastings algorithm, in that we use a
# symmetric proposal distribution

# target distribution is some single dimensional Guassian pdf (note it doesn't
# have to be a real, normalized pdf; a likehood function can also be sampled to
# produce samples from the true distribution)
def targetPdf(z):
    mean,var=0,0.95
    return np.exp((-.5/var)*(z-mean)**2) / np.sqrt(2*np.pi*var)

# proposal distribution, q(z*|z), where z is the current state
# a common choice is a Gaussian centered on the current state; this is a
# symmetric proposal distribution (i.e. the conditinal distributions, q(z|z*)
# and q(z*|z) are equal)
def sampleProposal(curz):
    '''Sample from proposal distribution based on current state'''
    rho=0.8     # scale of the proposal distribution; determines proportion of
                # of accepted distribution and rate of exploring state space
    return np.random.normal(curz,rho)


def sample(size=1000):
    z=0     # initialze first state (random guess)
    nreject=0   # number of rejected candidates
    samples=np.zeros(size)
    samples[0]=z
    for i in range(size):
        candidate=sampleProposal(z)     # sample a candidate from proposal dist
        aprob=min(1,targetPdf(candidate)/targetPdf(z))   # acceptance prob;
                                                         # note this doesn't
                                                         # involve the proposal
                                                         # distribution because
                                                         # it is symmetric so
                                                         # q(z|z*)/q(z*|z)=1
        if aprob > np.random.uniform(0,1):
            # accept
            z=candidate
        else:
            nreject+=1
        samples[i]=z   # always keep this sample (even if candidate sample
                       # is rejected
    return samples, nreject



if __name__ == "__main__":
    sam, rej = sample()
    print('Rejected '+str(rej)+' out of '+str(len(sam))+' samples')
    # plot results
    fig=pl.figure()
    pl.title('Metropolis sampling')
    gx=np.linspace(-4,4,1000)
    gy=targetPdf(gx)
    pl.hist(sam,normed=True,label="Sampled Distribution")
    pl.plot(gx,gy,'r',lw=2,label="True Distribution")
    pl.legend(loc='upper left')
    pl.show()
    #pl.savefig('Metropolis_demo.pdf')

