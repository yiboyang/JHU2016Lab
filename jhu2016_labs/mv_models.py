import numpy as np
from scipy.misc import logsumexp
from scipy.special import gammaln
from scipy.stats import multivariate_normal
from scipy.stats._multivariate import multivariate_normal_frozen
from scipy.stats._multivariate import invwishart_frozen

# multivariate models commonly used in Bayesian GMMs

class MVGaussian(multivariate_normal_frozen):
    def __init__(self, *args, **kwargs):
        super(MVGaussian, self).__init__(*args,**kwargs)

    def sample(self, *args, **kwargs):
        return self.rvs(*args, **kwargs)

    def logLikelihood(self, *args, **kwargs):
        return self.logpdf(*args, **kwargs)

    @staticmethod
    def maximumLikelihood(X):
        N = len(X)
        mean = np.sum(X, axis = 0) / N
        cov = np.sum((np.outer((X[i, :] - mean), X[i, :] - mean) for i in range(N)),
                axis=  0) / N
        return MVGaussian(mean, cov)

class MVGMM(object):
    def __init__(self, means, covs, weights):
        '''Example: MVGMM(np.array([[0,0],[1,1]]),np.array([np.eye(2),
        np.eye(2)]), [0.3, 0.7])'''
        self.weights = weights
        self.gaussians = [MVGaussian(means[k], covs[k]) for k in
                          range(len(means))]
        self.dim = self.gaussians[0].dim

        assert np.isclose(np.sum(self.weights), 1.), 'The weights should sum up to one.'

    @property
    def k(self):
        return len(self.gaussians)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise KeyError('The index should be an integer.')
        if key >= self.k or key < 0:
            raise IndexError('Index out of bounds.')
        return self.gaussians[key]

    def sampleData(self, n=1000):
        X = []  # all the data
        for k in range(self.k):
            gaussian = self.gaussians[k]
            Xk = gaussian.sample(int(n * self.weights[k]))
            X.append(Xk)
        if self.dim == 1:   # for single dimensional case
            X = np.hstack(X)[:, None]
        X = np.vstack(X)

        np.random.shuffle(X)
        return X

    def pdf(self, X, sum_pdf=True):
        compVals = np.empty((len(X), self.k))
        for k in range(self.k):
            compVals[:, k] = self.weights[k] * self.gaussians[k].pdf(X)
        if sum_pdf:
            return compVals.sum(axis = 1)
        return compVals

    def logLikelihood(self, X):
        return np.sum(np.log(self.pdf(X)))

    def logLikelihoodPerComponent(self, X):
        return np.log(self.pdf(X, sum_pdf = False))

    def EStep(self, X):
        llh_per_comp = self.logLikelihoodPerComponent(X)
        norms = logsumexp(llh_per_comp, axis = 1)
        resps = np.exp(llh_per_comp.T - norms).T
        return resps

    # this is a naive implementation that can take a long time to converge;
    # especially compared to the GMM class implemented earlier, the 1D MVGMM
    # seems quite slow and often converges to a smaller maximum value
    # (very likely due to large roundoff errors in the MStep summations)
    def MStep(self, X, resps):
        N = X.shape[0]
        for k in range(self.k):
            mean = X.T.dot(resps[:, k]) / resps[:, k].sum()
            self.gaussians[k].mean = mean
            self.gaussians[k].cov= \
                    np.sum((resps[i, k] * \
                    np.outer((X[i, :] - mean), X[i, :] - mean) \
                        for i in range(N)), axis = 0) / resps[:, k].sum()

            self.weights[k] = np.sum(resps[:, k]) / N

    def _EMStep(self, X):
            Z = self.EStep(X)
            self.MStep(X, Z)

    def EM(self, X, threshold=1e-2):
        previous_llh = float('-inf')
        current_llh = self.logLikelihood(X)
        while current_llh - previous_llh > threshold:
            print(current_llh)
            self._EMStep(X)
            previous_llh = current_llh
            current_llh = self.logLikelihood(X)


class MVStudentT(object):

    def __init__(self, mean, prec, df):
        '''Mu (mean), Sigma ('variance'), Nu(degrees of freedom)'''
        self.mu = mean
        self.sigma = prec
        self.nu = df    # degrees of freedom
        self.D = len(mean)  # dimension

    def pdf(self, x):
        '''pdf of a SINGLE observation'''
        #TODO: extend to multiple observations X
#        Num = gamma((self.nu + self.D) / 2.)
#        Denom = (gamma(self.nu/2.)) * (self.nu**(self.D/2.)) * \
#        (np.pi**(self.D/alpha2.)) * ((np.linalg.det(self.sigma))**0.5) * (1 + \
#                (1./self.D)*np.dot(np.dot((x-self.mu),np.linalg.inv(self.sigma)),
#                    (x-self.mu)))**((self.nu+self.D)/2.)
#
#        return 1. * Num / Denom
        return np.exp(self.logpdf(x))

    def logpdf(self, x):
        '''log pdf of a SINGLE observation'''
        res = gammaln((self.nu + self.D) / 2.)
        res -= gammaln(self.nu/2)
        res -= (self.D/2.) * np.log(self.nu)
        res -= (self.D/2.) * np.log(np.pi)
        res -= 0.5 * np.log(np.linalg.det(self.sigma))
        res -= ((self.nu+self.D)/2.) * np.log((1 + \
                (1./self.D)*np.dot(np.dot((x-self.mu), \
                np.linalg.inv(self.sigma)), (x-self.mu))))

        return res



class InvWishart(invwishart_frozen):
    def __init__(self, *args, **kwargs):
        super(InvWishart, self).__init__(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.rvs(*args, **kwargs)

    def logLikelihood(self, *args, **kwargs):
        return self.logpdf(*args, **kwargs)



class NormalInvWishart(object):

    def __init__(self, m0, k0, v0, S0):
        '''NIW distribution over the mean and covariance matrix of a
        multivariate Gaussian distribution.
        Parameters:
        `m0` is our prior mean (expected value) for the Gaussian
        `k0` is how strongly we believe in `m0`
        `S0` is proportional to our prior mean for the covariance matrix
        `v0` is how strongly we believe in `S0`
        Example: NormalInvWishart(np.array([1,1]),2,3,np.eye(2))
        '''

        D = len(m0)
        assert v0 >= D, "degrees of freedom can't be less than dimension of scale matrix"
        self.D = D
        self.m = m0
        self.k = k0
        self.v = v0
        self.S = S0
        self.iw = InvWishart(df=v0, scale=S0)
        # can't use a MVGaussian yet because its cov depends on self.iw
        #self.mvg =

    def sample(self):
        cov = self.iw.sample()
        mean = multivariate_normal.rvs(self.m, (cov)/self.k)
        return mean, cov

    def logpdf(self, mu, sigma):
        '''Returns the log pdf of a SINGLE pair of mu and sigma'''
        assert len(mu) == sigma.shape[0] == sigma.shape[1], \
            'mu and sigma should have the same dimension'
        logPSig = self.iw.logpdf(sigma)
        logPMu = multivariate_normal.logpdf(mu, mean=self.m,
                cov=(sigma/self.k))

        return logPSig + logPMu

    def pdf(self, mu, sigma):
        return np.exp(self.logpdf(mu, sigma))

    def posterior(self, X):
        N = len(X)
        X_bar = X.mean(axis=0)
        k = self.k + N
        v = self.v + N
        m = (self.k*self.m + N*X_bar) / k
        S_ = np.sum((np.outer(X[i, :], X[i, :]) for i in range(N)), axis=0)
        S = self.S + S_ + self.k*np.outer(self.m, self.m) - k*np.outer(m, m)

        return NormalInvWishart(m, k, v, S)

    def predictiveDensity(self):
        mu = self.m
        sigma = (self.k + 1) * self.S / (self.k * (self.v - self.D + 1))
        nu = self.v - self.D + 1
        return MVStudentT(mu, sigma, nu)

