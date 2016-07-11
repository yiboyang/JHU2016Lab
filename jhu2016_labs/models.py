
"""Gaussian distribution."""

import numpy as np
from scipy.misc import logsumexp
from scipy.special import gamma
#from scipy.stats import multivariate_normal
from scipy.stats._multivariate import multivariate_normal_frozen

class Gaussian(object):

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def pdf(self, x):
        norm = np.sqrt(2 * np.pi * self.var)
        return np.exp((-.5 / self.var) * ((x - self.mean) ** 2)) / norm

    def logLikelihood(self, x):
        return np.sum(np.log(self.pdf(x)))

    @staticmethod
    def maximumLikelihood(X):
        N = len(X)
        mean = np.sum(X)/N
        var = np.sum(X**2)/N
        return Gaussian(mean, var)


class GMM(object):

    def __init__(self, means, variances, weights):
        self.weights = weights
        self.gaussians = [Gaussian(means[k], variances[k]) for k in
                          range(len(means))]

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
            Xk = np.random.normal(gaussian.mean, np.sqrt(gaussian.var),
                                  int(n * self.weights[k]))
            X.append(Xk)
        X = np.hstack(X)

        np.random.shuffle(X)
        return X

    def pdf(self, x, sum_pdf=True):
        retval = np.zeros((len(x), self.k))
        for k in range(self.k):
            retval[:,k] = self.weights[k] * self.gaussians[k].pdf(x)
        if sum_pdf:
            retval = retval.sum(axis=1)
        return retval

    def logLikelihood(self, X):
        return np.sum(np.log(self.pdf(X)))

    def logLikelihoodPerComponent(self, X):
        return np.log(self.pdf(X, sum_pdf=False))

    def EStep(self, X):
        llh_per_comp = self.logLikelihoodPerComponent(X)
        norms = logsumexp(llh_per_comp, axis=1)
        resps = np.exp(llh_per_comp.T - norms).T
        return resps

    def MStep(self, X, resps):
        for k in range(self.k):
            mean = np.sum(resps[:, k] * X) / resps[:, k].sum()
            self.gaussians[k].mean = mean
            self.gaussians[k].var = \
                    np.sum((resps[:, k] * ((X - mean) ** 2))) / resps[:, k].sum()

            self.weights[k] = np.sum(resps[:, k]) / X.shape[0]

    def _EMStep(self, X):
            Z = self.EStep(X)
            self.MStep(X, Z)

    def EM(self, X, threshold=1e-2):
        previous_llh = float('-inf')
        current_llh = self.logLikelihood(X)
        while current_llh - previous_llh > threshold:
            self._EMStep(X)
            previous_llh = current_llh
            current_llh = self.logLikelihood(X)


class Dirichlet(object):

    def __init__(self, alphas):
        self.alphas = alphas

    def sample(self):
        return np.random.dirichlet(self.alphas)

    def posterior(self, x):
        Nk = x.sum(axis=0)
        return Dirichlet(self.alphas + Nk)


class BayesianGMM(object):

    def __init__(self, alphas, m, kappa, a, b):
        self.dir0 = Dirichlet(alphas)
        self.NG0 = NormalGamma(m, kappa, a, b)

        weights = self.dir0.sample()
        means = []
        variances = []
        for k in range(len(weights)):
            m, p = self.NG0.sample()
            means.append(m)
            variances.append(1/p)
        self.gmm = GMM(means, variances, weights)

        self.count_mv = 1
        self.count_w = 1
        self.sumMeans = np.array(means)
        self.sumVariances = np.array(variances)
        self.sumWeights = np.array(weights)

    def sampleLatentVariables(self, x):
        llh_per_comp = self.gmm.logLikelihoodPerComponent(x)
        norms = logsumexp(llh_per_comp, axis=1)
        resps = np.exp(llh_per_comp.T - norms).T
        zs = []
        for i in range(len(resps)):
            zs.append(np.random.multinomial(1, resps[i]))
        return np.array(zs, dtype=int)

    def sampleMeansVariances(self, x, zs):
        for k in range(self.gmm.k):
            indices = np.where(zs[:, k] == 1)
            if len(indices[0]) > 0:
                NG = self.NG0.posterior(x[indices])
            else:
                NG = self.NG0
            mean, prec = NG.sample()
            self.gmm.gaussians[k].mean = mean
            self.gmm.gaussians[k].var = 1 / prec
        self.sumMeans += np.array([g.mean for g in self.gmm.gaussians])
        self.sumVariances += np.array([g.var for g in self.gmm.gaussians])
        self.count_mv += 1

    def sampleWeights(self, zs):
        dirichlet = self.dir0.posterior(zs)
        self.gmm.weights = dirichlet.sample()
        self.sumWeights += self.gmm.weights
        self.count_w += 1

    def averageGMM(self):
        avg_means = self.sumMeans / self.count_mv
        avg_variances = self.sumVariances / self.count_mv
        avg_weights = self.sumWeights /  self.count_w
        return GMM(avg_means, avg_variances, avg_weights)


class NormalGamma(object):

    def __init__(self, mean, kappa, a, b):
        self.mean = mean
        self.kappa = kappa
        self.a = a
        self.b = b

    def sample(self):
        prec = np.random.gamma(self.a, 1/self.b)
        mu = np.random.normal(self.mean, np.sqrt(1/(self.kappa*prec)))
        return mu, prec

    def _normalPdf(self, x, y):
        var = 1/(self.kappa * y)
        norm = np.sqrt(2 * np.pi * var)
        return np.exp((-.5 / var) * ((x - self.mean) ** 2)) / norm

    def _gammaPdf(self, y):
        norm = (self.b**self.a)/gamma(self.a)
        return y**(self.a - 1) * np.exp(-self.b*y) * norm

    def pdf(self, x, y):
        assert len(x) == len(y), 'x and y should have the same dimension.'

        return self._normalPdf(x, y) * self._gammaPdf(y)

    def posterior(self, x):
        N = len(x)
        x_bar = x.mean()
        m = (self.kappa * self.mean + N * x_bar)/(self.kappa + N)
        kappa = self.kappa + N
        a = self.a + .5* N
        b = self.b + 0.5 * np.sum((x - x_bar)**2)
        b += (self.kappa*N*((x_bar - self.mean)**2)) / (2*(self.kappa + N))
        return NormalGamma(m, kappa, a, b)

    def predictiveDensity(self):
        nu = 2*self.a
        gamma = (self.a*self.kappa) /(self.b*(self.kappa + 1))
        return StudentT(self.mean, nu, gamma)


class StudentT(object):

    def __init__(self, mean, nu, gamma):
        self.mean = mean
        self.nu = nu
        self.gamma = gamma

    def pdf(self, x):
        norm = np.sqrt(self.gamma/(np.pi*self.nu))
        norm *= gamma(.5*self.nu + .5) / gamma(0.5*self.nu)
        density = 1 + (self.gamma*(x-self.mean)**2)/self.nu
        density = density**(-.5*self.nu - .5)
        return norm * density

    def logLikelihood(self, x):
        return np.sum(np.log(self.pdf(x)))


# multivariate extensions
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
        cov = np.sum((np.outer((X[i, :] - mu), X[i, :] - mu) for i in range(N)),
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

    # this is a naiive implementation that can take a long time to converge;
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


