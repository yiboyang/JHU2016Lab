import numpy as np
from scipy.misc import logsumexp
from scipy.special import gamma
from scipy.special import gammaln
from scipy.stats import multivariate_normal
from scipy.stats._multivariate import multivariate_normal_frozen
from scipy.stats._multivariate import invwishart_frozen
#import pdb

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

    def __init__(self, alpha):
        self.alpha = alpha
        self.alpha0 = np.sum(alpha)

    def sample(self):
        return np.random.dirichlet(self.alpha)

    def posterior(self, x):
        Nk = x.sum(axis=0)
        return Dirichlet(self.alpha + Nk)


class BayesianGMM(object):

    def __init__(self, alpha, m, kappa, a, b):
        self.Dir0 = Dirichlet(alpha)
        self.NG0 = NormalGamma(m, kappa, a, b)

        weights = self.Dir0.sample()
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
        dirichlet = self.Dir0.posterior(zs)
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


class MVStudentT():

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
#        (np.pi**(self.D/2.)) * ((np.linalg.det(self.sigma))**0.5) * (1 + \
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
        #TODO: maybe extend to allow multiple samples
#        cov = self.iw.sample(size)
#        if size==1:
#            cov = cov[None, ...]    # otherwise iw.sample(1) will return a
#            # "flat" sample instead of array of samples and disaster will ensue
#        mean = np.array([multivariate_normal.rvs(self.m, (c)/self.k) for c in cov])
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


class BayesianMVGMM(object):

    def __init__(self, alpha, m0, k0, v0, S0):
        alpha=np.array(alpha)
        m0=np.array(m0)
        self.Dir0 = Dirichlet(alpha)
        self.K = len(alpha) # number of mixture components
        self.NIW0 = NormalInvWishart(m0, k0, v0, S0)

        weights = self.Dir0.sample()
        means = []
        covs = []
        for k in range(self.K):
            m, c = self.NIW0.sample()
            means.append(m)
            covs.append(c)
        self.mvgmm = MVGMM(means, covs, weights)

#        self.count_mc = 1
#        self.count_w = 1
#        self.sumMeans = np.array(means)
#        self.sumCovs = np.array(covs)
#        self.sumWeights = np.array(weights)


    def collapsedSampleLatentVars(self, X, Z, niters):
        '''Sample component ids (conditioned on the rest of the data);
        Parameters: X: NxD data matrix X, Z: initial component assignments,
        niters: number of iterations'''
        N = len(X)  # number of data points
        K=self.K    # number of GMM components

        # samples of component assignments
        samples=np.empty((niters, N),dtype=int)  # rows are Zs

        # keep a record of posterior predictive dists (MVStudentT) per component
        predictives=np.empty((niters, K), dtype=object)   # each row has K post predictives

        # compute statistics for each cluster
        I=np.array([np.where(Z==k) for k in range(K)])[:,0]    # inverse indices
        #, i.e. lists of data points indices that belong to each mixture

        Counts=np.array([len(i) for i in I])    # counts of data points for each mixture
        compIds = np.arange(0, K)         # list of component ids

        for t in range(niters):
            for i in range(N):
                x = X[i, :]   # ith data entry, X_i
                z = Z[i]    # mixture/component id of x
                # pdb.set_trace()
                Ix = I[z]   # data indices of the zth component

                # remove X_i's statistics from component Z[i]
                I[z] = Ix[np.where(Ix != i)]
                Counts[z] -= 1
#                if Counts[z] <= 0:
#                    print("Warning: " + str(z) + "th component became empty")
#                    Counts[z]=0
#                    continue


                # compute log likelihood of Z_i conditioned on Z\_i and alpha;
                # this is only a likelihood because we're not dividing by the
                # total counts; the following produces a K dimensional vector
                # [24.26, P.843, Murphy text]
                logLZ_i = np.ones(K)*np.log((self.Dir0.alpha) + Counts)

                # compute conditional log probability of x (i.e. X_i),
                # conditioned on Z_i being assigned to each of K components)
                # [24.27, P.843, Murphy text]
                logPx = np.empty(K)
                for k in range(K):
                    # this is the posterior predictive, with evidence being all
                    # the data assigned to cluster k except for X_i
                    logPx[k] = \
                        self.NIW0.posterior(X[I[k],:]).predictiveDensity().logpdf(x)

                # the multinomial probability of Z_i is proportional to the
                # product of the two terms above (hence normalization)
                # [24.23, P.842, Murphy text]
                logPZ = logLZ_i + logPx
                probZ = np.exp(logPZ - logsumexp(logPZ))

                # draw/sample a new component id for X_i from the above
                # probability distribution
                knew = np.random.choice(compIds, p=probZ)
                # add X_i's statistics to the component Z_i = knew
                Z[i] = knew
                I[knew] = np.hstack((I[knew],i))
                Counts[knew] +=1

                # record the new assignment for X_i
                samples[t, i] = knew

            # calculate the posterior predictive probility distributions for
            # each component under this new assignment we just sampled
            for k in range(K):
                predictives[t, k] = self.NIW0.posterior(X[I[k],:]).predictiveDensity()



        return samples, predictives


#    def averageGMM(self):
#        avg_means = self.sumMeans / self.count_mv
#        avg_variances = self.sumVariances / self.count_mv
#        avg_weights = self.sumWeights /  self.count_w
#        return GMM(avg_means, avg_variances, avg_weights)


