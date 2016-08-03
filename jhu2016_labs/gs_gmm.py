import numpy as np
from .models import *

# univariate Bayesian GMM implementing (blocked) Gibbs sampling

class GSBayesianGMM(object):

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
