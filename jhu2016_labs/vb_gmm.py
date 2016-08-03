import numpy as np
from scipy.misc import logsumexp
from scipy.special import gamma, gammaln, psi
from .models import *

# univariate Bayesian GMMs implementing standard VB


class VarBayesGMM(object):

    def __init__(self, alphas, m, kappa, a, b):
        self.dir0 = Dirichlet(alphas)
        self.ng0 = NormalGamma(m, kappa, a, b)

        self.dir = Dirichlet(alphas)
        self.ng = [NormalGamma(self.ng0.sample()[0], kappa, a, b) for _ in alphas]
        self._cached_const = - .5 * np.log(np.pi*2)  # constant term added to log_rho during EStep (D=1)

    def EStep(self, x):
        ev_log_pi = self.dir.expLogPi()
        ev_log_lambda = np.array([psi(ng.a) - np.log(ng.b) for ng in self.ng])
        ev_lambda = np.array([ng.a / float(ng.b) for ng in self.ng])

        log_rho = np.array([ev_log_pi[k] + .5 * ev_log_lambda[k] + self._cached_const
                            - .5 * (ev_lambda[k] * np.square(x - ng.mean) + 1./ng.kappa)
                            for k, ng in enumerate(self.ng)]).T

        resp = np.exp((log_rho.T - logsumexp(log_rho, axis=1)).T)

        Nk = np.sum(resp, axis=0)
        xk_bar = x.dot(resp) / Nk

        #Sk = np.zeros((len(Nk),len(x)))
        #Sk = np.square((Sk + x).T - xk_bar)
        #Sk = np.sum(Sk * resp, axis=0) / Nk

        Sk = np.empty_like(Nk)
        for k in range(len(Sk)):
            Sk[k] = np.sum(resp[:, k] * np.square(x - xk_bar[k])) / Nk[k]

        # ev_llh = len(x) * np.sum(ev_log_pi + .5 * (ev_log_lambda - np.log(2*np.pi)))
        # for k, ng in enumerate(self.ng):
        #     ev_llh -= .5 * np.sum(ev_lambda[k] * np.square(x - ng.mean) + 1./ng.kappa)

        #ev_llh = logsumexp(log_rho, axis=1).sum()
        ev_llh = logsumexp(resp, axis=1).sum()

        return resp, ev_llh, (Nk, xk_bar, Sk)

    def MStep(self, x, stats):
        Nk, xk_bar, Sk = stats

        #print Nk, xk_bar, Sk

        self.dir = self.dir0.posterior(Nk)
        for k in range(len(self.ng)):
            #print k, self.ng[k].a, self.ng[k].b, self.ng[k].mean, self.ng[k].kappa
            self.ng[k] = self.ng0.posterior(Nk[k], xk_bar[k], Sk[k])

    def KLPosteriorPrior(self):
        return np.sum([ng.KL(self.ng0) for ng in self.ng]) + self.dir.KL(self.dir0)

    def sample_gmm(self):
        means, variances = zip(*[ng.sample() for ng in self.ng])
        variances = 1. / np.array(variances)
        weights = self.dir.sample()

        return GMM(means, variances, weights)

