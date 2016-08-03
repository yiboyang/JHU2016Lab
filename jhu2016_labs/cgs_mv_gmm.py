import numpy as np

from .mv_models import *
from .models import Dirichlet

# multivariate Bayesian GMM implementing collapsed Gibbs sampling

class CGSBayesianMVGMM(object):

    def __init__(self, alpha, m0, k0, v0, S0):
        alpha=np.array(alpha)
        m0=np.array(m0)
        self.dir0 = Dirichlet(alpha)
        self.K = len(alpha) # number of mixture components
        self.niw0 = NormalInvWishart(m0, k0, v0, S0)

        weights = self.dir0.sample()
        means = []
        covs = []
        for k in range(self.K):
            m, c = self.niw0.sample()
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
        N = len(X)
        # keep a record of historic component assignments
        record=np.empty((niters+1, N))  # rows are Zs; 1 extra row for initial Z
        record[0, :] = Z

        K=self.K    # number of GMM components
        # compute statistics for each cluster
        I=np.array([np.where(Z==k) for k in range(K)])[:,0]    # inverse indices
        #, i.e. lists of data points indices that belong to each mixture

        Counts=np.array([len(i) for i in I])    # counts of data points for each mixture
        compIds = np.arange(0, K)         # list of component ids

        for t in range(niters):
            for i in range(N):
                x = X[i, :]   # ith data entry, X_i
                z = Z[i]    # mixture/component id of x
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
                logLZ_i = np.ones(K)*np.log(self.dir0.alphas + Counts)

                # compute conditional log probability of x (i.e. X_i),
                # conditioned on Z_i being assigned to each of K components)
                # [24.27, P.843, Murphy text]
                logPx = np.empty(K)
                for k in range(K):
                    # this is the posterior predictive, with evidence being all
                    # the data assigned to cluster k except for X_i
                    logPx[k] = \
                        self.niw0.posterior(X[I[k], :]).predictiveDensity().logpdf(x)

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

                # record the new assignment for X_i (0th row for initial Z)
                record[t+1, i] = knew

        return record


#    def averageGMM(self):
#        avg_means = self.sumMeans / self.count_mv
#        avg_variances = self.sumVariances / self.count_mv
#        avg_weights = self.sumWeights /  self.count_w
#        return GMM(avg_means, avg_variances, avg_weights)

