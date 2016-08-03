import numpy as np
from scipy.misc import logsumexp
from scipy.special import gamma, gammaln, psi

# probabilistic models commonly used in Bayesian GMMs


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
        self.alphas = np.array(alphas)

    def sample(self):
        return np.random.dirichlet(self.alphas)

    def posterior(self, Nk):
        return Dirichlet(self.alphas + Nk)

    def expLogPi(self):
        return psi(self.alphas) - psi(self.alphas.sum())

    def KL(self, pdf):
        E_log_weights = self.expLogPi()
        dirichlet_KL = gammaln(self.alphas.sum())
        dirichlet_KL -= gammaln(pdf.alphas.sum())
        dirichlet_KL -= gammaln(self.alphas).sum()
        dirichlet_KL += gammaln(pdf.alphas).sum()
        dirichlet_KL += (E_log_weights*(self.alphas - pdf.alphas)).sum()
        return dirichlet_KL


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

    def expLogPrecision(self):
        return psi(self.a) - np.log(self.b)

    def expPrecision(self):
        return self.a/self.b

    def posterior(self, Nk, xk_bar, Sk):
        m = (self.kappa * self.mean + Nk * xk_bar)/(self.kappa + Nk)
        kappa = self.kappa + Nk
        a = self.a + .5* Nk
        b = self.b + 0.5 * Nk * Sk
        b += (self.kappa* Nk *((xk_bar - self.mean)**2)) / (2*(self.kappa + Nk))
        return NormalGamma(m, kappa, a, b)

    def predictiveDensity(self):
        nu = 2*self.a
        gamma = (self.a*self.kappa) /(self.b*(self.kappa + 1))
        return StudentT(self.mean, nu, gamma)

    def KL(p, q):
        exp_lambda = p.expPrecision()
        exp_log_lambda = p.expLogPrecision()
        return (.5 * (np.log(p.kappa) - np.log(q.kappa))
                - .5 * (1 - q.kappa * (1./p.kappa + exp_lambda * (p.mean - q.mean)**2))
                - (gammaln(p.a) - gammaln(q.a))
                + (p.a * np.log(p.b) - q.a * np.log(q.b))
                + exp_log_lambda * (p.a - q.a)
                - exp_lambda * (p.b - q.b))


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


