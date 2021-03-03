import numpy as np
import scipy
import scipy.stats
import scipy.spatial
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


def rndn(mu, cov, size):
    ret = scipy.stats.multivariate_normal(mu, cov, allow_singular=True).rvs(size)
    if ret.ndim == 1:
        ret = ret[:, np.newaxis]
    return ret


def cal_cov(x) -> np.ndarray:
    return np.cov(x.T)


class DyEnsembleTModel:
    def __init__(self, x_dim, rnd=None):
        self.x_dim = x_dim
        self.rnd = rnd
        if self.rnd is None:
            self.rnd = 'Normal'
            self.rnd_cov = None

    def train(self, x):
        pass

    def __call__(self, x, use_rnd=True):
        ret = self.transit(x)
        if use_rnd:
            size = x.shape[0] if isinstance(x, np.ndarray) else 1
            ret += self.random(size)
        return ret

    def transit(self, x):
        raise NotImplementedError

    def next_step(self):
        pass

    def random(self, size):
        if self.rnd == 'Normal':
            if self.rnd_cov is None:
                self.rnd_cov = np.identity(self.x_dim)
            ret = rndn(np.zeros(self.x_dim), self.rnd_cov, size)
        else:
            raise Exception("Random '{}' is not supported.".format(self.rnd))
        return ret


class DyEnsembleMModel:
    def __init__(self, z_dim, metric=None):
        self.z_dim = z_dim
        self.metric = metric
        if self.metric is None:
            self.metric = 'M-dis'
        if self.metric == 'Normal':
            self.cov = None
        elif self.metric == 'M-dis':
            self.cov = None

    def train(self, x, z, x_val=None, z_val=None):
        pass

    def __call__(self, x):
        ret = self.measure(x)
        return ret

    def measure(self, x):
        raise NotImplementedError

    def prob(self, z, z_):
        if self.metric == 'Normal':
            if self.cov is None:
                self.cov = np.identity(self.z_dim)
            prob = scipy.stats.multivariate_normal(np.squeeze(z), self.cov, allow_singular=True).pdf(z_)
        elif self.metric == 'L2':
            dis = np.sqrt(np.sum((z-z_) ** 2, axis=1))
            prob = np.exp(-0.5 * dis)
        elif self.metric == 'M-dis':
            if self.cov is None:
                self.cov = np.identity(self.z_dim)
            covi = np.linalg.inv(self.cov)
            dis = np.array([scipy.spatial.distance.mahalanobis(z_[i], z, covi) for i in range(z_.shape[0])])
            prob = np.exp(-0.5 * dis)
        else:
            raise Exception("Metric '{}' is not supported.".format(self.metric))
        return prob


class DyEnsembleTLinear(DyEnsembleTModel):
    def __init__(self, x_dim):
        super().__init__(x_dim)
        self.clf = linear_model.LinearRegression()

    def train(self, x):
        x0 = x[:-1]
        x1 = x[1:]
        self.clf.fit(x0, x1)
        err = x1 - self.clf.predict(x0)
        self.W = np.cov(err.T)

    def transit(self, x):
        x_prior = self.clf.predict(x)
        return x_prior

    def random(self, size):
        W = self.W
        w = rndn(np.zeros(self.x_dim), W, size)
        # w = scipy.stats.multivariate_normal(np.zeros(self.x_dim), W, allow_singular=True).rvs(size)
        return w


class DyEnsembleMLinear(DyEnsembleMModel):
    def __init__(self, z_dim):
        super().__init__(z_dim)
        self.clf = linear_model.LinearRegression()

    def train(self, x, z, x_val=None, z_val=None):
        self.clf.fit(x, z)
        err = z - self.clf.predict(x)
        self.cov = cal_cov(err)

    def measure(self, x):
        z = self.clf.predict(x)
        return z


class DyEnsembleMPoly(DyEnsembleMModel):
    def __init__(self, z_dim, degree):
        super().__init__(z_dim)
        self.degree = degree
        self.poly = PolynomialFeatures(degree=self.degree)

    def train(self, x, z, x_val=None, z_val=None):
        x_ = self.poly.fit_transform(x)
        self.clf = linear_model.LinearRegression()
        self.clf.fit(x_, z)
        err = z - self.clf.predict(x_)
        self.cov = cal_cov(err)

    def measure(self, x):
        x_ = self.poly.fit_transform(x)
        return self.clf.predict(x_)
