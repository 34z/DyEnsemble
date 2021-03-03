from filterpy.monte_carlo import systematic_resample
from DyEnsembleModel import *


class DyEnsemble:
    def __init__(self, x_dim, z_dim, t_model, *m_models):
        """
        params:
            x_dim: dimension of state
            z_dim: dimension of measurement
            t_model: transition model(function)
            m_models: measurement models(functions)
        """
        self.x_dim = x_dim
        self.z_dim = z_dim

        # assert isinstance(t_model, DyEnsembleTModel)
        assert t_model.x_dim == self.x_dim
        self.t_model = t_model
        self.m_models = []
        for model in m_models:
            # assert isinstance(model, DyEnsembleMModel)
            assert model.z_dim == self.z_dim
            self.m_models.append(model)
        self.m_model_num = len(self.m_models)

    def __reset_params(self, n_particle, x_idle):
        self.n_particle = n_particle
        self.particles = np.zeros((self.n_particle, self.x_dim))
        if x_idle is None:
            x_idle = np.zeros(self.x_dim)
        for i in range(self.n_particle):
            self.particles[i] = x_idle
        self.particles += scipy.stats.norm(0, 0.001).rvs([self.n_particle, 1])
        self.pm = np.ones(self.m_model_num) / self.m_model_num
        self.weights = np.ones(self.n_particle) / self.n_particle
        self.resampling_cnt = 0
        self.init_predict = True

    def __call__(self, z, n_particle=200, x_idle=None, alpha=0.9, save_pm=False, refresh=True):
        return self.predict(z, n_particle=n_particle, x_idle=x_idle, alpha=alpha, save_pm=save_pm, refresh=refresh)

    def predict(self, z, n_particle=200, x_idle=None, alpha=0.9, save_pm=False, refresh=True):
        if refresh or (not self.init_predict):
            self.__reset_params(n_particle, x_idle)
        self.alpha = alpha
        self.save_pm = save_pm
        if z.ndim < 2:
            z = z[np.newaxis, :]
        dataset_size = z.shape[0]
        x_hat = np.zeros((dataset_size, self.x_dim))
        if self.save_pm:
            self.pms = np.zeros((dataset_size, self.m_model_num))
        for i in range(dataset_size):
            x_hat[i] = self.__predict_once(z[i].reshape(1, self.z_dim))
            if self.save_pm:
                self.pms[i] = self.pm
        return x_hat

    def __predict_once(self, z):
        # predict prior
        self.__predict_prior()
        # update weights
        self.__update(z)
        # resampling
        if self.__neff(self.weights) < self.n_particle / 2:
            self.resampling_cnt += 1
            self.__resample()
        x_hat, _ = self.__estimate()
        return x_hat

    def __predict_prior(self):
        self.particles = self.t_model(self.particles)

    def __update(self, z):
        self.pm = self.pm ** self.alpha
        self.pm /= np.sum(self.pm)
        probs = np.zeros((self.m_model_num, self.n_particle))
        for i in range(self.m_model_num):
            z_ = self.m_models[i](self.particles)
            probs[i] = self.m_models[i].prob(z, z_)
            # print(probs[i].shape, np.max(probs[i]))
        # print(probs.shape)
        final_prob = np.sum(probs * self.weights, axis=1)
        final_prob /= np.sum(final_prob)
        # print(final_prob)
        # exit()
        self.pm *= final_prob
        if np.sum(self.pm) < 1e-300:
            self.pm[self.pm==0] = 1e-300
        self.pm /= np.sum(self.pm)
        self.weights *= np.average(probs, axis=0, weights=self.pm)
        self.weights /= np.sum(self.weights)

    def __estimate(self):
        mean = np.average(self.particles, weights=self.weights, axis=0)
        var = np.average((self.particles - mean) ** 2, weights=self.weights, axis=0)
        return mean, var

    def __resample(self):
        indexes = systematic_resample(self.weights)
        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]
        self.weights.fill(1.0 / len(self.weights))

    def __neff(self, weights):
        return 1. / np.sum(np.square(weights))

    def train(self, x, z, x_val=None, z_val=None):
        self.t_model.train(x)
        for model in self.m_models:
            model.train(x, z, x_val, z_val)
        self.init_predict = False