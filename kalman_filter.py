import numpy as np


class KalmanFilter:

	def __init__(self, x_dim, z_dim):
		self.trained = False
		self.x_dim = x_dim
		self.z_dim = z_dim

	def __call__(self, x_idle, z):
		return self.fit(x_idle, z)

	def fit(self, x_idle, z):
		if not self.trained:
			raise("Please train first!")
		# P_idle = np.random.normal(0, 0.1, size=(self.x_dim, self.x_dim))
		P_idle = np.identity(self.x_dim)

		n_iter = z.shape[0]
		z = z.T
		x_hat = np.zeros((self.x_dim, n_iter))
		x_hat[:, 0] = x_idle
		P = P_idle

		for k in range(1, n_iter):
			# prediction
			x_hat_minus = self.A @ x_hat[:, k - 1]
			P_minus = self.A @ P @ self.A.T + self.Q
			# correction
			K = P_minus @ self.H.T @ np.linalg.pinv(self.H @ P_minus @ self.H.T + self.R)
			x_hat[:, k] = x_hat_minus + K @ (z[:, k] - self.H @ x_hat_minus)
			P = (np.identity(self.x_dim) - K @ self.H) @ P_minus
		return x_hat.T

	def train(self, x, z):
		k = x.shape[0]
		x, z = x.T, z.T
		x0 = x[:, :-1]
		x1 = x[:, 1:]
		self.A = (x1 @ x0.T) @ np.linalg.inv(x0 @ x0.T)
		# print(self.A)
		self.Q = (x1 - self.A @ x0) @ (x1 - self.A @ x0).T / (k - 1)
		self.H = (z @ x.T) @ np.linalg.inv(x @ x.T)
		z_ = self.H @ x
		self.R = (z - z_) @ (z - z_).T / k
		cc2 = CC2(z, z_)
		# print(cc2)
		self.trained = True
		return self.A, self.Q, self.H, self.R

def CC2(y, fx):
	k, n = y.shape
	a = y - y.mean(axis=1)[:, np.newaxis]
	b = fx - fx.mean(axis=1)[:, np.newaxis]
	cc = np.sum(a * b, axis=1)**2 / (np.sum(a ** 2, axis=1) * np.sum(b ** 2, axis=1))
	return cc