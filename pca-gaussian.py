import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import compute_pca


def generate_2d_gaussian(n:int, R:np.ndarray, S:np.ndarray, x_c:np.ndarray) -> np.ndarray:
	"""Generate a gaussian distribution in 2 dimension, stretch it, rotate it and decenter it

	Args:
		n (int): number of samples
		R (np.ndarray): Rotation matrix. Shape = (2, 2)
		S (np.ndarray): Standard devs. Shape = (2,)
		x_c (np.ndarray): New center. Shape = (2,)

	Returns:
		np.ndarray: gaussian data. Shape = (n, m)
	"""
	X_g = np.random.randn(n, 2)
	return X_g @ np.diag(S) @ R + x_c


def compute_confidence_interval(U:np.ndarray, S:np.ndarray) -> np.ndarray:
	"""Returns the confidence interval of the data as an ellipsoid, centered around (0, 0).

	Args:
		U (np.ndarray): principal components matrix. Shape = (2, 2)
		S (np.ndarray): principal values. Shape = (2, 2)

	Returns:
		np.ndarray: confidence interval array. Shape = (100, 2)
	"""
	thetas = np.arange(0., 1., .01) * 2 * np.pi
	return np.array([np.cos(thetas), np.sin(thetas)]).T @ np.diag(S) @ U.T


if __name__ == '__main__':
	np.random.seed(0)
	
	#%% Generate data
	R = np.array([
		[np.cos(np.pi/3), np.sin(np.pi/3)],
		[-np.sin(np.pi/3), np.cos(np.pi/3)]
	])
	S = np.array([2., 0.5])
	x_c = np.array([2., 1.])

	data = generate_2d_gaussian(1000, R, S, x_c)

	fig, ax = plt.subplots()
	ax.scatter(data[:,0], data[:,1], s=6, c='k', alpha=.7)
	ax.set(xlabel='x1', ylabel='x2', title='Gaussian points streched, rotated and de-centered')
	ax.grid()
	# plt.show()

	#%% Compute the PCA
	U_pca, S_pca, VT_pca = compute_pca(data)
	# The principal components should match the rotation that we used to generate the data (up to a sign)
	print(U_pca)
	print(R.T)

	#%% Plot the confidence intervals
	x_avg = np.mean(data, axis=0)
	x_ci = compute_confidence_interval(U_pca, S_pca)

	fig, ax = plt.subplots()
	# plot data
	ax.scatter(data[:,0], data[:,1], s=6, c='k', alpha=.7)
	# plot the principal components
	ax.plot(np.array([0., U_pca[0, 0]]) * S_pca[0] + x_avg[0], np.array([0., U_pca[1, 0]]) * S_pca[0] + x_avg[1], 'c')
	ax.plot(np.array([0., U_pca[0, 1]]) * S_pca[1] + x_avg[0], np.array([0., U_pca[1, 1]]) * S_pca[1] + x_avg[1], 'c')
	# plot confidence intervals
	ax.plot(x_avg[0] + x_ci[:, 0], x_avg[1] + x_ci[:, 1], 'b-')
	ax.plot(x_avg[0] + 2*x_ci[:, 0], x_avg[1] + 2*x_ci[:, 1], 'b-')
	ax.plot(x_avg[0] + 3*x_ci[:, 0], x_avg[1] + 3*x_ci[:, 1], 'b-')
	ax.set(xlabel='x1', ylabel='x2', title="Principal components of gaussian data and confidence interval")
	plt.show()