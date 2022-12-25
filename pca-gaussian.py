import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def generate_2d_gaussian(n:int, R:np.ndarray, S:np.ndarray, x_c:np.ndarray) -> np.ndarray:
	X_g = np.random.randn(n, 2)
	return X_g @ np.diag(S) @ R + x_c


if __name__ == '__main__':
	np.random.seed(0)

	R = np.array([
		[np.cos(np.pi/3), np.sin(np.pi/3)],
		[-np.sin(np.pi/3), np.cos(np.pi/3)]
	])
	S = np.array([2., 0.5])
	x_c = np.array([2., 1.])

	data = generate_2d_gaussian(1000, R, S, x_c)

	fig, ax = plt.subplots()
	ax.scatter(data[:,0], data[:,1], s=6, c='k', alpha=.7)
	ax.set(xlabel='x1', ylabel='x2', title='Guassian points streched, rotated and de-centered')
	ax.grid()
	plt.show()