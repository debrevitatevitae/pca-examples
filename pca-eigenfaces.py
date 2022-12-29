import os
import sys
from typing import Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

from utils import compute_pca, plot_singular_values


def get_lfw_data(min_faces_pp:int,) -> Tuple[int, int, int, np.ndarray]:
	"""Downloads face images from the 'Labeled Faces in the Wild' dataset and returns dataset dimensions and data.

	Args:
		min_faces_pp (int): get the people with only so many face images available.

	Returns:
		Tuple[int, int, int, np.ndarray]: shape and data of the dataset. In order: number of samples, height, width of the image and data matrix (each row is a face).
	"""
	lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_pp, resize=0.4)
	n_samples, h, w = lfw_people.images.shape
	print(f"Number of faces: {n_samples:d}\n")
	print(f"Images heigth: {h:d} pixels\n")
	print(f"Images width: {w:d} pixels\n")
	return n_samples, h, w, lfw_people.data

def plot_faces(X:np.ndarray, h:int, w:int, n_rows:int=5, n_cols:int=5, plot_title="Some faces of the LFW dataset.") -> None:
	fig, axs = plt.subplots(n_rows, n_cols)
	for i in range(n_rows):
		for j in range(n_cols):
			face_id = np.random.randint(low=0, high=len(X))
			axs[i, j].imshow(X[face_id].reshape((h, w)), cmap=plt.cm.gray)
			axs[i, j].set_xticks(())
			axs[i, j].set_yticks(())
	fig.suptitle(plot_title)
	plt.show()

def plot_eigenfaces(U:np.ndarray, h:int, w:int, n_rows:int=5, n_cols:int=5) -> None:
	UT = U.T
	fig, axs = plt.subplots(n_rows, n_cols)
	for i in range(n_rows):
		for j in range(n_cols):
			axs[i, j].imshow(UT[i*n_cols+j].reshape((h, w)), cmap=plt.cm.gray)
			axs[i, j].set_xticks(())
			axs[i, j].set_yticks(())
	fig.suptitle(f"First {n_rows*n_cols:d} eigenfaces of LFW.")
	plt.show()

def reconstruct_face(x:np.ndarray, U:np.ndarray, deg:int=1) -> np.ndarray:
	U_trunc = U[:, :deg]
	x_rec = U_trunc @ U_trunc.T @ x
	return x_rec

def plot_reconstructed_face(x:np.ndarray, U:np.ndarray, h:int, w:int) -> None:
	degs = iter([5, 50, 200, 600, 1000])
	fig, axs = plt.subplots(2, 3)
	axs[0, 0].imshow(x.reshape((h, w)), cmap=plt.cm.gray)
	axs[0, 0].set_xticks(())
	axs[0, 0].set_yticks(())
	axs[0, 0].set_title("original")
	for i in range(2):
		j_range = range(1, 3) if i == 0 else range(3)
		for j in j_range:
			d = next(degs)
			x_r = reconstruct_face(x, U, deg=d)
			axs[i, j].imshow(x_r.reshape((h, w)), cmap=plt.cm.gray)
			axs[i, j].set_xticks(())
			axs[i, j].set_yticks(())
			axs[i, j].set_title(f"r={d:d}")
	fig.suptitle("Original and reconstructed face for increasing r")
	plt.show()

if __name__ == '__main__':
	np.random.seed(0)
	
	#%% Obtain data, split in train/test and plot some faces
	n_samples, h, w, X = get_lfw_data(min_faces_pp=70)
	X_train, X_test = train_test_split(X, test_size=.25, random_state=0)
	plot_faces(X_train, h, w)

	#%% Compute the PCA, plot singular values and the first 6 eigenfaces
	U, s, VT = compute_pca(X_train)
	plot_singular_values(s)
	plot_eigenfaces(U, h, w, n_rows=2, n_cols=3)

	#%% Reconstruct a test face from the eigenfaces
	x_i = X_test[np.random.randint(0, len(X_test))]
	plot_reconstructed_face(x_i, U, h, w)
	