import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import matplotlib.pyplot as plt

from utils import compute_pca


if __name__ == '__main__':
	np.random.seed(0)
	
	#%% Obtain data and split in train/test
	

	#%% Compute the PCA
	

	#%% Reconstruct a test face from the eigenfaces
	