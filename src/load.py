import numpy as np
# Load the .npz file
data = np.load('./data/processed/features.npz')

print(data["X_train"])