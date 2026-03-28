#%%


import os
import pickle
import numpy as np 
import matplotlib.pyplot as plt 

PCA_PATH = "/Volumes/Maxime Imperial Backup/DRR2-HUmap/_3.PCN-CNN/models_V2/pca.pkl"
pca = pickle.load(open(PCA_PATH, "rb"))

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
individual_variance = pca.explained_variance_ratio_
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.plot(np.arange(1, len(cumulative_variance)+1), cumulative_variance*100, marker='o')
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Variance Explained (%)')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(np.arange(1, len(individual_variance)+1), individual_variance*100, marker='o')
plt.axhline(y=0.01, color='r', linestyle="--", label="1% variance threshold")
plt.xlabel('Number of PCA Components')
plt.ylabel('Individual Variance Explained (%)')
plt.legend()
plt.grid(True)
plt.show()
# %%
