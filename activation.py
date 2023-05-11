import numpy as np

m = np.array([ [ 9.6, 3.3, 5.6, 7.2, 5.6, 5.4, 1.9, 8.2, 10. , 7.1],
               [ 1. , 4.7, 0.8, 9.1, 7.6, 2.1, 4.8, 2.5, 7.4, 3.8],
               [ 6.6, 7. , 0.6, 2. , 4. , 9.1, 7.9, 6.9, 9.7, 3.6],
               [ 9.1, 5.9, 8.4, 7.9, 7.8, 6.8, 6.3, 0.9, 1.4, 5.3] ])

epsilon = 1e-5
gamma = 4
beta = 0.3
n = np.tanh(m)   #data is activated
# calculate the mean and variance of each row
row_means = np.mean(n, axis=1)
row_variances = np.var(n, axis=1)
print("BATCH NORMALISATION")


# perform mean normalization and variance scaling for each row
for i in range(n.shape[0]):
    
    n_norm = gamma*((n[i] - row_means[i]) / np.sqrt(row_variances[i] + epsilon)) + beta
    print(" Normalized row", i, ":", n_norm)
    print(" ")
    
print(" ")
# calculate the mean and variance of each column
col_means = np.mean(n, axis=0)
col_variances = np.var(n, axis=0)

print("LAYER NORMALISATION")
# perform mean normalization and variance scaling for each column
for j in range(n.shape[0]):
    n__norm = gamma*((n[j] - col_means[j]) / np.sqrt(col_variances[j] + epsilon)) + beta
    print(" Normalized row", j, ":", n__norm)
    print(" ")
