import numpy as np

def euclidean_distance(W_S, W_R):
    return np.sum((W_S - W_R)**2)

def min_distance(W_S, I_i):
    l = len(W_S)
    distances = [euclidean_distance(W_S, I_i[i:i+l]) for i in range(len(I_i) - l + 1)]
    return min(distances)

def compute_distance_list(S_K, I):
    D_k = [min_distance(S_K, I_i) for I_i in I]
    return D_k

# Example usage:
S_K = np.array([1, 2, 3])  # Candidate behavior unit
I = [np.array([1, 2, 3, 4, 5]), np.array([3, 4, 5, 6, 7])]  # Behavior sequences

D_k = compute_distance_list(S_K, I)
print("Distance list:", D_k)
