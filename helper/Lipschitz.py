import numpy as np

def calculate_lipschitz_distribution(objective):
    A = objective.A

    L = list()

    for a_i in A:
        #eigen_values = np.linalg.eig(np.outer(a_i, a_i))[0]
        #eig_max = max(eigen_values)
        L_i = np.linalg.norm(a_i)**2
        L.append(L_i)

    L = np.array(L)

    return L / np.sum(L)