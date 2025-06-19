# Implementation of the BANMF algorithm from TRUONG, SKAU, DESANTIS, ALEXANDROV


import random
import numpy as np
from typing import Tuple
import time
import matplotlib.pyplot as plt
from sklearn import decomposition

epsilon = 1e-10


def banmf(X: np.ndarray, k: int, Niter: int) -> Tuple[Tuple[np.ndarray, np.ndarray],list]:

    (n, m) = X.shape

    # Initialization
    Y = X.copy().astype(float)
    W = np.random.rand(n, k) * 100
    H = np.random.rand(k, m) * 100

    # Solving the auxiliary problem
    iter = 0

    distances = []
    while iter < Niter:
        W = W * ((Y @ H.transpose()) / (W @ H @ H.transpose() + epsilon))
        H = H * ((W.transpose() @ Y) / (W.transpose() @ W @ H + epsilon))

        current_result = W @ H

        Y[(current_result < 1) & X] = 1
        Y[(current_result >= 1) & (current_result <= k) & X] = current_result[
            (current_result >= 1) & (current_result <= k) & X
        ]
        Y[(current_result >= k) & X] = k
        iter += 1

        distances.append(euclidian_distance(Y, W @ H))

    print(distances)
    print("ok")
    # Booleanization
    return booleanization(X, W, H, 10), distances


def booleanization(
    X: np.ndarray, W: np.ndarray, H: np.ndarray, npoints: int
) -> Tuple[np.ndarray, np.ndarray]:

    W_p = np.linspace(np.min(W), np.max(W), npoints)
    H_p = np.linspace(np.min(H), np.max(H), npoints)
    (n, m) = X.shape

    min_distance = n * m + 1
    argmin_W = 0
    argmin_H = 0
    distances_booleanization = []

    for delta_W in W_p:
        for delta_H in H_p:
            # apply threshold

            W_prime = W > delta_W
            H_prime = H > delta_H

            # compute distance

            current_distance = boolean_distance(X, (W_prime @ H_prime).astype(bool))
            if current_distance < min_distance:
                min_distance = current_distance
                argmin_W = delta_W
                argmin_H = delta_H

    W_prime = W > argmin_W
    H_prime = H > argmin_H

    return W_prime, H_prime


def euclidian_distance(A: np.ndarray, B: np.ndarray) -> float:
    return np.sum((A - B) ** 2)


def boolean_distance(A: np.ndarray, B: np.ndarray) -> float:
    return np.sum(A != B)


def banmf_test(nb_tests: int, iter_max: int) -> Tuple[list,list]:
    # test on random matrix

    distance_results_implem = []

    convergence=[]
    for iter in range(0, nb_tests):

        X = (np.random.rand(100, 100) * 10 > 0.5).astype(bool)

        k = random.randint(1, 100)

        ((W, H),convergence) = banmf(X, k, iter_max)
        distance_results_implem.append(boolean_distance(X, W @ H))

    return distance_results_implem, convergence


nb_tests = 1
iter_max=200
distance_result, convergence = banmf_test(nb_tests, iter_max)




x = range(0, iter_max)

fig, ax = plt.subplots()
ax.set_ylabel("distance")
ax.set_xlabel("size of X")
ax.set_title("Convergence")
ax.plot(x, convergence, label="boolean distance")
plt.legend()
plt.savefig("../results/banmf_convergence.png")
