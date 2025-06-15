# Implementation of the multiplicative update rule from Lee and Seung

# The implementation of sklearn uses the mean of the matrix to initialize H and W so that is
# is in the same order of magnitude

import random
import numpy as np
from typing import Tuple
import time
import matplotlib.pyplot as plt
from sklearn import decomposition


def euclidian_distance(A: np.ndarray, B: np.ndarray) -> float:
    return np.sum((A - B) ** 2)


def nmf(
    V: np.ndarray, r: int, iter_max: int, tolerance: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Take a matrix V of size n by m and r\n
    Return W of size n by r and H of size r by m s.t V=~WH\n
    Stop when convergence is reached fot the given tolerance or when iter_max is reached
    """

    # Initialize the W and H
    (n, m) = V.shape
    W = np.random.rand(n, r) * 10
    H = np.random.rand(r, m) * 10

    # We apply the update rule on H and V
    # @ is matrix multiplication, * and / are element wise operations

    previous_distance = euclidian_distance(V, W @ H)
    iter = 0

    while iter < iter_max:
        H = H * ((W.transpose() @ V) / (W.transpose() @ W @ H))
        W = W * ((V @ H.transpose()) / (W @ H @ H.transpose()))

        distance = euclidian_distance(V, W @ H)

        # print(distance)

        if previous_distance - distance < tolerance:
            break

        previous_distance = distance
        iter += 1

    return (W, H)


def nmf_test(
    nb_tests: int, iter_max: int, tolerance: float
) -> Tuple[list, list, list, list]:
    # test on random matrix

    time_results_implem = []
    time_results_sklearn = []

    distance_results_implem = []
    distance_results_sklearn = []

    for iter in range(1, nb_tests):

        V = np.random.rand(iter, iter) * iter
        (n, m) = V.shape

        r = random.randint(1, min(n, m))

        start = time.time()
        W, H = nmf(V=V, r=r, iter_max=iter_max, tolerance=tolerance)
        end = time.time()
        time_results_implem.append((end - start))
        distance_results_implem.append(euclidian_distance(V, W @ H))

        model = decomposition.NMF(n_components=r, init="random", random_state=0)

        start = time.time()
        W = model.fit_transform(V)
        H = model.components_
        end = time.time()
        time_results_sklearn.append((end - start))
        distance_results_sklearn.append(euclidian_distance(V, W @ H))

    return (
        time_results_implem,
        time_results_sklearn,
        distance_results_implem,
        distance_results_sklearn,
    )


nb_tests = 100

(
    time_results_implem,
    time_results_sklearn,
    distance_results_implem,
    distance_results_sklearn,
) = nmf_test(nb_tests, 200, 1e-6)

x = range(1, nb_tests)

fig, ax = plt.subplots()
ax.set_ylabel("time")
ax.set_xlabel("size of V")
ax.set_title("NMF computation time vs matrix size")
ax.plot(x, time_results_implem, label="implem")
ax.plot(x, time_results_sklearn, label="sklearn")
plt.legend()
plt.savefig("../results/time_results.png")

fig, ax = plt.subplots()
ax.set_ylabel("euclidean distance")
ax.set_xlabel("size of V")
ax.set_title("NMF euclidean distance vs matrix size")
ax.plot(x, distance_results_implem, label="implem")
ax.plot(x, distance_results_sklearn, label="sklearn")
plt.legend()
plt.savefig("../results/distance_results.png")
