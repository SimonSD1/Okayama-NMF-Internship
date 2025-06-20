# Implementation of the BANMF algorithm from TRUONG, SKAU, DESANTIS, ALEXANDROV


import random
import numpy as np
from typing import Optional, Tuple
import time
import matplotlib.pyplot as plt
from sklearn import decomposition
from mpl_toolkits import mplot3d

epsilon = 1e-10


def euclidian_distance(A: np.ndarray, B: np.ndarray) -> float:
    return np.sum((A - B) ** 2)


def boolean_distance(A: np.ndarray, B: np.ndarray) -> float:
    return np.sum(A != B)


def banmf_initialization(
    X: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    (n, m) = X.shape

    # Initialization
    Y = X.copy().astype(float)
    print(m, k)
    W = np.random.rand(n, k) * 100
    H = np.random.rand(k, m) * 100

    return Y, W, H


def banmf_auxiliary_solve(
    X: np.ndarray, Y: np.ndarray, W: np.ndarray, H: np.ndarray, k: int, Niter: int
) -> Tuple[np.ndarray, np.ndarray]:

    for _ in range(Niter):
        W = W * ((Y @ H.transpose()) / (W @ H @ H.transpose() + epsilon))
        H = H * ((W.transpose() @ Y) / (W.transpose() @ W @ H + epsilon))

        current_result = W @ H
        Y[X] = np.clip(current_result[X], 1, k)

    return W, H


def booleanization(
    X: np.ndarray, W: np.ndarray, H: np.ndarray, npoints: int
) -> Tuple[np.ndarray, np.ndarray]:

    W_p = np.linspace(np.min(W), np.max(W), npoints)
    H_p = np.linspace(np.min(H), np.max(H), npoints)

    argmin_W, argmin_H, _ = min(
        (
            (
                delta_W,
                delta_H,
                boolean_distance(X, ((W > delta_W) @ (H > delta_H)).astype(bool)),
            )
            for delta_W in W_p
            for delta_H in H_p
        ),
        key=lambda t: t[2],
    )

    W_prime = W > argmin_W
    H_prime = H > argmin_H

    return W_prime, H_prime


def banmf(X: np.ndarray, k: int, Niter: int) -> Tuple[np.ndarray, np.ndarray]:

    Y, W, H = banmf_initialization(X, k)

    W, H = banmf_auxiliary_solve(X, Y, W, H, k, Niter)

    return booleanization(X, W, H, 10)


def test_nb_point_booleanization(n: int, m: int, nb_tests: int, plot: bool, k: Optional[int] = None):
    X = (np.random.rand(n, m) > 0.5).astype(bool)

    if k is None:
        k = random.randint(1, min(n, m))

    Y, W, H = banmf_initialization(X, k)

    W, H = banmf_auxiliary_solve(X, Y, W, H, k, 200)

    booleanization_result = []
    for npoints in range(1, nb_tests):
        W_prime, H_prime = booleanization(X, W, H, npoints)
        booleanization_result.append(boolean_distance(X, W_prime @ H_prime))

    if plot == True:
        x = range(1, nb_tests)

        fig, ax = plt.subplots()
        ax.set_ylabel("final distance")
        ax.set_xlabel("number of points")
        ax.set_title(
            f"Final distance vs number of point in booleanization on a {n} by {m}"
        )
        ax.plot(x, booleanization_result, label="final distance")
        plt.legend()
        plt.savefig("../results/npoints_booleanization.png")

    return booleanization_result


def test_nb_points_3d(nb_tests, plot: bool):
    # for different matrix sizes
    result_matrix = []

    for size in range(1, nb_tests):
        bool_result = test_nb_point_booleanization(size, size, nb_tests, plot=False, k=max(1,int(size/2)))
        result_matrix.append(
            bool_result
        )  # chaque ligne correspond à une taille de matrice

    result_matrix = np.array(result_matrix)

    if plot:
        #plt.figure(figsize=(10, 8))
        plt.imshow(
            result_matrix,
            origin="lower",
            extent=(1.0, float(nb_tests - 1), 1.0, float(nb_tests - 1)),
        )
        plt.colorbar(label="final boolean distance")
        plt.xlabel("nb points in booleanization")
        plt.ylabel("Size of the matrix")
        plt.title("nb points and size of matrix vs boolean distance (k=size/2)")
        plt.savefig("../results/3d_booleanization_heatmap.png")

def test_latent_dimension(n: int, m: int, nb_tests: int):

    X = (np.random.rand(n, m) > 0.5).astype(bool)

    results_distance = []
    for k in range(1, nb_tests):
        W, H = banmf(X, k, 200)
        results_distance.append(boolean_distance(X, W @ H))

    x = range(1, nb_tests)

    fig, ax = plt.subplots()
    ax.set_ylabel("final distance")
    ax.set_xlabel("latent dimension")
    ax.set_title(f"final distance vs latent dimension for {n} by {m} matrix")
    ax.plot(x, results_distance, label="final distance")
    plt.legend()
    plt.savefig("../results/banmf_latent_dimension.png")


# test_nb_point_booleanization(50,50,100)
#test_latent_dimension(50, 50, 100)
test_nb_points_3d(50,True)