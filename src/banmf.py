# Implementation of the BANMF algorithm from TRUONG, SKAU, DESANTIS, ALEXANDROV


import random
import numpy as np
from typing import Tuple
import time
import matplotlib.pyplot as plt
from sklearn import decomposition

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
    print(m,k)
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

def test_nb_point_booleanization(n:int,m:int,nb_tests:int):
    X=(np.random.rand(n,m)>0.5).astype(bool)

    k=random.randint(1,min(n,m))

    Y,W,H=banmf_initialization(X,k)

    W,H=banmf_auxiliary_solve(X,Y,W,H,k,200)

    booleanization_result=[]
    for npoints in range (1,nb_tests):
        W_prime,H_prime=booleanization(X,W,H,npoints)
        booleanization_result.append(boolean_distance(X,W_prime@H_prime))
    

    x = range(1, nb_tests)

    fig, ax = plt.subplots()
    ax.set_ylabel("final distance")
    ax.set_xlabel("number of points")
    ax.set_title("Final distance vs number of point in booleanization")
    ax.plot(x, booleanization_result, label="final distance")
    plt.legend()
    plt.savefig("../results/npoints_booleanization.png")
    


test_nb_point_booleanization(50,50,100)