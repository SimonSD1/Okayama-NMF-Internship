# Implementation of the BANMF algorithm from TRUONG, SKAU, DESANTIS, ALEXANDROV


import random
import numpy as np
from typing import Tuple
import time
import matplotlib.pyplot as plt
from sklearn import decomposition

epsilon = 1e-10

def banmf(X: np.ndarray, k: int, Niter: int):

    (n, m) = X.shape

    # Initialization
    Y = X
    W = np.random.rand(n, k)*10
    H = np.random.rand(k, m)*10

    # Solving the auxiliary problem
    iter = 0

    while iter < Niter:
        H = H * ((W.transpose() @ Y) / (W.transpose() @ W @ H + epsilon))
        W = W * ((Y @ H.transpose()) / (W @ H @ H.transpose()+epsilon))

        current_result = W @ H


        Y[current_result < 1] = 1
        Y[(current_result >= 1) & (current_result <= k)] = current_result[
            (current_result >= 1) & (current_result <= k)
        ]
        Y[current_result >= k] = current_result[current_result >= k]

        iter += 1

    # Booleanization

    print(W)

    print(H)

    return booleanization(X, W, H, 10)


def booleanization(
    X: np.ndarray, W: np.ndarray, H: np.ndarray, npoints: int
) -> Tuple[np.ndarray, np.ndarray]:
    W_p = np.linspace(np.min(W), np.max(W), npoints)
    H_p = np.linspace(np.min(H), np.max(H), npoints)

    min_distance = -1
    argmin_W = 0
    argmin_H = 0

    for delta_W in W_p:
        for delta_H in H_p:
            # apply threshold

            W_prime = W > delta_W
            H_prime = H > delta_H

            # compute distance

            current_distance = euclidian_distance(X, W @ H)
            if current_distance < min_distance:
                min_distance = current_distance
                argmin_W = delta_W
                argmin_H = delta_H

    W_prime = W > argmin_W
    H_prime = H > argmin_H

    return W_prime, H_prime


def euclidian_distance(A: np.ndarray, B: np.ndarray) -> float:
    return np.sum((A - B) ** 2)


def test():
    # random boolean matrix
    X = np.random.rand(3, 3) > 0.5

    W,H=banmf(X,2,100)

    print(W)

    print(H)


test()