## Implementation of the algorithm from "A New Boolean Matrix Factorization Algotithm Based on Cardano's Mathod"


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


def boolean_distance(A: np.ndarray, B: np.ndarray) -> int:
    return np.sum(A != B)


def boolean_distance_axis(X, Y, axis=1):
    return np.count_nonzero(X != Y, axis=axis)


def euclidian_norm_squared(v):
    return np.dot(v, v)

def full_objective(Y, W, H, lam):
    fit = np.sum((Y - W @ H.T)**2)
    regW = lam/2 * np.sum((W*W - W)**2)
    regH = lam/2 * np.sum((H*H - H)**2)
    return fit + regW + regH


def banmf_initialization(
    X: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    (n, m) = X.shape

    # Initialization
    Y = X.copy().astype(float)
    W = np.random.rand(n, k) * 10
    H = np.random.rand(k, m) * 10

    return Y, W, H.T


def solve_cardano(p, q):
    delta = (q / 2) ** 2 + (p / 3) ** 3
    sqrt_delta = np.sqrt(delta)
    u = np.cbrt(-q / 2 + sqrt_delta)
    v = np.cbrt(-q / 2 - sqrt_delta)
    return u + v


def update_w_h_y(X, W, H, Y, lam, delta):
    M, K = W.shape

    E = Y - W @ H.T

    for k in range(K):

        ## the outer function compute the correct thing
        # wk@hk.T does not work because hk.T==hk here
        Rk = E + np.outer(W[:,k], H[:,k])

        norm_hk_sq = euclidian_norm_squared(H[:,k])
        p = norm_hk_sq / lam + (2 * delta - 1) / 4
        q = (
            (norm_hk_sq / (2 * lam) + delta / 4)
            - (1 / lam) * (Rk @ H[:,k])
            - (delta / 2) * W[:,k]
        )

        sm = solve_cardano(p, q)
        W[:, k] = np.maximum(0, sm + 0.5)
        
        ## need to update here dont forget

        E = Rk - np.outer(W[:,k], H[:,k])

        norm_wk_sq = euclidian_norm_squared(W[:,k])
        u = norm_wk_sq / lam + (2 * delta - 1) / 4
        v = (
            (norm_wk_sq / (2 * lam) + delta / 4)
            - (1 / lam) * (Rk.T @ W[:,k])
            - (delta / 2) * H[:,k]
        )

        tn = solve_cardano(u, v)
        H[:, k] = np.maximum(0, tn + 0.5)

        E = Rk - np.outer(W[:, k], H[:, k])

    WH = W @ H.T
    Y[X] = np.clip(WH[X], 1, K)

    return W, H, Y


def check_stopping_condition(W, H, Y, lam, tau1, tau2):
    E = Y - W @ H.T
    grad_W = -2 * (E @ H) + lam * 2 * (W**2 - W) * (2 * W - 1)
    grad_H = -2 * (E.T @ W) + lam * 2 * (H**2 - H) * (2 * H - 1)
    grad_Y = 2 * E

    # np.where(cond, val true, val false)
    # where W <= tau2 we check if grad_W >= -tau1 and elsewhere we check if grad_W in [-tau1, tau1]
    W_condition = np.where(W <= tau2, grad_W >= -tau1, np.abs(grad_W) <= tau1)

    H_condition = np.where(H <= tau2, grad_H >= -tau1, np.abs(grad_H) <= tau1)

    K = H.shape[1]
    cond_low = Y <= 1 + tau2
    cond_high = Y >= K - tau2
    cond_mid = (~cond_low) & (~cond_high)

    ## create matrix like Y with True entries
    Y_condition = np.full_like(Y, True, dtype=bool)

    Y_condition[cond_low] = grad_Y[cond_low] >= -tau1
    Y_condition[cond_high] = grad_Y[cond_high] <= tau1
    Y_condition[cond_mid] = np.abs(grad_Y[cond_mid]) <= tau1

    return W_condition.all() and H_condition.all() and Y_condition.all()


def cardano_solve_aux(X, W, H, Y, lam, delta, tau1, tau2):
    while not check_stopping_condition(W, H, Y, lam, tau1, tau2):
        W, H, Y = update_w_h_y(X, W, H, Y, lam, delta)
        print("distance : ", full_objective(Y,W,H,lam))


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
                boolean_distance(X, ((W > delta_W) @ (H > delta_H).T).astype(bool)),
            )
            for delta_W in W_p
            for delta_H in H_p
        ),
        key=lambda t: t[2],
    )

    W_prime = W > argmin_W
    H_prime = H > argmin_H

    return W_prime, H_prime


def cardano_bmf(X, k, lam, delta, tau1, tau2, L) -> Tuple[np.ndarray, np.ndarray]:

    Y, W, H = banmf_initialization(X, k)

    print("distance : ", boolean_distance(X, W @ H.T))

    cardano_solve_aux(X, W, H, Y, lam, delta, tau1, tau2)

    W, H = booleanization(X, W, H, L)

    print("distance : ", boolean_distance(X, W @ H.T))

    return W, H


if __name__ == "__main__":

    X = (np.random.rand(50, 50) > 0.5).astype(bool)
    k = 30
    lam = 1
    delta = 0.6
    tau1 = 0.5
    tau2 = 0.5
    booleanization_points = 25

    W, H = cardano_bmf(X, k, lam, delta, tau1, tau2, booleanization_points)
