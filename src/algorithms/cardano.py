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
    fit = np.sum((Y - W @ H.T) ** 2)
    regW = lam / 2 * np.sum((W * W - W) ** 2)
    regH = lam / 2 * np.sum((H * H - H) ** 2)
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
        Rk = E + np.outer(W[:, k], H[:, k])

        norm_hk_sq = np.linalg.norm(H[:, k]) ** 2
        p = norm_hk_sq / lam + (2.0 * delta - 1.0) / 4.0
        q = (
            (norm_hk_sq / (2 * lam) + delta / 4)
            - (1 / lam) * (Rk @ H[:, k])
            - (delta / 2) * W[:, k]
        )

        sm = solve_cardano(p, q)
        W[:, k] = np.maximum(0, sm + 0.5)

        norm_wk_sq = np.linalg.norm(W[:, k]) ** 2
        u = norm_wk_sq / lam + (2 * delta - 1) / 4
        v = (
            (norm_wk_sq / (2 * lam) + delta / 4)
            - (1 / lam) * (Rk.T @ W[:, k])
            - (delta / 2) * H[:, k]
        )

        tn = solve_cardano(u, v)
        H[:, k] = np.maximum(0, tn + 0.5)

        E = Rk - np.outer(W[:, k], H[:, k])

    WH = W @ H.T
    Y[X] = np.clip(WH[X], 1, K)

    return W, H, Y


def check_stopping_condition(X, W, H, Y, lam, tau1, tau2):
    M, K = W.shape

    E = Y - W @ H.T
    grad_W = -2 * (E @ H) + lam * (2 * W**3 - 3 * W**2 + W)
    grad_H = -2 * (E.T @ W) + lam * (2 * H**3 - 3 * H**2 + H)
    grad_Y = 2 * E

    w_cond_1 = (grad_W[W <= tau2] >= -tau1).all()
    w_cond_2 = (np.abs(grad_W[W > tau2]) <= tau1).all()
    if not (w_cond_1 and w_cond_2):
        return False

    h_cond_1 = (grad_H[H <= tau2] >= -tau1).all()
    h_cond_2 = (np.abs(grad_H[H > tau2]) <= tau1).all()
    if not (h_cond_1 and h_cond_2):
        return False

    grad_Y_masked = grad_Y[X]
    Y_masked = Y[X]

    idx_low = Y_masked <= 1 + tau2
    if np.any(grad_Y_masked[idx_low] < -tau1):
        return False

    idx_high = Y_masked >= K - tau2
    if np.any(grad_Y_masked[idx_high] > tau1):
        return False

    idx_mid = (~idx_low) & (~idx_high)
    if np.any(np.abs(grad_Y_masked[idx_mid]) > tau1):
        return False

    return True


def cardano_solve_aux(X, W, H, Y, lam, delta, tau1, tau2):
    i = 0
    previous = np.inf
    while not check_stopping_condition(X, W, H, Y, lam, tau1, tau2):

        W, H, Y = update_w_h_y(X, W, H, Y, lam, delta)
        
        if full_objective(Y, W, H, lam) > previous:
            print("increase :(")
            break
        
        previous = full_objective(Y, W, H, lam)
        print("distance=", previous)
        i += 1
    print("iterations:",i)


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


def convergence(X, Y, W, H, lamb, del1, del2):
    N, M = X.shape
    K = W.shape[1]
    conv = 1
    gradf_W = 2 * np.dot(np.dot(W, H.T) - Y, H) + lamb * np.multiply(
        W**2 - W, 2 * W - np.ones((N, K))
    )
    gradf_H = 2 * np.dot(np.dot(H, W.T) - Y.T, W) + lamb * np.multiply(
        H**2 - H, 2 * H - np.ones((M, K))
    )
    gradf_Y = 2 * (Y - np.dot(W, H.T))

    if np.any(gradf_W < -del1):
        conv = 0
    elif np.any(gradf_W[W > del2] > del1):
        conv = 0
    elif np.any(gradf_H < -del1):
        conv = 0
    elif np.any(gradf_H[H > del2] > del1):
        conv = 0
    elif np.any(gradf_Y[(Y < K - del2) & (X == 1)] < -del1):
        conv = 0
    elif np.any(gradf_Y[(Y > 1 + del2) & (X == 1)] > del1):
        conv = 0

    return conv


def cardano_bmf(X, k, lam, delta, tau1, tau2, L) -> Tuple[np.ndarray, np.ndarray]:

    Y, W, H = banmf_initialization(X, k)

    cardano_solve_aux(X, W, H, Y, lam, delta, tau1, tau2)

    W, H = booleanization(X, W, H, L)

    print(boolean_distance(X,W@H.T))

    return W, H


if __name__ == "__main__":

    X = (np.random.rand(50, 50) > 0.5).astype(bool)
    k = 25
    lam = 1
    delta = 0.501
    tau1 = 0.005
    tau2 = 0.001
    booleanization_points = 25

    W, H = cardano_bmf(X, k, lam, delta, tau1, tau2, booleanization_points)

    print(W)
    print(H)
