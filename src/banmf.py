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
    W = np.random.rand(n, k) * 100
    H = np.random.rand(k, m) * 100

    return Y, W, H


def yamada_solve(loop, W, H, Y, X, n, m, rank) -> Tuple[np.ndarray, np.ndarray]:

    for t in range(loop):  # t=0から loop-1 回 繰り返す
        # loop_cnt += 1

        # 前の近似誤差（Frobeniusノルム）を記録
        error = np.linalg.norm(Y - np.dot(W, H), "fro")
        # error_val_hist = np.array([error])   #初期値が f_val_hist の先頭に入る

        # 乗法的更新規則
        W *= (Y @ H.T) / (W @ H @ H.T)
        H *= (W.T @ Y) / (W.T @ W @ H)
        WH = W @ H

        # step3 補助行列の更新=====================
        for i in range(n):
            for j in range(m):
                if X[i, j] == 1:
                    if WH[i, j] < 1:
                        Y[i, j] = 1
                    elif WH[i, j] > rank:
                        Y[i, j] = rank
                    else:
                        Y[i, j] = WH[i, j]
                else:
                    Y[i, j] = 0

    return W, H


def yamade_booleanization(W, H, X, npoint) -> Tuple[np.ndarray, np.ndarray]:
    w_thresholds = np.linspace(W.min(), W.max(), npoint)
    h_thresholds = np.linspace(H.min(), H.max(), npoint)

    best_error = np.inf
    W_best = W
    H_best = H
    X_best = None

    # 2. すべてのしきい値の組み合わせで試す（総当たり）
    for wt in w_thresholds:
        for ht in h_thresholds:
            W_bin = (W > wt).astype(int)  # 更新後のWの各要素をしきい値でブール化
            H_bin = (H > ht).astype(int)

            # ブール積で再構成 X_hat（論理積+論理和）
            X_hat = (W_bin @ H_bin > 0).astype(int)

            # 元のXとの誤差を計算
            error = np.sum(np.abs(X - X_hat))

            # 最小誤差なら保存
            if error < best_error:
                best_error = error
                W_best = W_bin
                H_best = H_bin
                X_best = X_hat

    return W_best, H_best


def banmf_auxiliary_solve(
    X: np.ndarray, Y: np.ndarray, W: np.ndarray, H: np.ndarray, k: int, Niter: int
) -> Tuple[np.ndarray, np.ndarray, list]:

    convergence_result = []
    for _ in range(Niter):
        W = W * ((Y @ H.transpose()) / (W @ H @ H.transpose()))
        H = H * ((W.transpose() @ Y) / (W.transpose() @ W @ H))

        current_result = W @ H
        # clip allows to put all data in the constraint
        Y[X] = np.clip(current_result[X], 1, k)
        convergence_result.append(euclidian_distance(Y, W @ H))

    return W, H, convergence_result


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


def local_search(
    X: np.ndarray, W: np.ndarray, H: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    finished = False

    n, m = np.shape(X)

    lowest = boolean_distance(X, W @ H)

    while not finished:
        print("iter")
        finished = True
        for i in range(n):
            for j in range(k):
                W[i][j] = not W[i][j]

                if boolean_distance(X, W @ H) >= lowest:
                    W[i][j] = not W[i][j]
                else:
                    lowest = boolean_distance(X, W @ H)
                    finished = False

        for i in range(k):
            for j in range(m):
                H[i][j] = not H[i][j]

                if boolean_distance(X, W @ H)>=lowest:
                    H[i][j] = not H[i][j]
                else:
                    lowest = boolean_distance(X, W @ H)
                    finished = False

    return W, H


def banmf(
    X: np.ndarray, k: int, Niter: int, nb_points: int
) -> Tuple[np.ndarray, np.ndarray]:

    Y, W, H = banmf_initialization(X, k)

    W, H, _ = banmf_auxiliary_solve(X, Y, W, H, k, Niter)

    W, H = booleanization(X, W, H, nb_points)

    return W, H

def banmf_local_search(
    X: np.ndarray, k: int, Niter: int, nb_points: int
) -> Tuple[np.ndarray, np.ndarray]:

    Y, W, H = banmf_initialization(X, k)

    W, H, _ = banmf_auxiliary_solve(X, Y, W, H, k, Niter)

    W, H = booleanization(X, W, H, nb_points)

    print("avant local search : ", boolean_distance(X,W@H))
    
    W,H = local_search(X,W,H,k)

    return W, H

def brute_force(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    n, m = np.shape(X)

    lowest = np.inf

    total_bits_W = n * k
    nb_matrices_W = 2**total_bits_W

    total_bits_H = k * m
    nb_matrices_H = 2**total_bits_H

    best_W = np.random.rand(n, k)
    best_H = np.random.rand(k, m)

    for int_w in range(nb_matrices_W):
        for int_h in range(nb_matrices_H):

            bits_w = np.array(
                list(np.binary_repr(int_w, width=total_bits_W)), dtype=int
            ).astype(bool)
            W = bits_w.reshape((n, k))

            bits_h = np.array(
                list(np.binary_repr(int_h, width=total_bits_H)), dtype=int
            ).astype(bool)
            H = bits_h.reshape((k, m))

            print(W)
            print(H)
            distance = boolean_distance(X, W @ H)
            if distance < lowest:
                best_H = H
                best_W = W
                lowest = distance

    return best_W, best_H
