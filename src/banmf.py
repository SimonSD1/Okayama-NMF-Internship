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


def yamada_solve(loop,W,H,Y,X,n,m,rank)->Tuple[np.ndarray,np.ndarray]:
    for t in range(loop): # t=0から loop-1 回 繰り返す
        #loop_cnt += 1 
        
        # 前の近似誤差（Frobeniusノルム）を記録
        error = np.linalg.norm(Y - np.dot(W, H), 'fro')
        #error_val_hist = np.array([error])   #初期値が f_val_hist の先頭に入る
        
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

    return W,H

def yamade_booleanization(W,H,X,npoint)->Tuple[np.ndarray,np.ndarray]:
    w_thresholds = np.linspace(W.min(), W.max(), npoint)
    h_thresholds = np.linspace(H.min(), H.max(), npoint)

    best_error = np.inf
    W_best = W
    H_best = H
    X_best = None

    # 2. すべてのしきい値の組み合わせで試す（総当たり）
    for wt in w_thresholds:
        for ht in h_thresholds:
            W_bin = (W > wt).astype(int)   #更新後のWの各要素をしきい値でブール化
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
        W = W * ((Y @ H.transpose()) / (W @ H @ H.transpose() + epsilon))
        H = H * ((W.transpose() @ Y) / (W.transpose() @ W @ H + epsilon))

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


def banmf(
    X: np.ndarray, k: int, Niter: int, nb_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    
    n,m=np.shape(X)

    Y, W, H = banmf_initialization(X, k)

    W_yamada,H_yamada=yamada_solve(200,W,H,Y,X,n,m,k)

    W, H, _ = banmf_auxiliary_solve(X, Y, W, H, k, Niter)

    W_yamada,H_yamada=yamade_booleanization(W_yamada,H_yamada,X,nb_points)

    W,H=booleanization(X, W, H, nb_points)

    print("yamada : ")
    print(W_yamada,H_yamada)

    print("simon")
    print(W,H)

    return W,H
