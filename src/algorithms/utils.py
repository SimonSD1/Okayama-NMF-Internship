import numpy as np
from typing import Optional, Tuple


def euclidian_distance(A: np.ndarray, B: np.ndarray) -> float:
    return np.sum((A - B) ** 2)


def boolean_distance(A: np.ndarray, B: np.ndarray) -> int:
    return np.sum(A != B)


def boolean_distance_factors(A: np.ndarray, W: np.ndarray, H: np.ndarray):
    return np.sum(A != W @ H)


def boolean_distance_axis(X, Y, axis=1):
    return np.count_nonzero(X != Y, axis=axis)


def euclidian_norm_squared(v):
    return np.dot(v, v)


def random_initialization_Y_W_H(
    X: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    (n, m) = X.shape

    # Initialization
    Y = X.copy().astype(float)
    W = np.random.rand(n, k) 
    H = np.random.rand(k, m)

    return Y, W, H


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
    WH = W @ H

    while not finished:
        finished = True

        row_distances = boolean_distance_axis(X, WH, axis=1)

        for i in range(n):
            for j in range(k):
                W[i][j] ^= True

                # compute the matrix product only on modified data
                temp_result = W[i] @ H
                distance_modified_row = boolean_distance(X[i], temp_result)

                if distance_modified_row >= row_distances[i]:
                    W[i][j] ^= True
                else:
                    row_distances[i] = distance_modified_row
                    WH[i] = temp_result
                    finished = False

        column_distance = boolean_distance_axis(X, WH, axis=0)

        for i in range(k):
            for j in range(m):
                H[i][j] ^= True

                # si la distance de la colonnes j est inferieur on garde
                # et met a jour l'array des distance colonnes
                temp_result = W @ H[:, j]
                distance_modified_column = boolean_distance(X[:, j], temp_result)

                if distance_modified_column >= column_distance[j]:
                    H[i][j] ^= True
                else:
                    column_distance[j] = distance_modified_column
                    WH[:, j] = temp_result
                    finished = False

    return W, H


# possible uptimization compute only the dot product of the desired i and j
# not the full matrix product for the distance
# nb different (X, W@H)
# since we change only one component we can use a stored matrix and change only
# the component that need
def naive_local_search(
    X: np.ndarray, W: np.ndarray, H: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    finished = False

    n, m = np.shape(X)

    lowest = boolean_distance(X, W @ H)

    while not finished:
        finished = True
        for i in range(n):
            for j in range(k):
                W[i][j] ^= True

                distance = boolean_distance(X, W @ H)
                if distance >= lowest:
                    W[i][j] ^= True
                else:
                    lowest = distance
                    finished = False

        for i in range(k):
            for j in range(m):
                H[i][j] ^= True
                distance = boolean_distance(X, W @ H)

                if distance >= lowest:
                    H[i][j] ^= True
                else:
                    lowest = distance
                    finished = False

    return W, H
