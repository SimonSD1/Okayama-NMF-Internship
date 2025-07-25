## Implementation of the algorithm from "A New Boolean Matrix Factorization Algotithm Based on Cardano's Mathod"

from algorithms.utils import *


def solve_cardano(p, q):
    delta = (q / 2) ** 2 + (p / 3) ** 3
    sqrt_delta = np.sqrt(delta)
    u = np.cbrt(-q / 2 + sqrt_delta)
    v = np.cbrt(-q / 2 - sqrt_delta)
    return u + v


# try to parallelize change update E and RK
def update_w_h_y(X, W, H, Y, lam, delta):
    M, K = W.shape

    E = Y - W @ H

    for k in range(K):

        hk = H[k, :]

        ## the outer function compute the correct thing
        # wk@hk.T does not work because hk.T==hk here
        Rk = E + np.outer(W[:, k], hk)

        norm_hk_sq = np.linalg.norm(hk) ** 2
        p = norm_hk_sq / lam + (2.0 * delta - 1.0) / 4.0
        q = (
            (norm_hk_sq / (2 * lam) + delta / 4)
            - (1 / lam) * (Rk @ hk)
            - (delta / 2) * W[:, k]
        )

        sm = solve_cardano(p, q)

        W[:, k] = np.maximum(0, sm + 0.5)

        norm_wk_sq = np.linalg.norm(W[:, k]) ** 2
        u = norm_wk_sq / lam + (2 * delta - 1) / 4
        v = (
            (norm_wk_sq / (2 * lam) + delta / 4)
            - (1 / lam) * (Rk.T @ W[:, k])
            - (delta / 2) * H[k, :]
        )

        tn = solve_cardano(u, v)
        H[k, :] = np.maximum(0, tn + 0.5)

        E = Rk - np.outer(W[:, k], H[k, :])

    WH = W @ H
    Y[X] = np.clip(WH[X], 1, K)

    return W, H, Y


def check_stopping_condition(X, W, H, Y, lam, tau1, tau2):
    M, K = W.shape

    E = Y - W @ H
    grad_W = -2 * (E @ H.T) + lam * (2 * W**3 - 3 * W**2 + W)
    grad_H = -2 * (E.T @ W) + lam * (2 * H.T**3 - 3 * H.T**2 + H.T)
    grad_Y = 2 * E

    w_cond_1 = (grad_W[W <= tau2] >= -tau1).all()
    w_cond_2 = (np.abs(grad_W[W > tau2]) <= tau1).all()
    if not (w_cond_1 and w_cond_2):
        return False

    h_cond_1 = (grad_H[H.T <= tau2] >= -tau1).all()
    h_cond_2 = (np.abs(grad_H[H.T > tau2]) <= tau1).all()
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
    # previous = np.inf
    while not check_stopping_condition(X, W, H, Y, lam, tau1, tau2):

        W, H, Y = update_w_h_y(X, W, H, Y, lam, delta)


def cardano_bmf(X, k, lam, delta, tau1, tau2, L) -> Tuple[np.ndarray, np.ndarray]:

    Y, W, H = random_initialization_Y_W_H(X, k)
    cardano_solve_aux(X, W, H, Y, lam, delta, tau1, tau2)

    W, H = booleanization(X, W, H, L)

    print("cardano before local search:",boolean_distance(X,W@H))

    W, H = local_search(X, W, H, k)

    print("cardano after local search:",boolean_distance(X,W@H))
    return W, H


if __name__ == "__main__":

    np.random.seed(42)
    #X = (np.random.rand(5, 5) > 0.5).astype(bool)
    X = np.random.randint(0,2,(40,40))
    k =     4
    lam = 2
    delta = 0.51
    tau1 = 0.07
    tau2 = 0.07
    booleanization_points = 20

    print("z")
    W, H = cardano_bmf(X, k, lam, delta, tau1, tau2, booleanization_points)

    print(W)
    print(H)

    print("disantce : ",boolean_distance(X,W@H))
