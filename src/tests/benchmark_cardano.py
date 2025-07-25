from algorithms.utils import *
import time
import matplotlib.pyplot as plt

from algorithms.cardano import cardano_bmf
from algorithms.banmf import *

from typing import Tuple, Dict, Any

import time


def find_params_cardano(
    X: np.ndarray, k: int, booleanization_points: int, num_trials: int
) -> Dict[str, Any]:

    # start, end, nb points
    # parameters to test
    lams_set = np.linspace(0.0001, 5, 50)
    deltas_set = np.linspace(0.51, 10, 50)

    # fixed values, decrease -> less error so not tested
    tau1, tau2 = 0.5, 0.1

    best_avg_distance = 1000000000000000000000000000
    best_params = {}

    avg_distances = np.zeros((len(lams_set), len(deltas_set)))

    for i, lam in enumerate(lams_set):
        for j, delta in enumerate(deltas_set):

            distances = [
                boolean_distance_factors(
                    X,
                    *cardano_bmf(
                        X,
                        k,
                        lam,
                        delta,
                        tau1,
                        tau2,
                        booleanization_points,
                    ),
                )
                for _ in range(num_trials)
            ]

            print("next delta")

            current_avg_distance = np.mean(distances)
            avg_distances[i, j] = current_avg_distance

            if current_avg_distance < best_avg_distance:
                best_avg_distance = current_avg_distance
                best_params = {"lam": lam, "delta": delta, "tau1": tau1, "tau2": tau2}

    print(avg_distances)
    fig, ax = plt.subplots()
    im = ax.imshow(
        avg_distances,
        origin="lower",
        aspect="auto",
        extent=(deltas_set[0], deltas_set[-1], lams_set[0], lams_set[-1]),
        cmap="viridis",
    )

    fig.colorbar(im, ax=ax, label="Average distance")
    ax.set_xlabel("delta")
    ax.set_ylabel("lambda")
    ax.set_title("Cardano parametres heatmap")
    plt.savefig("../results/cardano_parameters_heatmap")
    plt.close(fig)

    return best_params


def compare_cardano_banmf(
    X: np.ndarray,
    k: int,
    cardano_params: Dict[str, Any],
    niter_banmf: int,
    booleanization_points: int,
    num_trials: int,
):
    cardano_distances = []
    banmf_distances = []
    rand_distances=[]
    regularized_banmf_distances=[]

    cardano_times = []
    banmf_times = []
    rand_times=[]
    regularized_banmf_times=[]

    for _ in range(num_trials):

        start = time.time()
        W_cardano, H_cardano = cardano_bmf(
            X, k, **cardano_params, L=booleanization_points
        )
        cardano_times.append(time.time() - start)

        start = time.time()
        W_banmf, H_banmf = banmf_local_search(X, k, niter_banmf, booleanization_points)
        banmf_times.append(time.time() - start)

        start = time.time()
        W_rand, H_rand = random_plus_local_search(X, k, 1500)
        rand_times.append(time.time() - start)

        start = time.time()
        W_reg, H_reg = regularized_banmf_local_search(X, k, 1500,30,0.1)
        regularized_banmf_times.append(time.time() - start)

        cardano_distances.append(boolean_distance(X, W_cardano @ H_cardano))
        banmf_distances.append(boolean_distance(X, W_banmf @ H_banmf))
        rand_distances.append(boolean_distance(X, W_rand @ H_rand))
        regularized_banmf_distances.append(boolean_distance(X,W_reg@H_reg))

    cardano_mean = np.mean(cardano_distances)
    cardano_std = np.std(cardano_distances)
    rand_std = np.std(rand_distances)
    regularized_banmf_std=np.std(regularized_banmf_distances)
    banmf_mean = np.mean(banmf_distances)
    banmf_std = np.std(banmf_distances)
    rand_mean = np.mean(rand_distances)
    regularized_banmf_mean=np.mean(regularized_banmf_distances)
    plt.figure()
    algos = ["Cardano BMF", "BANMF","Rand+localsearch","regularized banmf"]
    means = [cardano_mean, banmf_mean,rand_mean,regularized_banmf_mean]
    stds = [cardano_std, banmf_std,rand_std,regularized_banmf_std]
    times=[np.mean(cardano_times),np.mean(banmf_times),np.mean(rand_times),np.mean(regularized_banmf_times)]

    bars = plt.bar(algos, means, yerr=stds)
    plt.ylabel("Average distance")
    plt.title("cardano vs banmf")
    plt.bar_label(bars)
    for bar, t in zip(bars, times):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,  # un peu au-dessus de l'erreur
            f"Mean time: {t:.2f}s",
            ha='center', va='bottom',
            fontsize=9, color='gray'
        )
    plt.savefig("../results/cardano_vs_banmf.png")
    plt.close()


if __name__ == "__main__":
    DIMENSION = (20, 20)
    X = (np.random.rand(*DIMENSION) > 0.5).astype(bool)
    K = 10
    BOOLEANIZATION_POINTS = 30
    NUM_TRIALS = 20

    # for 20 by 20
    # best_params=  {'lam': np.float64(0.0112), 'delta': np.float64(1.7591919191919192), 'tau1': 0.001, 'tau2': 0.001}
    best_params=  {'lam': np.float64(0.1), 'delta': np.float64(0.501), 'tau1': 0.005, 'tau2': 0.001}
    #best_params = {'lam': np.float64(0.1), 'delta': np.float64(0.501), 'tau1': 0.007, 'tau2': 0.007}
    #best_params={'lam': np.float64(0.10213877551020409), 'delta': np.float64(6.126530612244897), 'tau1': 0.5, 'tau2': 0.1}
    #best_params = find_params_cardano(X, K, BOOLEANIZATION_POINTS, NUM_TRIALS)

    print("best params: ", best_params)

    compare_cardano_banmf(X, K, best_params, 1500, BOOLEANIZATION_POINTS, NUM_TRIALS)
