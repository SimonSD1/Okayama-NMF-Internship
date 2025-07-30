from algorithms.utils import *
import time
import matplotlib.pyplot as plt

from algorithms.cardano import *
from algorithms.banmf import *

from typing import Tuple, Dict, Any

import time
import csv
import os


def find_params_cardano(
    X: np.ndarray, k: int, booleanization_points: int, num_trials: int, max_iter: int
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
                        X, k, lam, delta, tau1, tau2, booleanization_points, max_iter
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


def get_zoo_dataset_matrix():

    filename = "../data/zoo.data"

    with open(filename, "r") as f:
        X = []
        for line in f:
            parts = line.strip().split(",")[1:]  # Skip animal name
            filtered = [
                int(x) for i, x in enumerate(parts) if i not in (13, 17)
            ]  # need to skip some columns that have non boolean value
            bool_values = [val > 0 for val in filtered]
            X.append(bool_values)
    X = np.array(X, dtype=bool)

    return X


def get_lucap0_dataset_matrix():
    X = []

    filename = "../data/lucap0_train.data"

    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            int_parts = [int(x) for x in parts]
            X.append(int_parts)

    print(X)

    X = np.array(X, dtype=bool)
    # print(X.astype(bool))
    return X


def plot_comparison_from_csv(filename: str):
    csv_path = f"../results/csv_results/{filename}.csv"
    image_path = f"../results/{filename}.png"

    algos = []
    means = []
    stds = []
    times = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            algos.append(row["Algorithm"])
            means.append(float(row["Mean"]))
            stds.append(float(row["Std"]))
            times.append(float(row["Time"]))

    plt.figure(figsize=(13,6))
    bars = plt.bar(algos, means, yerr=stds)
    plt.ylabel("Average boolean distance")
    plt.title("Algorithms comparison on lung cancer dataset, k=100, lambda=0.1")
    plt.bar_label(bars)

    plt.subplots_adjust(bottom=0.2)

    for bar, t in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            -max(means) * 0.1,  
            f"mean time: {t:.2f}s",
            ha="center",
            va="top",
            fontsize=7,
            color="steelblue",
            rotation=0,
            fontweight="medium"
        )


    os.makedirs("../results", exist_ok=True)
    plt.savefig(image_path)
    plt.close()


def compare_cardano_banmf(
    X: np.ndarray,
    k: int,
    cardano_params: Dict[str, Any],
    niter_banmf: int,
    booleanization_points: int,
    num_trials: int,
    max_iter: int,
    filename: str,
):
    cardano_local_search_distances = []
    cardano_distances=[]
    banmf_local_search_distances = []
    rand_distances = []
    regularized_banmf_local_search_distances = []
    regularized_banmf_distances=[]


    cardano_local_search_times = []
    cardano_times=[]
    banmf_local_search_times = []
    rand_times = []
    regularized_banmf_local_search_times = []
    regularized_banmf_times=[]

    for _ in range(num_trials):

        start = time.time()
        W_cardano_local_search, H_cardano_local_search = cardano_bmf_local_search(
            X, k, **cardano_params, L=booleanization_points, max_iter=max_iter
        )
        cardano_local_search_times.append(time.time() - start)

        start = time.time()
        W_cardano, H_cardano = cardano_bmf(
            X, k, **cardano_params, L=booleanization_points, max_iter=max_iter
        )
        cardano_times.append(time.time() - start)

        start = time.time()
        W_banmf_local_search, H_banmf_local_search = banmf_local_search(X, k, niter_banmf, booleanization_points)
        banmf_local_search_times.append(time.time() - start)

        start = time.time()
        W_rand_local_search, H_rand_local_search = random_plus_local_search(X, k, max_iter)
        rand_times.append(time.time() - start)

        start = time.time()
        W_reg_local_search, H_reg_local_search = regularized_banmf_local_search(
            X, k, max_iter, booleanization_points, 0.1
        )
        regularized_banmf_local_search_times.append(time.time() - start)

        start = time.time()
        W_reg, H_reg = regularized_banmf(
            X, k, max_iter, booleanization_points, 0.1
        )
        regularized_banmf_times.append(time.time() - start)
       

        cardano_local_search_distances.append(boolean_distance(X, W_cardano_local_search @ H_cardano_local_search))
        cardano_distances.append(boolean_distance(X, W_cardano @ H_cardano))

        banmf_local_search_distances.append(boolean_distance(X, W_banmf_local_search @ H_banmf_local_search))
        rand_distances.append(boolean_distance(X, W_rand_local_search @ H_rand_local_search))
        regularized_banmf_local_search_distances.append(boolean_distance(X, W_reg_local_search @ H_reg_local_search))
        regularized_banmf_distances.append(boolean_distance(X, W_reg @ H_reg))


    cardano_mean = np.mean(cardano_distances)
    cardano_std = np.std(cardano_distances)

    cardano_mean_local_search = np.mean(cardano_local_search_distances)
    cardano_std_local_search = np.std(cardano_local_search_distances)

    rand_mean = np.mean(rand_distances)
    rand_std = np.std(rand_distances)

    regularized_banmf_std = np.std(regularized_banmf_distances)
    regularized_banmf_mean = np.mean(regularized_banmf_distances)

    regularized_banmf_std_local_search = np.std(regularized_banmf_local_search_distances)
    regularized_banmf_mean_local_search = np.mean(regularized_banmf_local_search_distances)

    banmf_mean = np.mean(banmf_local_search_distances)
    banmf_std = np.std(banmf_local_search_distances)
    
    plt.figure()
    algos = ["Cardano BMF", "BANMF local search", "Rand+localsearch", "regularized banmf", "Cardano local search", "Reg Banmf local search"]
    means = [cardano_mean, banmf_mean, rand_mean, regularized_banmf_mean, cardano_mean_local_search,regularized_banmf_mean_local_search]
    stds = [cardano_std, banmf_std, rand_std, regularized_banmf_std, cardano_std_local_search, regularized_banmf_std_local_search]
    times = [
        np.mean(cardano_times),
        np.mean(banmf_local_search_times),
        np.mean(rand_times),
        np.mean(regularized_banmf_times),
        np.mean(cardano_local_search_times),
        np.mean(regularized_banmf_local_search_times)
    ]

    os.makedirs("../results/csv_results", exist_ok=True)
    with open(f"../results/csv_results/{filename}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Mean", "Std", "Time"])
        for algo, mean, std, t in zip(algos, means, stds, times):
            writer.writerow([algo, mean, std, t])

    plot_comparison_from_csv(filename)


if __name__ == "__main__":
    # DIMENSION = (20, 20)
    # X = (np.random.rand(*DIMENSION) > 0.5).astype(bool)
    #X = get_zoo_dataset_matrix()
    X = get_lucap0_dataset_matrix()
    print(X)
    K = 100
    BOOLEANIZATION_POINTS = 30
    NUM_TRIALS = 5

    # for 20 by 20
    # best_params=  {'lam': np.float64(0.0112), 'delta': np.float64(1.7591919191919192), 'tau1': 0.001, 'tau2': 0.001}
    best_params = {
        "lam": np.float64(0.1),
        "delta": np.float64(0.501),
        "tau1": 0.005,
        "tau2": 0.001,
    }
    # best_params = {'lam': np.float64(0.1), 'delta': np.float64(0.501), 'tau1': 0.007, 'tau2': 0.007}
    # best_params={'lam': np.float64(0.10213877551020409), 'delta': np.float64(6.126530612244897), 'tau1': 0.5, 'tau2': 0.1}
    # best_params = find_params_cardano(X, K, BOOLEANIZATION_POINTS, NUM_TRIALS)

    print("best params: ", best_params)

    max_iter = 500

    compare_cardano_banmf(
       X,
       K,
       best_params,
       max_iter,
       BOOLEANIZATION_POINTS,
       NUM_TRIALS,
       max_iter=max_iter,
       filename="lucap_k=100_lam=0_1",
    )

    #plot_comparison_from_csv("zoo_k=12_lam=0_1")
