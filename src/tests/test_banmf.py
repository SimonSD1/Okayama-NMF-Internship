from banmf import *
import time

def test_nb_point_booleanization(
    n: int, m: int, nb_tests: int, plot: bool, k: Optional[int] = None
):
    X = (np.random.rand(n, m) > 0.5).astype(bool)

    if k is None:
        k = random.randint(1, min(n, m))

    Y, W, H = banmf_initialization(X, k)

    W, H = banmf_auxiliary_solve(X, Y, W, H, k, 200)

    booleanization_result = []
    for npoints in range(1, nb_tests):
        W_prime, H_prime = booleanization(X, W, H, npoints)
        booleanization_result.append(boolean_distance(X, W_prime @ H_prime))

    if plot == True:
        x = range(1, nb_tests)

        fig, ax = plt.subplots()
        ax.set_ylabel("final distance")
        ax.set_xlabel("number of points")
        ax.set_title(
            f"Final distance vs number of point in booleanization on a {n} by {m}"
        )
        ax.plot(x, booleanization_result, label="final distance")
        plt.legend()
        plt.savefig("../results/npoints_booleanization.png")

    return booleanization_result


def test_nb_points_3d(nb_tests, plot: bool):
    # for different matrix sizes
    result_matrix = []

    for size in range(1, nb_tests):
        bool_result = test_nb_point_booleanization(
            size, size, nb_tests, plot=False, k=max(1, int(size / 2))
        )
        result_matrix.append(
            bool_result
        )  # chaque ligne correspond à une taille de matrice

    result_matrix = np.array(result_matrix)

    if plot:
        # plt.figure(figsize=(10, 8))
        plt.imshow(
            result_matrix,
            origin="lower",
            extent=(1.0, float(nb_tests - 1), 1.0, float(nb_tests - 1)),
        )
        plt.colorbar(label="final boolean distance")
        plt.xlabel("nb points in booleanization")
        plt.ylabel("Size of the matrix")
        plt.title("nb points and size of matrix vs boolean distance (k=size/2)")
        plt.savefig("../results/3d_booleanization_heatmap.png")


def test_latent_dimension(X: np.ndarray, nb_tests: int, nb_points):

    results_distance = []
    for k in range(1, nb_tests):
        W, H = banmf(X, k, 1000, nb_points)
        results_distance.append(boolean_distance(X, W @ H))

    x = range(1, nb_tests)
    return results_distance


def test_latent_booleanization_3d(n: int, m: int, nb_tests: int, plot: bool):
    result_matrix = []

    X = (np.random.rand(n, m) > 0.5).astype(bool)

    for nb_points in range(1, nb_tests):
        result_matrix.append(test_latent_dimension(X, nb_tests, nb_points))

    if plot:
        # plt.figure(figsize=(10, 8))
        plt.imshow(
            result_matrix,
            origin="lower",
            extent=(1.0, float(nb_tests - 1), 1.0, float(nb_tests - 1)),
        )
        plt.colorbar(label="final boolean distance")
        plt.xlabel("latent dimension")
        plt.ylabel("nb points in booleanization")
        plt.title("nb of points and latent dimension vs final distance")
        plt.savefig("../results/latent_booleanization_heatmap.png")

def compair_banmf(X:np.ndarray,k:int, Niter:int, nb_points:int)->bool:
    n,m=np.shape(X)

    Y, W, H = banmf_initialization(X, k)

    W_save, H_save, Y_save = W.copy(),H.copy(),Y.copy()

    W_yamada,H_yamada=yamada_solve(Niter,W,H,Y,X,n,m,k)

    W, H = banmf_auxiliary_solve(X, Y_save, W_save, H_save, k, Niter)


    W_yamada,H_yamada=yamade_booleanization(W_yamada,H_yamada,X,nb_points)

    W,H=booleanization(X, W, H, nb_points)
    return (W_yamada==W).all() and (H_yamada==H).all()


def test_brute_force(X:np.ndarray, k:int):
    W,H=brute_force(X,k)
    print("")
    print(boolean_distance(X,W@H))

def compair_factorizations(n:int, m:int, k:int):

    X=(np.random.rand(n,m)>0.5).astype(bool)

    W_banmf,H_banmf = banmf(X,k,200,25) 

    W_brute,H_brute = brute_force(X,k)

    print("distance banmf = ",boolean_distance(X,W_banmf@H_banmf))
    print("distance brute = ",boolean_distance(X,W_brute@H_brute))

def test_local_search( nb_tests:int):

    before_distances = []
    final_distances = []
    sizes = []
    distance_gains = []
    time_pourcentage=[]


    for i in range(1, nb_tests + 1):
        size = i * 15
        X = (np.random.rand(size, size) > 0.5).astype(bool)

        W, H, before,time_before, time_local_search = banmf_local_search(X, size // 2, 1000, 25)

        after = boolean_distance(X, W @ H)
        gain = 100 * (before - after) / before if before != 0 else 0

        before_distances.append(before)
        final_distances.append(after)
        sizes.append(size)
        distance_gains.append(gain)
        time_pourcentage.append(100 * ((time_local_search))/(time_local_search+time_before) )


    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_ylabel("Boolean distance")
    ax.set_xlabel("Matrix size (n x n)")
    ax.set_title("Local search: before vs after")

    ax.plot(sizes, before_distances, label="before local search", marker='o')
    ax.plot(sizes, final_distances, label="after local search", marker='x')

    for i, size in enumerate(sizes):
        ax.annotate(
            f"distance gained:{distance_gains[i]:.2f}%\ntime over total:{time_pourcentage[i]:.2f}%",
            (size, final_distances[i]),
            textcoords="offset points",
            xytext=(5, -10),
            fontsize=8,
            color='gray'
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig("../results/local_search_with_info.png")
    plt.show()

def test_optimisation_local_search(nb_tests:int, Niter:int):

    result_opti=[]
    result_non_opti=[]
    sizes=[]

    for i in range(1, nb_tests + 1):
        size = i * 5

        sizes.append(size)

        X = (np.random.rand(size, size) > 0.5).astype(bool)

        k=size//2

        Y, W, H = banmf_initialization(X, k)

        W, H = banmf_auxiliary_solve(X, Y, W, H, k, Niter)

        W,H=booleanization(X, W, H, 25)

        W_opti=W.copy()
        H_opti=H.copy()

        start = time.time()
        W,H = local_search(X,W,H,k)
        time_non_opti = time.time()-start

        start=time.time()
        W_opti,H_opti = opti_local_search(X,W_opti,H_opti,k)
        time_opti=time.time()-start


        result_opti.append(time_opti)
        result_non_opti.append(time_non_opti)

        if((W!=W_opti).any() or (H!=H_opti).any()):
            print("error!!!!!!!!!!!!!!!!!!!!!!!!")

    fig, ax = plt.subplots()
    ax.set_ylabel("Time")
    ax.set_xlabel("Matrix size (n x n)")
    ax.set_title("Local search: opti vs naive")

    print(time_non_opti)
    print(time_opti)
    print(sizes)

    ax.plot(sizes, result_non_opti, label="time non opti", marker='o')
    ax.plot(sizes, result_opti, label="time opti search", marker='x')
    plt.legend()
    plt.tight_layout()
    plt.savefig("../results/optimization_local_search.png")
    plt.show()





# test_nb_point_booleanization(50,50,100)
# test_latent_dimension(100, 100, 100)
# test_nb_points_3d(50,True)
#test_latent_booleanization_3d(50, 50, 50, True)
# test_convergence_auxiliary(50,50,200,True)
#test_local_search(10)
test_optimisation_local_search(15,200)
