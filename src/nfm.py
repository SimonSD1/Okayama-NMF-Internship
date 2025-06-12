# Implementation of the multiplicative update rule from Lee and Seung

import random
import numpy as np
from typing import Tuple
import time
import matplotlib.pyplot as plt

def euclidian_distance(A:np.ndarray, B:np.ndarray)->float:
    return np.sum((A-B)**2)


def nmf(V:np.ndarray, r:int, iter_max:int, tolerance:float)->Tuple[np.ndarray,np.ndarray]:
    """
    Take a matrix V of size n by m and r\n
    Return W of size n by r and H of size r by m s.t V=~WH\n
    Stop when convergence is reached fot the given tolerance or when iter_max is reached
    """
    
    # Initialize the W and H
    (n,m)=V.shape
    W=np.random.rand(n,r)*10
    H=np.random.rand(r,m)*10

    
    # We apply the update rule on H and V
    # @ is matrix multiplication, * and / are element wise operations

    previous_distance=euclidian_distance(V,W@H)
    iter=0

    while iter<iter_max:
        H = H * ((W.transpose() @ V) / (W.transpose() @ W @ H))
        W = W * ((V @ H.transpose()) / (W @ H @ H.transpose()))

        distance = euclidian_distance(V, W@H)

        #print(distance)

        if previous_distance-distance<tolerance:
            break

        previous_distance=distance    
        iter+=1

    return (W,H)

def nmf_test(nb_tests:int)->list:
    # test on random matrix    

    time_results=[]

    for iter in range (1, nb_tests):

        V=np.random.rand(iter,iter)*iter
        (n,m)=V.shape
        r = random.randint(1,min(n,m))

        start = time.time()
        nmf(V=V,r=r,iter_max=1000,tolerance=1e-10)
        end = time.time()

        time_results.append((end-start))
        print("elapsed time = ",end-start)

    return time_results


nb_tests=200

time_results=nmf_test(nb_tests)

fig, ax = plt.subplots()
ax.set_ylabel("time")
ax.set_xlabel("size of V")
ax.set_title("NMF computation time vs matrix size")
ax.plot(range(1,nb_tests),time_results)
plt.savefig("../results/time_results.png")

