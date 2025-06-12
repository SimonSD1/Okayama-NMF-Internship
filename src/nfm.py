# Implementation of the multiplicative update rule from Lee and Seung

import numpy as np
from typing import Tuple

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

    print("V=",V)
    print("W=",W)
    print("H=",H)
    print("WH=",W@H)

    
    # We apply the update rule on H and V
    # @ is matrix multiplication, * and / are element wise operations

    previous_distance=euclidian_distance(V,W@H)
    iter=0

    while iter<iter_max:
        H = H * ((W.transpose() @ V) / (W.transpose() @ W @ H))
        W = W * ((V @ H.transpose()) / (W @ H @ H.transpose()))

        distance = euclidian_distance(V, W@H)

        print(distance)

        if previous_distance-distance<tolerance:
            break

        previous_distance=distance    
        iter+=1

    return (W,H)



V=np.random.rand(2,2)*10

nmf(V,1,1000, 1e-50)

