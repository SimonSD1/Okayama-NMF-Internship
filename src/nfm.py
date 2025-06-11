# Implementation of the multiplicative update rule from Lee and Seung

import numpy as np
from typing import Tuple

INITIAL_SIZE = 2
REDUCE_SIZE=1


def euclidian_distance(A:np.ndarray, B:np.ndarray)->float:
    return np.sqrt(np.sum((A-B))**2)


def nmf(V:np.ndarray, r:int, threshold:float)->Tuple[np.ndarray,np.ndarray]:
    """
    Take a matrix V of size n by m and r\n
    Return W of size n by r and H of size r by m s.t V=~WH 
    for the given threshold on the euclidean distance
    """
    
    # Initialize the W and H
    (n,m)=V.shape
    W=np.random.rand(n,r)
    H=np.random.rand(r,m)
    
    # We apply the update rule alternating between H and V
    
    
    
    
    


A=np.random.rand(3,2)
B=np.random.rand(3,2)

euclidian_distance(A,B)
