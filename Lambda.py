import numpy as np
from scipy.integrate import fixed_quad
from scipy.special import legendre

def ComputeLambda(i,j,k):
    """ (1/4)\int_{-1}^1 P_i(x) P_j(x) P_k(x) dx
    where P_i is Legendre polynomical of degree i
    """
    n = i+j+k
    if n&1 or i+j<k or j+k<i or k+i<j: return 0

    y = legendre(i) * legendre(j) * legendre(k)
    return fixed_quad(y,0,1,n=(n>>1)+1)[0]/2


# Lambda[i,j,k] = (1/2)(i,j,k;0,0,0)^2
#   where (i,j,k;l,m,n) is Wigner 3j symbol
N = 9
Lambda = np.zeros((N,N,N))
for i in range(N):
    for j in range(i,N):
        for k in range(j,N):
            Lambda[i,j,k] = ComputeLambda(i,j,k)
            Lambda[j,k,i] = Lambda[i,j,k]
            Lambda[k,i,j] = Lambda[i,j,k]
            Lambda[k,j,i] = Lambda[i,j,k]
            Lambda[j,i,k] = Lambda[i,j,k]
            Lambda[i,k,j] = Lambda[i,j,k]
