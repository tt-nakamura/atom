# reference:
#   W. R. Johnson, "Atomic Structure Theory"
#     section 2.3

import numpy as np
from scipy.integrate import solve_ivp, trapz
from scipy.interpolate import CubicSpline

def V(r,Z,E,l2,U):
    """ potential energy
    r = np.array of radial coordinates
    U = screening potential for an electron
    """
    return U(r) - Z/r + l2/(2*r*r) - E

def outward(r,Z,E,l,U):
    """ integrate outward from r[0] to r[-1]
    r[0] must be very small (practical origin)
    """
    l2 = l*(l+1)
    a = -Z/(l+1)*r[0]
    P = (1+a)*r[0]
    Q = l+1 + (l+2)*a
    f = lambda r,y: [y[1], 2*V(r,Z,E,l2,U)*y[0]]
    s = solve_ivp(f, [r[0],r[-1]], [P,Q], t_eval=r)
    return s.y[0], s.y[1]

def inward(r,Z,N,E,l,U):
    """ integrate inward from r[-1] to r[0]
    r[-1] must be very large (practical infinity)
    """
    l2 = l*(l+1)
    lm = (-2*E)**0.5

    k,a,s = 1,1,(Z-N+1)/lm
    P,P1,Q = 1,0,-lm
    while P!=P1:
        a /= 2*k*r[-1]
        P1 = P
        Q += a*((s+k)*(s-k+1) - l2)
        a *= (l2 - (s-k)*(s-k+1))/lm
        P += a
        k += 1

    f = lambda r,y: [y[1], 2*V(r,Z,E,l2,U)*y[0]]
    s = solve_ivp(f, [r[-1],r[0]], [P,Q], t_eval=r[::-1])
    return s.y[0,::-1], s.y[1,::-1]

def Schroedinger(Z,n,l,N=1,U=None,E=0):
    """ Z = atomic number,
    n = principal quantum number
    l = angular momentum,
    N = number of bound electrons,
    U = electron screening potential 
    E = initial guess for energy eigenvalue
    return an orbital having n-l-1 nodes
    """
    if N>Z: print('N must be N<=Z'); exit()
    if l>=n: print('l must be l<n'); exit()

    infty = 40
    dr = 1e-8
    npt,npt1 = 256,64
    EPS = 1e-12
    l2 = l*(l+1)
    n_nodes = n-l-1
    if E>=0: E = -0.5*((Z-N+1)/n)**2
    if U is None: U = lambda r: 0

    def shoot(E):
        R = infty/(-2*E)**0.5
        r = np.geomspace(dr,R,npt)
        rf = r[np.max(np.nonzero(V(r[1:], Z,E,l2,U) *
                                 V(r[:-1],Z,E,l2,U) < 0))]
        r0 = np.geomspace(dr,rf,npt)
        r1 = np.geomspace(rf,R,npt1)
        P0,Q0 = outward(r0,Z,E,l,U)
        P1,Q1 = inward(r1,Z,N,E,l,U)
        return r0,r1,P0,P1,Q0,Q1

    def count_nodes(y):
        return np.count_nonzero(y[1:] * y[:-1] < 0)

    while True:
        r0,r1,P0,P1,Q0,Q1 = shoot(E)
        m = count_nodes(P0) + count_nodes(P1)
        if m == n_nodes: break
        elif m > n_nodes: E *= 1.1
        else: E *= 0.9
            
    while True:
        d = P0[-1]/P1[0]
        q = trapz(P0**2, r0) + trapz(P1**2, r1) * d**2
        dE = (Q0[-1] - Q1[0]*d) * P0[-1] /(2*q)
        if np.abs(dE/E) <= EPS: break
        E += dE
        r0,r1,P0,P1,Q0,Q1 = shoot(E)

    q = q**0.5
    r = np.concatenate((r0, r1[1:]))
    P = np.concatenate((P0, P1[1:]*d))/q
    Q = np.concatenate((Q0, Q1[1:]*d))/q

    class result:
        def __init__(self):
            self.E = E
            self.n = n
            self.l = l
            self.r = r
            self.P = P
            self.Q = Q
            self.P_interp = CubicSpline(r,P)

        def WaveFunc(self, r):
            return np.where((r < self.r[0])|
                            (r > self.r[-1]),
                            0, self.P_interp(r))

    return result()
