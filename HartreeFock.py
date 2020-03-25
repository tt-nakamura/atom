# reference:
#   W. R. Johnson, "Atomic Structure Theory"
#     section 3.3

import numpy as np
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import CubicSpline
from Schroedinger import Schroedinger, inward, outward
from Lambda import Lambda

def integ(a,b,k=0,interp=False):
    """ Hartree-Fock integral
    int_0^infty Pa(s) Pb(s) min(r,s)**k / max(r,s)**(k+1) ds
    a,b = orbitals as output of Schroedinger
    if interp: return (r,V,f) object
      where r = np.array of radial coordinates in a
            V = np.array of integrals evaluated at r
            f = interpolation of V
    else: return V = array of integrals evaluated at r in a
    """
    r,P = a.r, a.P
    rmax = np.fmax(r, r[:,np.newaxis])
    rmax[0,0] = 1
    if a==b: f = P**2/rmax
    else:    f = P * b.WaveFunc(r) /rmax
    if k>0:
        rmin = np.fmin(r, r[:,np.newaxis])
        f *= (rmin/rmax)**k
    V = trapz(f,r)
    if not interp: return V

    class result:
        def __init__(self):
            self.r = r
            self.V = V
            self.V_interp = CubicSpline(r,V)

        def eval_at(self, r):
            return np.where(r > self.r[-1],
                            1/r, self.V_interp(r))

    return result()

def inhomog(a,E,R,Z,N,U):
    """ solve HF equation for orbital a
    E = guessed energy
    R = inhomogeneous terms (r.h.s) of HF equation
    Z,N = atomic nubmer, number of bound electrons
    U = screening potential for an electron
    updated orbital is returned
    """
    r,l = a.r, a.l
    P0,Q0 = outward(r,Z,E,l,U)
    P1,Q1 = inward(r,Z,N,E,l,U)
    k = len(r)//2
    W = P0[k]*Q1[k] - Q0[k]*P1[k] # Wronskian
    q0 = cumtrapz(P0*R, r, initial=0)
    q1 = cumtrapz(P1*R, r, initial=0)
    q1 = q1[-1] - q1
    P = (P1*q0 + P0*q1)/W
    Q = (Q1*q0 + Q0*q1)/W

    EPS = 1e-12
    while True: # normalize
        q = trapz(P*P, r) - 1
        if np.abs(q) < EPS: break
        q0 = cumtrapz(P0*P, r, initial=0)
        q1 = cumtrapz(P1*P, r, initial=0)
        q1 = q1[-1] - q1
        P2 = (P1*q0 + P0*q1)/W
        Q2 = (Q1*q0 + Q0*q1)/W
        dE = q/(4*trapz(P*P2, r))
        P -= 2*dE*P2
        Q -= 2*dE*Q2
        E += dE

    a.P = P
    a.Q = Q
    a.E = E
    a.P_interp = CubicSpline(r,P)

    return a

def V_exc(S,a):
    """ exchange potential
    S = list of orbitals
    a = orbital of exclusion
    """
    v = 0
    if a.w>1:
        for k in range(2, 2*a.l+1, 2):
            w = integ(a,a,k) * a.P
            v -= Lambda[a.l, k, a.l] * w
        v *= (4*a.l+2)/(4*a.l+1)*(a.w-1)

    for b in S:
        if b==a: continue
        u = 0
        for k in range(abs(a.l-b.l), a.l+b.l+1, 2):
            w = integ(a,b,k) * b.WaveFunc(a.r)
            u += Lambda[a.l, k, b.l] * w
        v -= b.w * u

    return v

def V_dir(S,r,a=None):
    """ direct potential
    S = list of orbitals
    r = np.array of radial coordinates
    a = orbital of exclusion
    """
    u = 0
    for b in S:
        v = integ(b,b, interp=True).eval_at(r)
        if a is None or b!=a: u += b.w * v
        else: u += (b.w-1) * v
    return u

def HartreeFock(Z,n,l,w=None, randomize=False):
    """ Z = atomic number
    n = list of principal quantum number for each orbital
    l = list of angular momentum for each orbital
    w = list of number of electrons in each orbital
    if w is None: all orbitals are filled with 4l+2 electrons
    if randomize: pick up random orbital in HF iteration
    return list of orbitals
    """
    if w is None: w = [4*l1+2 for l1 in l]
    N = sum(w) # total number of electrons

    EPS = 1e-4 if Z>2 else 1e-9
    eta = 0.5
    j = 0
    while True:
        S = []
        for i in range(len(n)):
            if j: s = Schroedinger(Z,n[i],l[i],N,U,E[i])
            else: s = Schroedinger(Z,n[i],l[i])
            s.w = w[i]
            S.append(s)

        E1 = E if j else 0
        E = np.array([s.E for s in S])
        P = np.array([s.P for s in S])
        r = np.array([s.r for s in S])
        print(E)
        if np.all(np.abs(E1/E-1) <= EPS): break

        U1 = U(r) if j else 0
        if j: dU = eta * (V_dir(S,r)*(N-1)/N - U1)
        else: dU = V_dir(S,r)*(N-1)/N
        E += trapz(dU*P**2, r)
        U1 += dU
        i = np.argmax(r[:,-1])
        r1 = r[i,-1]
        U1 = CubicSpline(r[i], U1[i])
        U = lambda r: np.where(r>r1, (N-1)/r, U1(r))
        j += 1

    if Z<=2: return S

    d = np.ones_like(E)
    EPS = 1e-9
    while np.any(d > EPS):
        i,i1 = np.argmax(d),i
        if randomize and i==i1: i = np.random.randint(len(n))
        R = V_exc(S,S[i]) + (V_dir(S,r[i],S[i]) - U(r[i]))*S[i].P
        dE = trapz(P[i]*R, r[i])/trapz(P[i]*S[i].P, r[i])
        E1 = S[i].E
        S[i] = inhomog(S[i], E[i]+dE, 2*R, Z,N,U)
        d[i] = np.abs(1 - E1/S[i].E)

        print(np.array([s.E for s in S]))

    return S

def E_total(S):
    """ total energy of an atom
    S = list of orbitals as output of HartreeFock
    refrence: R. D. Cowan
      "The Theory of Atomic Structure and Spectra" eq(6.39)
    """
    E = 0
    for a in S:
        E1 = 0
        if a.w>1:
            for k in range(2, 2*a.l+1, 2):
                y = a.P**2 * integ(a,a,k)
                E1 -= Lambda[a.l,k,a.l] * trapz(y, a.r)
            E1 *= (4*a.l+2)/(4*a.l+1)
            E1 += trapz(a.P**2 * integ(a,a), a.r)
            E1 *= a.w-1

        for b in S:
            if b==a: continue
            v = integ(b,b, interp=True).eval_at(a.r)
            R = trapz(a.P**2 * v, a.r)
            for k in range(abs(a.l-b.l), a.l+b.l+1, 2):
                y = a.P * b.WaveFunc(a.r) * integ(a,b,k)
                R -= Lambda[a.l,k,b.l] * trapz(y, a.r)
            E1 += b.w * R
            
        E += a.w*(a.E - E1/2)

    return E
