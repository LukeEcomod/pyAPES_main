# -*- coding: utf-8 -*-
"""
.. module: utils.utilities
    :synopsis: pyAPES component
.. moduleauthor:: Kersti Leppä, Samuli Launiainen

General utility functions for numerical solutions of 1D ODE's and PDE's
"""

import numpy as np

def forward_diff(y: np.ndarray, dx: float) -> np.ndarray:
    """
    Computes gradient dy/dx using forward difference method.
    
    Args:
        y (array): variable
        dx (float): grid size (must be constant)
    Returns (array):
        dy/dx: gradient

    """
    N = len(y)
    dy = np.ones(N) * np.NaN
    dy[0:-1] = np.diff(y)
    dy[-1] = dy[-2]

    return dy / dx


def central_diff(y, dx) -> np.ndarray:
    """
    Computes gradient dy/dx with central difference method

    Args:
        y (array): variable
        dx (float): grid size (must be constant)
    Returns (array):
        dy/dx: gradient

    """
    N = len(y)
    dydx = np.ones(N) * np.NaN
    # -- use central difference for estimating derivatives
    dydx[1:-1] = (y[2:] - y[0:-2]) / (2 * dx)
    # -- use forward difference at lower boundary
    dydx[0] = (y[1] - y[0]) / dx
    # -- use backward difference at upper boundary
    dydx[-1] = (y[-1] - y[-2]) / dx

    return dydx


def tridiag_fsm(Nvec, Nmax, a, b, c, r):
    '''
    Input:
    - Nvec: Vector length
    - Nmax: Maximum vector length
    - a: Below-diagonal matrix elements
    - b: Diagonal matrix elements
    - c: Above-diagonal matrix elements
    - r: Matrix equation rhs
        
    Output:
    - x: Solution vector
    '''

    x = np.zeros(Nmax)
    g = np.zeros(Nmax)
        
    beta = b[0]
    x[0] = r[0] / beta

    for n in range(1, Nvec):
        g[n] = c[n - 1] / beta
        beta = b[n] - a[n] * g[n]
        x[n] = (r[n] - a[n] * x[n - 1]) / beta

    for n in range(Nvec - 2, -1, -1):
        x[n] = x[n] - g[n + 1] * x[n + 1]

    return x

def tridiag(a: np.ndarray, b: np.ndarray, C: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Solves tridiagonal matrix using Thomas - algorithm
    a=subdiag, b=diag, C=superdiag, D=rhs
    """
    n = len(a)
    V = np.zeros(n)
    G = np.zeros(n)
    U = np.zeros(n)
    x = np.zeros(n)

    V[0] = b[0].copy()
    G[0] = C[0] / V[0]
    U[0] = D[0] / V[0]

    for i in range(1, n):  # nr of nodes
        V[i] = b[i] - a[i] * G[i - 1]
        U[i] = (D[i] - a[i] * U[i - 1]) / V[i]
        G[i] = C[i] / V[i]

    x[-1] = U[-1]
    inn = n - 2
    for i in range(inn, -1, -1):
        x[i] = U[i] - G[i] * x[i + 1]
    return x

def smooth(a: np.ndarray, WSZ: int) -> np.ndarray:
    """
    Smooths array by taking WSZ point moving average.
    Note: even WSZ is converted to next odd number.
    
    Args: 
        a (array): vector
        WSZ (int): number of points for moving average
    Returns
        x (array): smoothed vector of len(a)
    """
    WSZ = int(np.ceil(WSZ) // 2 * 2 + 1)
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    x = np.concatenate((start, out0, stop))
    return x

def spatial_average(y: np.ndarray, x: np.ndarray=None, method: str='arithmetic'):
    """
    Calculates spatial average of quantity y, from node points to soil compartment edges.

    Args: 
        y (array): quantity to average
        x (array): grid,<0, must be monotonically decreasing [m]
        method (str): 'arithmetic', 'geometric','dist_weighted'
    Returns: 
        f (array): averaged y, note len(f) = len(y) + 1
    """

    N = len(y)
    f = np.empty(N+1)  # Between all nodes and at surface and bottom

    if method == 'arithmetic':
        f[1:-1] = 0.5*(y[:-1] + y[1:])
        f[0] = y[0]
        f[-1] = y[-1]

    elif method == 'geometric':
        f[1:-1] = np.sqrt(y[:-1] * y[1:])
        f[0] = y[0]
        f[-1] = y[-1]

    #elif method is 'dist_weighted':  # not in use
    #    a = (x[0:-2] - x[2:])*y[:-2]*y[1:-1]
    #    b = y[1:-1]*(x[:-2] - x[1:-1]) + y[:-2]*(x[1:-1] - x[2:])
    #
    #    f[1:-1] = a / b
    #    f[0] = y[0]
    #    f[-1] = y[-1]

    return f

def lad_weibul(z: np.ndarray, LAI: float, h: float, hb: float=0.0,
               b: float=None, c: float=None, species: str=None) -> np.ndarray:
    """
    Generates leaf-area density profile from Weibull-distribution.

    SOURCE:
        Teske, M.E., and H.W. Thistle, 2004, A library of forest canopy structure for 
        use in interception modeling. Forest Ecology and Management, 198, 341-350. 
    Note: their formula is missing brackets for the scale param. Here their profiles are used between hb and h.
    Args:
        z (array): [m] monotonically increasing, constant step
        LAI (float): [m2 m-2] leaf-area index
        h (float): [m] canopy height
        hb (float): [m] crown base height
        b (float): Weibull shape parameter 1
        c (float): Weibull shape parameter 2
        species (str): 'pine', 'spruce', 'birch' to use table values for b, c

    Returns:
        LAD (array): [m2 m-3] leaf-area density

    """
    
    para = {'pine': [0.906, 2.145], 'spruce': [2.375, 1.289], 'birch': [0.557, 1.914]} 
    
    if (max(z) <= h) | (h <= hb):
        raise ValueError("h must be lower than uppermost gridpoint")
        
    if b is None or c is None:
        b, c = para[species]
    
    z = np.array(z)
    dz = abs(z[1]-z[0])
    N = np.size(z)
    LAD = np.zeros(N)

    a = np.zeros(N)

    # dummy variables
    ix = np.where( (z > hb) & (z <= h)) [0]
    x = np.linspace(0, 1, len(ix)) # normalized within-crown height

    # weibul-distribution within crown
    cc = -(c / b)*(((1.0 - x) / b)**(c - 1.0))*(np.exp(-((1.0 - x) / b)**c)) \
            / (1.0 - np.exp(-(1.0 / b)**c))

    a[ix] = cc
    a = np.abs(a / sum(a*dz))    

    LAD = LAI * a

    # plt.figure(1)
    # plt.plot(LAD,z,'r-')      
    return LAD

def lad_constant(z: np.ndarray, LAI: float, h: float, hb: float=0.0):
    """
    Creates uniform leaf-area density distribution.

    Args:
        z (array): [m] monotonically increasing, constant step
        LAI (float): [m2 m-2] leaf-area index
        h (float): [m] canopy height
        hb (float): [m] crown base height
    Returns:
        LAD (array): [m2 m-3] leaf-area density

    """
    if max(z) <= h:
        raise ValueError("h must be lower than uppermost gridpoint")

    z = np.array(z)
    dz = abs(z[1]-z[0])
    N = np.size(z)
    
    # dummy variables
    a = np.zeros(N)
    ix = np.where( (z > hb) & (z <= h)) [0]
    if ix.size == 0:
        ix = [1]

    a[ix] = 1.0
    a = a / sum(a*dz)
    LAD = LAI * a

    return LAD

def ludcmp(N, A, b):
    '''
    #
    Solve matrix equation Ax = b for x by LU decomposition
    #
    
    Args:
    N # Number of equations to solve
    A(N,N) # Matrix
    b(N) # RHS of matrix equation
    Out:
    x(N) # Solution of matrix equation

    integer :: i,ii,imax,j,k,ll,indx(N)

    real :: Acp(N,N),aamax,dum,sum,vv(N)
    '''

    Acp = A[:,:]
    x = b[:]

    vv = np.zeros(N)
    indx = np.zeros(N, dtype=int)

    for i in range(N):
        aamax = 0
        for j in range(N):
            if (abs(Acp[i,j]) > aamax):
                aamax = abs(Acp[i,j])
        vv[i] = 1/aamax

    for j in range(N):
        for i in range(j):
            sum = Acp[i,j]
            if (i > 1):
                for k in range(i):
                    sum -= Acp[i,k] * Acp[k,j]
                Acp[i,j] = sum
                    
        aamax = 0
        for i in range(j, N):
            sum = Acp[i,j]
            for k in range(j):
                sum -= Acp[i,k] * Acp[k,j]
            Acp[i,j] = sum

            dum = vv[i] * abs(sum)
            if dum >= aamax:
                imax = i
                aamax = dum
        if j != imax:
            for k in range(N):
                dum = Acp[imax, k]
                Acp[[imax, j], :] = Acp[[j, imax], :]
            vv[imax] = vv[j]

        indx[j] = imax
        if (Acp[j,j] == 0):
            Acp[j,j] = 1e-20
        if j != N-1:
            dum = 1 / Acp[j,j]
            for i in range(j+1, N):
                Acp[i,j] *= dum

    ii = 0
    for i in range(N):
        ll = indx[i]
        sum = x[ll]
        x[ll] = x[i]

        if ii != 0:
            for j in range(ii, i):
                sum -= Acp[i,j] * x[j]
        elif sum != 0:
            ii = i

        x[i] = sum

    for i in range(N-1, 0, -1):
        sum = x[i]
        for j in range(i+1, N):
            x[i] = sum / Acp[i,i]

    return x

# EOF