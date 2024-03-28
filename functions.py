import numpy as np
from scipy.special import comb
from scipy import integrate
from scipy import special
from itertools import combinations_with_replacement
from collections import Counter

def make_combinatorial_products(x):
    """
    Generate combinatorial products based on the input array x, useful for constructing polynomial feature spaces.

    Parameters
    ----------
    x : np.ndarray
        Input array of shape (n+1, d) + x.shape[2:], where n is the polynomial degree and d is the dimension.

    Returns
    -------
    np.ndarray
        Array of combinatorial products with shape (comb(n+d, d),) + x.shape[2:].
    """

    d=x.shape[1]
    n=x.shape[0]-1
    output=np.ones((comb(n+d,d, exact = True),) + x.shape[2:], dtype=x.dtype)
    for i,ell in enumerate(combinations_with_replacement(list(range(d+1)),n)):
        temp=Counter(ell)
        for j in range(1,d+1):
            output[i]=output[i]*x[temp[j],j-1]
    return output

def constV_exp(X, p, n, sigma, b=None):
    """
    Construct V^n_X using the exponential kernel for given data points, a center point, and a bandwidth.

    Parameters
    ----------
    X : np.ndarray
        Data points of shape (d, n1, ..., nr).
    p, b : np.ndarray
        Center point of shape (d, 1). If b is None, b is set to p.
    n : int
        Degree of the polynomial space.
    sigma : float
        Bandwidth of the exponential kernel.

    Returns
    -------
    np.ndarray
        Constructed V^n_X array with shape (comb(n+d, d),) + X.shape[1:].
    """

    d=X.shape[0]
    if b is None: #if b is omitted, we set b = p
        b=p
    V_nonflat=np.ones((n+1,) + X.shape)
    factor=((X.T-p.T)/sigma).T #shape=X.shape, transposes are just for broadcasting
    for i in range(n):
        V_nonflat[i+1]=V_nonflat[i]*factor
    V_nonflat*=np.exp((2*X.T - p.T - b.T)*(p-b).T/(2*sigma**2)).T
    return make_combinatorial_products(V_nonflat) #shape = (comb(n+d,d),) + X.shape[1:]

def constW_exp(X, Y, p, n, sigma, b=None):
    """
    Construct W^n_{X,Y} using the exponential kernel for given data points, output points, a center point, degree, and bandwidth.

    Parameters
    ----------
    X, Y : np.ndarray
        Data points and output points of shape (d, n1, ..., nr).
    p : np.ndarray
        Center point of shape (d, 1).
    n : int
        Degree of the polynomial space.
    sigma : float
        Bandwidth of the exponential kernel.
    b : np.ndarray, optional
        Base point for the kernel, defaults to p if None.

    Returns
    -------
    np.ndarray
        Constructed W^n_{X,Y} array.
    """
    d=X.shape[0]
    if b is None: #if b is omitted, we set b = p
        b=p
    V_nonflat=np.ones((n+1,) + X.shape)
    dV_nonflat=np.zeros((n+1,) + X.shape)
    factor=((X.T-p.T)/sigma).T
    for i in range(n):
        V_nonflat[i+1]=V_nonflat[i]*factor
    V_nonflat*=np.exp((2*X.T - p.T - b.T)*(p-b).T/(2*sigma**2)).T
    #compute the derivative of v's
    dV_nonflat+=(V_nonflat.T*((p-b)/sigma)).T
    dV_nonflat[1:]+=(V_nonflat[:n].T*np.arange(1,n+1)).T 
    dV_nonflat /= sigma
    temp=np.copy(V_nonflat[:n+1])
    #
    W=np.zeros((comb(n+d,d, exact=True),) + X.shape[1:])
    for j in range(d):
        temp[:,j] = np.copy(dV_nonflat[:,j])
        W += make_combinatorial_products(temp)*Y[j]
        temp[:,j] = np.copy(V_nonflat[:,j])
    return W

def constDV_exp(X, p, n, sigma, b=None):
    """
    Construct the derivative of V^n_X using the exponential kernel.

    Parameters
    ----------
    X : np.ndarray
        Data points of shape (d, n1, ..., nr).
    p : np.ndarray
        Center point of shape (d, 1).
    n : int
        Degree of the polynomial space.
    sigma : float
        Bandwidth of the exponential kernel.
    b : np.ndarray, optional
        Base point for the kernel, defaults to p if None.

    Returns
    -------
    np.ndarray
        Derivative of V^n_X array.
    """

    d=X.shape[0]
    if b is None: #if b is omitted, we set b = p
        b=p
    V_nonflat=np.ones((n+1,) + X.shape)
    dV_nonflat=np.zeros((n+1,) + X.shape)
    factor=((X.T-p.T)/sigma).T
    for i in range(n):
        V_nonflat[i+1]=V_nonflat[i]*factor
    V_nonflat*=np.exp((2*X.T - p.T - b.T)*(p-b).T/(2*sigma**2)).T
    #compute the derivative of v's
    dV_nonflat+=(V_nonflat.T*((p-b)/sigma)).T
    dV_nonflat[1:]+=(V_nonflat[:n].T*np.arange(1,n+1)).T 
    dV_nonflat /= sigma
    temp=np.copy(V_nonflat[:n+1])
    #
    dV=np.zeros((d, comb(n+d,d, exact=True),) + X.shape[1:])
    for j in range(d):
        temp[:,j] = np.copy(dV_nonflat[:,j])
        dV[j] = make_combinatorial_products(temp)
        temp[:,j] = np.copy(V_nonflat[:,j])
    return dV

def constV_gauss(X, p, n, sigma, *arg):
    """
    Construct V^n_X using the Gaussian kernel.

    Parameters
    ----------
    X : np.ndarray
        Data points of shape (d, n1, ..., nr).
    p : np.ndarray
        Center point of shape (dim,).
    n : int
        Degree of the polynomial space.
    sigma : float
        Bandwidth of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Constructed V^n_X array.
    """

    d=X.shape[0]
    V_nonflat=np.ones((n+1,) + X.shape)
    factor=((X.T-p.T)/sigma).T
    for i in range(n):
        V_nonflat[i+1]=V_nonflat[i]*factor
    V_nonflat*=np.exp((-factor**2).T/2).T
    return make_combinatorial_products(V_nonflat)

def constW_gauss(X, Y, p, n, sigma, *arg):
    """
    Construct W^n_{X,Y} using the Gaussian kernel.

    Parameters
    ----------
    X, Y : np.ndarray
        Data points and output points.
    p : np.ndarray
        Center point.
    n : int
        Degree of the polynomial space.
    sigma : float
        Bandwidth of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Constructed W^n_{X,Y} array.
    """

    d=X.shape[0]
    V_nonflat=np.ones((n+2,) + X.shape)
    dV_nonflat=np.zeros((n+1,) + X.shape)
    factor=((X.T-p.T)/sigma).T
    for i in range(n+1):
        V_nonflat[i+1]=V_nonflat[i]*factor
    V_nonflat*=np.exp((-factor**2).T/2).T
    dV_nonflat[1:]+=(V_nonflat[:n].T*np.arange(1,n+1)).T 
    dV_nonflat -= V_nonflat[1:]
    dV_nonflat /= sigma
    temp=np.copy(V_nonflat[:n+1])
    W=np.zeros((comb(n+d,d, exact=True),) + X.shape[1:])
    for j in range(d):
        temp[:,j] = np.copy(dV_nonflat[:,j])
        W += make_combinatorial_products(temp)*Y[j]
        temp[:,j] = np.copy(V_nonflat[:n+1,j])
    return W

def constDV_gauss(X, p, n, sigma, *arg):
    """
    Construct the derivative of V^n_X using the Gaussian kernel.

    Parameters
    ----------
    X : np.ndarray
        Data points.
    p : np.ndarray
        Center point.
    n : int
        Degree of the polynomial space.
    sigma : float
        Bandwidth of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Derivative of V^n_X array.
    """

    d=X.shape[0]
    V_nonflat=np.ones((n+2,) + X.shape)
    dV_nonflat=np.zeros((n+1,) + X.shape)
    factor=((X.T-p.T)/sigma).T
    for i in range(n+1):
        V_nonflat[i+1]=V_nonflat[i]*factor
    V_nonflat*=np.exp((-factor**2).T/2).T
    dV_nonflat[1:]+=(V_nonflat[:n].T*np.arange(1,n+1)).T 
    dV_nonflat -= V_nonflat[1:]
    dV_nonflat /= sigma
    temp=np.copy(V_nonflat[:n+1])
    dV=np.zeros((d, comb(n+d,d,exact=True),) + X.shape[1:])
    for j in range(d):
        temp[:,j] = np.copy(dV_nonflat[:,j])
        dV[j] = make_combinatorial_products(temp)
        temp[:,j] = np.copy(V_nonflat[:n+1,j])
    return dV
    
def make_Gs(n, ps, sigma, b=None, deg=20):
    """
    Compute G_{ij} matrices using Hermite-Gaussian quadrature.

    Parameters
    ----------
    n : int
        Degree of the polynomial space.
    ps : np.ndarray
        Points at which G matrices are computed.
    sigma : float
        Bandwidth parameter.
    b : np.ndarray, optional
        Base point for the kernel, defaults to zeros if None.
    deg : int, optional
        Degree of the Hermite-Gaussian quadrature.

    Returns
    -------
    np.ndarray
            Computed G_{ij} matrices.
    """

    d=ps.shape[0]
    r=ps.shape[1]
    if b==None:
        b=np.zeros((d,1))
    ps=ps-b
    rn = comb(n+d,d, exact=True)
    output = np.ones((r,rn,r,rn))
    x,w = special.roots_hermite(deg)
    z=x.reshape(1,1,deg,1) + x.reshape(1,1,1,deg)*1j
    for k in range(d):
        temp = np.zeros((r,rn,r,rn,deg,deg),dtype='complex128')
        for i1,ell1 in enumerate(combinations_with_replacement(list(range(d+1)),n)):
            alpha=Counter(ell1)
            for i2,ell2 in enumerate(combinations_with_replacement(list(range(d+1)),n)):
                beta=Counter(ell2)
                temp[:,i1,:,i2]=((z - ps[k].reshape(r,1,1,1)/sigma)**alpha[k+1])*((z.conjugate() - ps[k].reshape(1,r,1,1)/sigma)**beta[k+1])
                temp[:,i1,:,i2]*=np.exp(z*ps[k].reshape(r,1,1,1)/sigma + z.conjugate()*ps[k].reshape(1,r,1,1)/sigma)
                temp[:,i1,:,i2]*=np.exp((-ps[k].reshape(r,1,1,1)**2 - ps[k].reshape(1,r,1,1)**2)/(2*sigma**2))
        output *= ((temp.dot(w)).dot(w)).real/np.pi
    return output

def jacmat(x, f, epsilon=1e-7):
    """
    Compute the Jacobian matrix of a function f at points x.
    
    Parameters
    ----------
    x : np.ndarray
        Points at which to compute the Jacobian, shape (d, N).
    f : function
        Function for which the Jacobian is computed.
    epsilon : float, optional
        Perturbation for finite differences.
    
    Returns
    -------
    np.ndarray
        Jacobian matrix of f at x, shape (N, d, d).
    """
    
    d=x.shape[0]
    N=x.shape[1]
    output = np.zeros((d,d,N))
    e=np.zeros((d,1))
    fd_coef = np.array([1, -8, 0, 8, -1], dtype=float)
    for i in range(d):
        e[i,0] = 1.0 
        output[:,i] = np.sum(np.array([c*f(x + (i-2)*epsilon*e) for i,c in enumerate(fd_coef)]), axis=0)/(12*epsilon)
        e[i,0] = 0.0
    return output.transpose(2,0,1) #shape of output is (N,d,d)