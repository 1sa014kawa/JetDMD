import numpy as np
from scipy.special import comb
from scipy import integrate
from scipy import special
from itertools import combinations_with_replacement
from collections import Counter

# Construct combinatorial products, e.g. [[1,1],[x,y],[x^2,y^2]] --> [1,x,y,x^2, xy, y^2]
def make_combinatorial_products(x): #x.shape = (n+1, d) + x.shape[2:] --> ndarray of shape (comb(n+d,d)) + x.shape[2:]
  d=x.shape[1]
  n=x.shape[0]-1
  output=np.ones((comb(n+d,d, exact = True),) + x.shape[2:], dtype=x.dtype)
  for i,ell in enumerate(combinations_with_replacement(list(range(d+1)),n)):
    temp=Counter(ell)
    for j in range(1,d+1):
      output[i]=output[i]*x[temp[j],j-1]
  return output

# Construct V_n^X using the exponential kernel exp((x-b)^T (y-b) / sigma^2) and the orthogonal basis in Example 7.1
def constV_exp(X,p,n,sigma, b=None): #p.shape=b.shape=(d,1) X.shape=(d,n1,...,nr)
  d=X.shape[0]
  if b is None: #if b is omitted, we set b = p
    b=p
  V_nonflat=np.ones((n+1,) + X.shape)
  factor=((X.T-p.T)/sigma).T #shape=X.shape, transposes are just for broadcasting
  for i in range(n):
    V_nonflat[i+1]=V_nonflat[i]*factor
  V_nonflat*=np.exp((2*X.T - p.T - b.T)*(p-b).T/(2*sigma**2)).T
  return make_combinatorial_products(V_nonflat) #shape = (comb(n+d,d),) +  X.shape[1:]

# Construct W_n^{X,Y} using the exponential kernel exp((x-b)^T (y-b) / sigma^2) and the orthogonal basis in Example 7.1
def constW_exp(X, Y, p, n, sigma, b=None): #p.shape=(d,1), X.shape=Y.shape = (d,n1,...,nr)
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

# Construct \partial V_n^X using the exponential kernel exp((x-b)^T (y-b) / sigma^2) and the orthogonal basis in Example 7.1
def constDV_exp(X, p, n, sigma, b=None): #p.shape=(d,1), X.shape=Y.shape = (d,n1,...,nr)
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
  return W

# Construct V_n^X using the Gaussian kernel exp(|x-y|^2 / 2sigma^2) and the orthogonal basis in Example 7.2
def constV_gauss(X,p,n,sigma, *arg): #p.shape=(dim,)
  d=X.shape[0]
  V_nonflat=np.ones((n+1,) + X.shape)
  factor=((X.T-p.T)/sigma).T
  for i in range(n):
    V_nonflat[i+1]=V_nonflat[i]*factor
  V_nonflat*=np.exp((-factor**2).T/2).T
  return make_combinatorial_products(V_nonflat)

# Construct W_n^{X,Y} using the Gaussian kernel exp(|x-y|^2 / 2sigma^2) and the orthogonal basis in Example 7.2
def constW_gauss(X, Y, p, n, sigma, *arg):
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

# Construct nabla of V_n^X using the Gaussian kernel exp(|x-y|^2 / 2sigma^2) and the orthogonal basis in Example 7.2
def constDV_gauss(X, p, n, sigma, *arg):
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
  
# Computation of G_{ij} using the Hermite-Gaussian quadrature
def make_Gs(n,ps,sigma,b=None,deg=20):
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

# Computation of Jacobian matrix
def jacmat(x, f, epsilon=1e-7): #x.shape = (d,N)
    d=x.shape[0]
    N=x.shape[1]
    output = np.zeros((d,d,N))
    e=np.zeros((d,1))
    fd_coef = np.array([1, -8, 0,  8, -1], dtype=float)
    for i in range(d):
      e[i,0] = 1.0 
      output[:,i] = np.sum(np.array([c*f(x + (i-2)*epsilon*e) for i,c in enumerate(fd_coef)]), axis=0)/(12*epsilon)
      e[i,0] = 0.0
    return output.transpose(2,0,1) #shape of output is (N,d,d)