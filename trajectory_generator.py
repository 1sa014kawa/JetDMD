import numpy as np

class TrajectoryGenerator():
  def __init__(self, reverse=False):
    self.reverse=reverse
  def _step_forward(self, x, vf, h, sign, **kwargs): #Runge-Kutta method
    kwargs['k'][0]=sign*vf(x)
    kwargs['y']=x + kwargs['k'][0]*h/2
    kwargs['k'][1]=sign*vf(kwargs['y'])
    kwargs['y']=x + kwargs['k'][1]*h/2
    kwargs['k'][2]=sign*vf(kwargs['y'])
    kwargs['y']=x + kwargs['k'][2]*h
    kwargs['k'][3]=sign*vf(kwargs['y'])
    return x + (kwargs['k'][0] + 2*kwargs['k'][1] + 2*kwargs['k'][2] + kwargs['k'][3])*h/6.0
  def generate_trajectory(self, x0, t, h, vf=None): #output.shape =  (N,) + x0.shape
    if vf==None:
      vf = self._vf 
    sign = 1-2*((t<0)^(self.reverse))
    if abs(t) < h:
      return x0[np.newaxis,...]
    N=int(abs(t)/h)
    Z=np.zeros((N,)+x0.shape) 
    y=np.zeros_like(x0)
    k=np.zeros((4,)+x0.shape)
    Z[0]=x0
    for i in range(N-1):
      Z[i+1]=self._step_forward(Z[i], vf, h, sign, y=y, k=k)
    return Z
  def jacmat_vf(self, p, epsilon=1e-7, vf=None):
    fd_coef = [1.0, -8.0, 0.0,  8.0, -1.0]
    if vf==None:
      vf = self.vector_field
    assert p.ndim==2, "p.shape should be (d,N)"
    d=p.shape[0]
    N=p.shape[1]
    output = np.zeros((d,d,N))
    # e=np.identity(d).reshape(d,d,1)
    # output = (vf(p[:,np.newaxis] + epsilon*e) - vf(p[:,np.newaxis] - epsilon*e))/(2*epsilon)
    e=np.zeros((d,1))
    for i in range(d):
      e[i] = 1.0 
      output[:,i] = np.sum(np.array([c*vf(p + epsilon*(i-2)*e) for i,c in enumerate(fd_coef)]), axis=0)/(12*epsilon)
      e[i] = 0
    return output.transpose(2,0,1)
  def evaluate(self, x, t, h=1e-4, vf=None):
    if vf==None:
      vf = self.vector_field
    return self.generate_trajectory(x, t, h)[-1]
  def vector_field(self, x, vf=None):
    if vf==None:
      vf = self._vf
    return vf(x)*(1-2*self.reverse)
    
class model_user_defined(TrajectoryGenerator):
  def __init__(self, func):
    super().__init__()
    self.func=func
  def _vf(self, x):
    return self.func(x)

class VanderPol(TrajectoryGenerator): #Van der Pol osscilator
  '''
  x' = y
  y' = μ(1-x^2)y - x
  '''
  def __init__(self,mu):
    super().__init__()
    self.mu=mu
  def _vf(self, x): #(2,*) --> (2,*)
    return np.array([x[1], self.mu*(1-x[0]**2)*x[1] - x[0]])


class Duffing(TrajectoryGenerator): #Van del Pol osscilator
  '''
  x' = y
  y' = -δy-αx-βx^3 (+ γcos(ωt))
  '''
  def __init__(self, alpha,beta,delta):
    super().__init__()
    self.alpha=alpha
    self.beta=beta
    self.delta=delta
  def _vf(self, x): #(2,*) --> (2,*)
    return np.array([x[1], -self.delta*x[1] - self.alpha*x[0] - self.beta*x[0]**3])
  # def jacmat_vf(self, p):
    # N=p.shape[1]
    # return np.array([[np.zeros(N), np.ones(N)],[-self.alpha-3*self.beta*p[0]**2, -self.delta*np.ones(N)]]).transpose(2,0,1)

class StuartLandau(TrajectoryGenerator): #Stuart_Landau equation
  '''
  x' = ax - ωy - (x - by)(x^2 + y^2)
  y' = ay + ωx - (y + bx)(x^2 + y^2)
  '''
  def __init__(self,a,b,omega):
    super().__init__()
    self.a=a
    self.b=b
    self.omega=omega
  def _vf(self, x): #(2,*) --> (2,*)
    return np.array([self.a*x[0] - self.omega*x[1]- (x[0] - self.b*x[1])*(x[0]**2 + x[1]**2 ), \
                     self.a*x[1] + self.omega*x[0]- (x[1] + self.b*x[0])*(x[0]**2 + x[1]**2)])

class FitzHughNagumo(TrajectoryGenerator): #FitzHugh-Nagumo equation
  def __init__(self, a, b, tau, R, I):
    super().__init__()
    self.a=a
    self.b=b
    self.tau=tau
    self.R=R
    self.I=I
  def _vf(self, x): #(2,*) --> (2,*)
    return np.array([x[0] - x[0]**3/3 - x[1] + self.R*self.I \
                     (x[0] + self.a - self.b*x[1])/self.tau])

class Lorenz(TrajectoryGenerator): #Lorenz attractor
  '''
  x = σ(y - x)
  y = x(ρ - z) - y
  z = xy - βz
  '''
  def __init__(self, sigma=10.0, rho=28.0, beta=8/3):
    super().__init__()
    self.sigma=sigma
    self.rho=rho
    self.beta=beta
  def _vf(self, x):
    return np.array([self.sigma*(x[1]-x[0]), x[0]*(self.rho - x[2]) - x[1], x[0]*x[1] - self.beta*x[2]])

class BogdanovTakens(TrajectoryGenerator): 
  '''
  x' = y
  y' = β_1 + β_2x + x^2 + sign xy
  '''
  def __init__(self, beta1, beta2, sign):
    super().__init__()
    self.beta1=beta1
    self.beta2=beta2
    self.sign=sign
  def _vf(self, x): #(2,*) --> (2,*)
    return np.array([x[1], self.beta1 + self.beta2*x[0] + x[0]**2 + self.sign*x[0]*x[1]])

class Rossler(TrajectoryGenerator): #Lorenz attractor
  '''
  x' = - y - z
  y' = x + ay
  z' = b + z(x-c)
  '''
  def __init__(self, a,b,c):
    super().__init__()
    self.a=a
    self.b=b
    self.c=c
  def _vf(self, x):
    return np.array([-x[1]-x[2], x[0] + self.a*x[1], self.b + x[2]*(x[0] - self.c)])
    
class Aizawa(TrajectoryGenerator): #Lorenz attractor
  '''
  x' = (z-b)x - dy
  y' = dx + (z-b)y
  z' = c + az - z^3/3 - x^2 + fzx^3
  '''
  def __init__(self, a,b,c,d,e,f):
    super().__init__()
    self.a=a
    self.b=b
    self.c=c
    self.d=d
    self.e=e
    self.f=f
  def _vf(self, x):
    return np.array([(x[2]- self.b)*x[0] - self.d*x[1], self.d*x[0] + (x[2] - self.b)*x[1],\
                     self.c + self.a*x[2] - x[2]**3/3 - x[0]**2 + self.f*x[2]*x[0]])
  
class Linear(TrajectoryGenerator): #linear
  def __init__(self, A):
    super().__init__()
    self.A=A
  def _vf(self, x):
    return (x.T@(self.A.T)).T















