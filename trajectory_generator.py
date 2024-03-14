import numpy as np

class TrajectoryGenerator():
    def __init__(self, reverse=False):
        """
        Initialize the TrajectoryGenerator.

        Parameters
        ----------
        reverse : bool, optional
            If Tr
            ue, the trajectory is generated in reverse. Default is False.
        """
        self.reverse= reverse
        
    def _step_forward(self, x, vf, h, sign, **kwargs):
        """
        Perform a single step forward using the fourth-order Runge-Kutta method.

        Parameters
        ----------
        x : np.ndarray
            Current state from which the step is taken.
        vf : function
            Vector field function that defines the system dynamics.
        h : float
            Step size.
        sign : int
            Determines the direction of the step based on time direction (forward or backward).
        **kwargs : dict
            Additional arguments, specifically for storing intermediate Runge-Kutta values.

        Returns
        -------
        np.ndarray
            The state of the system after taking a step forward.
        """
        kwargs['k'][0]=sign*vf(x)
        kwargs['y']=x + kwargs['k'][0]*h/2
        kwargs['k'][1]=sign*vf(kwargs['y'])
        kwargs['y']=x + kwargs['k'][1]*h/2
        kwargs['k'][2]=sign*vf(kwargs['y'])
        kwargs['y']=x + kwargs['k'][2]*h
        kwargs['k'][3]=sign*vf(kwargs['y'])
        return x + (kwargs['k'][0] + 2*kwargs['k'][1] + 2*kwargs['k'][2] + kwargs['k'][3])*h/6.0
    def generate_trajectory(self, x0, t, h):
        """
        Generates a trajectory for a system defined by a specific vector field, starting from an initial state
        and progressing over a specified time interval. The trajectory is constructed using the fourth-order 
        Runge-Kutta method for numerical integration, allowing for both forward and reverse time integration 
        based on the sign of the time parameter `t` and the object's `reverse` attribute.

        Parameters
        ----------
        x0 : np.ndarray
            The initial state of the system from which to start the trajectory. This array should be shaped 
            according to the dimensions of the system's state space (e.g., (dimensions,)).
        t : float
            The total time over which to generate the trajectory. The trajectory can be generated in the forward
            direction (positive `t`) or in reverse (negative `t`), with the actual direction also considering the 
            object's `reverse` attribute.
        h : float
            The time step size to use for each integration step along the trajectory. This value determines the 
            resolution of the trajectory and should be chosen based on the dynamics of the system and the desired
            accuracy of the trajectory.

        Returns
        -------
        Z : np.ndarray
            An array containing the sequence of states along the generated trajectory. The shape of this array will
            be (N,)+x0.shape, where N is the number of steps determined by the total time `t` and the step size `h`.
            Each entry in the array represents the state of the system at a consecutive time step.

        Notes
        -----
        The direction of time integration (forward or backward) is determined by both the sign of `t` and the
        `reverse` attribute of the class instance. The integration method employed (fourth-order Runge-Kutta)
        provides a balance between computational efficiency and the accuracy of the trajectory for a wide range
        of dynamical systems.
        """
        sign = 1-2*((t<0)^(self.reverse))
        if abs(t) < h:
            return x0[np.newaxis,...]
        N = int(abs(t)/h)
        Z = np.zeros((N,)+x0.shape)
        y = np.zeros_like(x0)
        k = np.zeros((4,)+x0.shape)
        Z[0] = x0
        for i in range(N-1):
            Z[i+1] = self._step_forward(Z[i], self.vector_field, h, sign, y=y, k=k)
        return Z

    
    def jacmat(self, p, epsilon=1e-7):
        """
        Calculate the Jacobian matrix of the vector field at point p using finite differences.

        Parameters
        ----------
        p : np.ndarray
            Point at which to calculate the Jacobian matrix, with shape (d, N).
        epsilon : float, optional
            Small perturbation used for finite difference calculation. Default is 1e-7.

        Returns
        -------
        output : np.ndarray
            The Jacobian matrix of the vector field at point p.
        """
        fd_coef = [1.0, -8.0, 0.0,    8.0, -1.0]
        assert p.ndim==2, "p.shape should be (d,N)"
        d = p.shape[0]
        N = p.shape[1]
        output = np.zeros((d,d,N))
        e = np.zeros((d,1))
        for i in range(d):
            e[i] = 1.0 
            output[:,i] = np.sum(np.array([c*self.vector_field(p + epsilon*(i-2)*e) for i,c in enumerate(fd_coef)]), axis=0)/(12*epsilon)
            e[i] = 0
        return output.transpose(2,0,1)
    
    def evaluate(self, x, t, h=1e-4):
        """
        Evaluate the final state of the system after evolving for a specified time from an initial state.
        
        This method calculates the system's state after progressing for time `t` starting from the initial state `x`. 
        It uses the fourth-order Runge-Kutta method to integrate the system's dynamics defined by `vector_field`. 
        The integration can proceed in the forward or backward direction in time depending on the sign of `t` and 
        the `reverse` attribute of the class. For `t` smaller than the step size `h`, it directly returns the initial state.
        
        Parameters
        ----------
        x : np.ndarray
            Initial state of the system. This array should match the expected dimensions for the system's state vector.
        t : float
            Total time over which to evolve the system. Positive values of `t` evolve the system forward in time, 
            while negative values evolve it backward. The actual direction of evolution also depends on the `reverse` 
            attribute of the class.
        h : float, optional
            Time step size to be used in the Runge-Kutta integration. Default value is 1e-4.

        Returns
        -------
        np.ndarray
            The state of the system after evolving for time `t` from the initial state `x`.

        """
        sign = 1-2*((t<0)^(self.reverse))
        if abs(t) < h:
            return x[np.newaxis,...]
        N = int(abs(t)/h)
        y = np.zeros_like(x)
        k = np.zeros((4,) + x.shape)
        for _ in range(N-1):
            x = self._step_forward(x, self.vector_field, h, sign, y=y, k=k)
        return x
        
class model_user_defined(TrajectoryGenerator):
    def __init__(self, func):
        super().__init__()
        self.func=func
    def vector_field(self, x):
        return self.func(x)

class VanderPol(TrajectoryGenerator): #Van der Pol osscilator
    '''
    x' = y
    y' = μ(1-x^2)y - x
    '''
    def __init__(self,mu):
        super().__init__()
        self.mu=mu
    def vector_field(self, x): #(2,*) --> (2,*)
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
    def vector_field(self, x): #(2,*) --> (2,*)
        return np.array([x[1], -self.delta*x[1] - self.alpha*x[0] - self.beta*x[0]**3])

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
    def vector_field(self, x): #(2,*) --> (2,*)
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
    def vector_field(self, x): #(2,*) --> (2,*)
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
    def vector_field(self, x):
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
    def vector_field(self, x): #(2,*) --> (2,*)
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
    def vector_field(self, x):
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
    def vector_field(self, x):
        return np.array([(x[2]- self.b)*x[0] - self.d*x[1], self.d*x[0] + (x[2] - self.b)*x[1],\
                                     self.c + self.a*x[2] - x[2]**3/3 - x[0]**2 + self.f*x[2]*x[0]])
    
class Linear(TrajectoryGenerator): #linear
    def __init__(self, A):
        super().__init__()
        self.A=A
    def vector_field(self, x):
        return (x.T@(self.A.T)).T















