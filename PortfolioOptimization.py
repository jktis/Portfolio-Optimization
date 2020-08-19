from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
import numpy as np 
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import warnings 


def utility_transcosts(d,mu,k,V,g,x0,n):
    """Negative mean-variance utility function with transaction costs"""
    return -d[:n]@mu + d[n:]@mu - x0@mu + d[:n]@k + d[n:]@k + \
        (d[:n]@V@d[:n] - 2*d[:n]@V@d[n:] + 2*d[:n]@V@x0 + d[n:]@V@d[n:] - 2*d[n:]@V@x0 + x0@V@x0)*g/2


def constr_d(n,d):
    return d[:n]@d[n:]


def jac_c(n,d):
    return np.append(d[n:],d[:n])


def hessian_c(n,d,v):
    return np.vstack([np.hstack([np.zeros((n,n)),np.eye(n)]),np.hstack([np.eye(n),np.zeros((n,n))])])


def constr_transcosts(n,x0,nonlin_constr=True,supply_H_and_jac=True):
    bounds = Bounds(np.array([0]*(2*n)), np.append(1-x0,x0) if nonlin_constr else np.inf)
    
    A = np.append(np.ones((1,n)),np.ones((1,n))*-1)
    lb,ub = 0,0
    if not nonlin_constr:
        lincomb_bound = np.zeros((n,2*n))
        for i in range(n):
            lincomb_bound[i,i] = 1
            lincomb_bound[i,i+n] = -1
        
        A = np.vstack([A[None,:],lincomb_bound])
        lb = np.append(lb,-x0)
        ub = np.append(ub,1-x0)
        
    constr = [LinearConstraint(A,lb,ub)]

    if nonlin_constr:
        constr.append( NonlinearConstraint(lambda x: constr_d(n,x),0,0,
                                           **({} if not supply_H_and_jac \
                                              else {'jac': lambda x: jac_c(n,x),
                                                    'hess': lambda x,v: hessian_c(n,x,v)})) 
        )
    
    return {'bounds': bounds, 'constraints': constr}


def jac_trcost(d,mu,k,V,g,x0,n):
    return np.append(-mu.T + k + g*(d[:n]-d[n:]+x0)@V, mu.T + k - g*(d[:n]-d[n:]+x0)@V)


def H_trcost(d,mu,k,V,g,x0,n):
    return g*np.vstack([np.hstack([V,-V]),np.hstack([-V,V])])


def random_change(x0,alpha):
    allo = np.random.dirichlet(alpha)
    dx = allo - x0
    
    dplus = np.zeros(len(x0))
    dplus[dx>0] = dx[dx>0]

    dminus = np.zeros(len(x0))
    dminus[dx<0] = dx[dx<0]*-1

    return np.append(dplus,dminus) 


class OptP:

    def __init__(self,exp_return,tr_cost,cov,risk_aversion,start_allocation):
        """
        Class for mean-variance portfolio optimization under proportional transaction
        costs.

        ..math:: \min_{x} \frac{\gamma}{2}x'\Sigma x + k'|x^+ + x^-| - \mu'x 

        subject to `x = x^+ - x^- + x_0`, `1'x = 1`, `0<=x<=1` and `x^+, x^- >=0` 
        (and optionally x^+'x^- = 0). `x_0` is the start allocation.

        Parameters:
        -----------
        exp_return : numpy.array
            One dimensional array of n assets' expected returns.
        tr_cost : numpy.array
            One dimensional array of n assets' transaction costs.
        cov : numpy.array
            (n x n) return covariance matrix.
        risk_aversion : float
            Investor risk aversion.
        start_allocation : numpy.array
            Current assets' portfolio weights. 
        
        """

        self.mu = exp_return 
        self.k = tr_cost 
        self.V = cov
        self.g = risk_aversion
        self.x0 = start_allocation 


    def generate_qp_matrices(self):
        """Generate matrices used by the cvxopt solver."""
        
        m = self.x0.shape[0]
        n = m*2
        Zero = np.zeros(self.V.shape)
        
        self.P = matrix(np.c_[np.r_[self.V,self.V], np.r_[self.V,self.V]]*self.g)

        Q = np.c_[np.r_[self.V,Zero], np.r_[Zero,self.V]]
        self.q = matrix(self.g*np.append(self.x0,self.x0)@Q+np.append(self.k,-self.k)-np.append(self.mu,self.mu))
        
        upper_bound = np.zeros((m,n))
        for i in range(m):
            upper_bound[i,i] = 1
            upper_bound[i,i+m] = 1
        
        lower_bound = upper_bound*-1
        
        self.G = matrix(np.r_[upper_bound,
                              lower_bound,
                              np.diag(np.append(np.ones(m)*-1,np.ones(m)))]
        )
                
        self.h = matrix(np.r_[1-self.x0,self.x0,np.zeros(n)])
        self.A = matrix(1.0,(1,n))
        self.b = matrix(0.0)
        
        return 


    def optimize_p(self,solver='scipy',**kwargs):
        """
        Optimize portfolio.
        
        Parameters
        ----------
        solver = str (default 'scipy')
            If 'scipy', uses scipy's minimize function. Allows handling
            of non-linear constraints. If 'qp', uses cvxopt's solver 
            function. A lot faster than scipy, but cannot handle non-linear
            contraints.
        krawgs : dict
            Optional keyword arguments passed to the solver function.

        Returns
        -------
        None. Result is provided in the `solution` attribute of the class.
        
        """

        if solver=='scipy':
            self.opt_scipy(**kwargs)
        elif solver=='qp':
            if kwargs.get('nonlin_constr')==True:
                raise ValueError("Non-linear constraints not compatible with quadratic program.")

            self.generate_qp_matrices()

            kwargs.update({'nonlin_constr': False})
            self.opt_cvx(**kwargs)
        else:
            raise ValueError(f"{solver} not a valid method; use 'scipy' or 'qp'")

        return

    def opt_cvx(self,**kwargs):
        """
        Solve optimization problem using CVXOPT solver for 
        quadratic programs. 

        If called standalone (not via `optimize_p`), the P, q, 
        G, h, A and b matrices (see CVXOPT documentation) must
        be provided as class attributes first.

        Parameters
        ----------
        default_to_scipy : bool (optional)
            If True, use scipy solver if cvxopt solver throws an error.
        kwargs : dict
            Rest of the optional keyword arguments are passed to scipy
            solver, if used.

        Returns
        -------
        None. Result is provided in the `solution` attribute of the class.

        """

        try:
            sol = solvers.qp(self.P,self.q,self.G,self.h,self.A,self.b)
            if 'optimal' not in sol['status']:
                raise ValueError("Method did not converge")
        except Exception as e:
            warnings.warn(f"CVXOPT solver unsuccessful; default to scipy `minimize`; Error raised: {e}")
            if kwargs.get('default_to_scipy',False):
                self.opt_scipy(**kwargs)
            else: 
                self.solution = {'dx': np.nan, 'x': np.nan}
        else: 
            dx_star = np.array(sol['x']).reshape(2,-1)
            x_star = self.x0 + dx_star.sum(axis=0)
            self.solution = {'dx': dx_star, 'x': x_star}

        return  


    def opt_scipy(self,init='random',n_repeat=10,alpha=1,nonlin_constr=True,supply_H_and_jac=True):
        """
        Solve optimization problem using scipy minimize.

        Parameters
        ----------
        init : str or numpy.ndarray (default 'random')
            Starting value for the optimization. If 'at_zero',
            the starting value for the change in allocation is set
            to zero for all assets. If 'random', the optimization
            is initialized at a random starting point in addition
            to an initialization at a zero change. A custom starting
            point can be provided by passing a numpy array of shape
            (2*n,) where n is the number of assets. The first n 
            elements are the portfolio increases (x^+), the latter
            n elements are portfolio decreases (x^-).
        n_repeat : int
            Only relevent if `init='random'`. Number of times the 
            optimization is performed at different random starting
            points.
        alpha : float (>0)
            Only relevent if `init='random'`. Starting point for
            the optimization is given by the difference between a
            vector of random portfolio allocations and the initial
            portfolio weights (x_0). The random allocation is drawn
            from a Dirichlet distribution with parameters (alpha, ...
            alpha).
        nonlin_constr : bool (default True)
            If True, the optimization takes the constraint `x^+'x^-=0`
            into account. That is, simulatenous allocation increases and
            decreases are not allowed.
        supply_H_and_jac : bool (default True)
            If True, the analytical Hessian and jacobian for the optimization
            problem and the nonlinear contraint (if chosen) are passed. If False,
            scipy computes them numerically.

        Returns
        -------
        None. Result is provided in the `solution` attribute of the class.

        """

        n = len(self.mu)
        if init == 'random' or init=='at_zero':

            # initalize with zero change
            res = minimize(utility_transcosts,[0]*2*n,\
                            args=(self.mu,self.k,self.V,self.g,self.x0,n),\
                            method='trust-constr',\
                            **constr_transcosts(n,self.x0,nonlin_constr=nonlin_constr,supply_H_and_jac=supply_H_and_jac),
                            **({} if not supply_H_and_jac else \
                               {'jac': jac_trcost,
                                'hess': H_trcost})
            )


            u_init = utility_transcosts(res.x,self.mu,self.k,self.V,self.g,self.x0,n)
            dx_init = res.x
            
            if init == 'random':
                for i in range(n_repeat):
                    d_init = random_change(self.x0, [alpha]*n)

                    res = minimize(utility_transcosts,d_init,\
                                args=(self.mu,self.k,self.V,self.g,self.x0,n),\
                                method='trust-constr',\
                                **constr_transcosts(n,self.x0,nonlin_constr=nonlin_constr,supply_H_and_jac=supply_H_and_jac),
                                **({} if not supply_H_and_jac else \
                                   {'jac': jac_trcost,
                                    'hess': H_trcost})
                    )


                    u_new = utility_transcosts(res.x,self.mu,self.k,self.V,self.g,self.x0,n)
                    if u_new < u_init:
                        u_init = u_new
                        dx_init = res.x

            dx_star = dx_init.reshape(2,-1)
            dx_star[1,:] *= -1
            x_star = self.x0 + dx_star.sum(axis=0)

        else:
            res = minimize(utility_transcosts,init,\
                           args=(self.mu,self.k,self.V,self.g,self.x0,n),\
                           method='trust-constr',\
                           **constr_transcosts(n,self.x0,nonlin_constr=nonlin_constr,supply_H_and_jac=supply_H_and_jac),
                            **({} if not supply_H_and_jac else \
                               {'jac': jac_trcost,
                                'hess': H_trcost})
            )
                    

            dx_star = res.x.reshape(2,-1)
            dx_star[1,:] *= -1
            x_star = self.x0 + dx_star.sum(axis=0)

        self.solution = {'dx': dx_star, 'x': x_star}
        return 


    def utility(self,dx):
        """
        Utility in terms of allocation changes.
        
        Parameters:
        -----------
        dx : numpy.array
            (2,n) array of allocation changes of n assets. First row refers 
            to allocation increases, second row to decreases. Accordingly, 
            the first row should contain positive values only, the second row
            only negative values.

        Returns:
        --------
        float

        """
        
        x = self.x0 + dx.sum(axis=0)
        return self.portfolio_return(dx,gross=False) - self.portfolio_var(x)*self.g/2

    def portfolio_var(self,x):
        """Portfolio variance given allocations `x`."""
        return x@(self.V)@x

    def portfolio_return(self,dx,gross=False):
        """
        Portfolio return.

        Parameters:
        -----------
        dx : numpy.array
            (2,n) array of allocation changes of n assets. First row refers 
            to allocation increases, second row to decreases. Accordingly, 
            the first row should contain positive values only, the second row
            only negative values.
        gross : bool (default : False)
            If False, returns portfolio return minus transaction costs. Returns
            portfolio return not accounted for transaction costs otherwise

        Returns:
        --------
        float

        """

        x = self.x0 + dx.sum(axis=0)
        if gross:
            return x@self.mu
        else:
            return x@self.mu - self.portfolio_implcost(dx)

    def portfolio_implcost(self,dx):
        """
        Transaction costs for implementing portfolio allocation
        changes.

        Parameters:
        -----------
        dx : numpy.array
            (2,n) array of allocation changes of n assets. First row refers 
            to increases in allocation, second row to decreases. Accordingly, 
            the first row should contain positive values only, the second row
            only negative values.

        Returns:
        --------
        float

        """
        return dx@self.k@[1,-1]