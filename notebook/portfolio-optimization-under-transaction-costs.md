```python
import numpy as np 
import sys
sys.path.append('..')
from PortfolioOptimization import OptP
```

# Portfolio optimization under transaction costs

The optimization problem is given by 

\begin{align}
    \max_{x^+,x^-}\; & x' \mu - k' (x^+ + x^-) - \gamma x' \mathbb V x/2 \\
    \text{s.t.} & \nonumber \\ 
     x_i^+,x_i^- & \geq 0\; \forall i \\
     x^+ - x^- + \bar{x} & = x \\
    \mathbf 1'x & = 1 \\
    x_i & \geq 0\; \forall i \\ 
    \text{optional:} & \nonumber \\
     x^{+\prime} x^- & = 0
\end{align}

where $x$ is a vector of the assets' weights in the portfolio, $\mu$ is a vector of expected returns, $k$ is a vector of proportional transaction costs, $x^+$ and $x^-$ are vectors of increases and decreases in asset allocations, respectively, $\gamma$ is the investor's risk aversion, $\mathbb V$ is the assets' covariance matrix and $\bar{x}$ is the initial asset allocation.

For the two asset case we can get an analytical solution.

The Karush-Kuhn-Tucker conditions (including the optional constraint, although the optional constraint does not matter for the two asset case) are
\begin{align}
-\mu + k + \gamma V'x + \lambda \mathbb{1} - \nu + \omega x^{-} - \theta_{+} & = 0 \\
\mu + k - \gamma V'x - \lambda \mathbb{1} + \nu + \omega x^{+} - \theta_{-} & = 0   \\
x & = x^{+} - x^{-} + \bar{x} \\
\mathbb{1}'x & = 1 \\
\nu_i x_i & = 0\;\forall i \\
x & \geq 0 \\
\nu & \geq 0 \\
x^{+\prime} x^{-} & = 0 \\
x_i^{+}\theta_{+i} & = 0\;\forall i \\
x_i^{-}\theta_{-i} & = 0\;\forall i \\
x^{+},x^{-},\theta_{+},\theta_{-} & \geq 0
\end{align}

Solving for the final allocation $x_i$, $i=1,2$, we obtain
\begin{align}
\hat{x}_1 & := \min\{1, \max\{0, \frac{(\mu_1 - \mu_2 - k_1 - k_2)/\gamma + V_{(22)} - V_d}{V_{(11)} + V_{(22)} - 2V_d} \} \} \\ 
\tilde{x}_2 & := \min\{1, \max\{0, \frac{(\mu_2 - \mu_1 - k_1 - k_2)/\gamma + V_{(11)} - V_d}{V_{(11)} + V_{(22)} - 2V_d} \} \} \\
x^* & = \begin{cases}
\begin{pmatrix} \hat{x}_1 \\
1-\hat{x}_1 \end{pmatrix} & \text{if $\hat{x}_1>\bar{x}_1$} \\
\begin{pmatrix} 1-\tilde{x}_2 \\
\tilde{x}_2 \end{pmatrix} & \text{if $\tilde{x}_2>\bar{x}_2$} \\
\bar{x}  & \text{otherwise} 
\end{cases}  
\end{align}
Note that the transaction costs induce a no-trade region, where keeping the starting portfolio is the optimal solution.

Let see if the optimization tool outputs the correct result.


```python
def opt_x_analytical(mu,g,k,V,x0):
    """Optimal asset allocation under transaction costs; 2 asset case."""
    x1 = ((mu[0] - mu[1] - k[0] - k[1])/g + V[1,1] - V[0,1])/(V[0,0]+V[1,1]-2*V[0,1])
    x1 = min([1, max([0,x1])])
    x2 = ((mu[1] - mu[0] - k[1] - k[0])/g + V[0,0] - V[0,1])/(V[0,0]+V[1,1]-2*V[0,1])
    x2 = min([1, max([0,x2])])    
    if x1>x0[0]:
        x2 = 1-x1
        return x1,x2
    elif x2>x0[1]:
        x1 = 1-x2
        return x1,x2
    else:
        return x0
```


```python
# simulate some data
n = 2
mu = np.sort(1 + np.random.rand(n))
Q = np.random.uniform(-1,1,size=(n,n)) 
V = Q.T@Q 
V /= V[0,0]
x0 = np.array([1/n]*n)
k = np.random.rand(n)/10
g = 5 # risk aversion parameter
```


```python
x1 = opt_x_analytical(mu,g,k,V,x0)
print("Expected Return:\n%.2f, %.2f" % (mu[0],mu[1]))
print("Covariance Matrix:\n%s" % '\n'.join([', '.join(['%.2f' % v for v in row]) for row in V]) )
print("Transaction costs:\n%s" % ', '.join(['%.4f' % c for c in k]))
print("Optimal allocation, analytical (in %%):\n%s" % ', '.join(['%.2f' % (x*100) for x in x1]))
```

    Expected Return:
    1.73, 1.78
    Covariance Matrix:
    1.00, -0.07
    -0.07, 0.56
    Transaction costs:
    0.0387, 0.0381
    Optimal allocation, analytical (in %):
    37.53, 62.47



```python
# numerical solution
op = OptP(mu,k,V,g,x0)
```


```python
op.optimize_p(solver='qp')

x_qp = op.solution['x']
print("Optimal allocation using quadratic programming (ie. w/o optional constraint):\n%s" % ', '.join(['%.2f' % (x*100) for x in x_qp]))
```

    Optimal allocation using quadratic programming (ie. w/o optional constraint):
    37.53, 62.47



```python
op.optimize_p(solver='scipy',init='at_zero',nonlin_constr=False)

x_scipy = op.solution['x']
print("Optimal allocation using scipy (w/o optional constraint):\n%s" % ', '.join(['%.2f' % (x*100) for x in x_scipy]))
```

    Optimal allocation using scipy (w/o optional constraint):
    37.52, 62.48



```python
op.optimize_p(solver='scipy',init='random',n_repeat=10,alpha=5,nonlin_constr=True,supply_H_and_jac=False)

x_scipy_nl = op.solution['x']
print("Optimal allocation using scipy (with the optional constraint):\n%s" % ', '.join(['%.2f' % (x*100) for x in x_scipy_nl]))
```

    /home/simon/anaconda3/lib/python3.7/site-packages/scipy/optimize/_trustregion_constr/projections.py:182: UserWarning: Singular Jacobian matrix. Using SVD decomposition to perform the factorizations.
      warn('Singular Jacobian matrix. Using SVD decomposition to ' +
    /home/simon/anaconda3/lib/python3.7/site-packages/scipy/optimize/_hessian_update_strategy.py:187: UserWarning: delta_grad == 0.0. Check if the approximated function is linear. If the function is linear better results can be obtained by defining the Hessian as zero instead of using quasi-Newton approximations.
      'approximations.', UserWarning)


    Optimal allocation using scipy (with the optional constraint):
    37.53, 62.47



```python
op.optimize_p(solver='scipy',init='random',n_repeat=10,alpha=5,nonlin_constr=True,supply_H_and_jac=True)

x_scipy_nl = op.solution['x']
print("Optimal allocation using scipy (with the optional constraint):\n%s" % ', '.join(['%.2f' % (x*100) for x in x_scipy_nl]))
```

    Optimal allocation using scipy (with the optional constraint):
    37.53, 62.47


## Utility and Variance
We can also get the utility for a given change in portfolio weights and the portfolio variance.


```python
print("Investor utility:\n",op.utility(op.solution['dx']),'\n')
print("Portfolio Variance:\n",op.portfolio_var(op.solution['x']),'\n')
print("Portfolio Expected Return:\n",op.portfolio_return(op.solution['dx'], gross=True),'\n')
print("Portfolio implementation cost:\n",op.portfolio_implcost(op.solution['dx']),'\n')
print("Portfolio Expected Return net of transaction costs:\n",op.portfolio_return(op.solution['dx'], gross=True),'\n')
```

    Investor utility:
     0.9357442889319956 
    
    Portfolio Variance:
     0.32662792809345154 
    
    Portfolio Expected Return:
     1.7618935232665447 
    
    Portfolio implementation cost:
     0.009579414100920142 
    
    Portfolio Expected Return net of transaction costs:
     1.7618935232665447 
    



```python

```
