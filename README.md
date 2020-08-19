# Portfolio-Optimization

Module to optimize portfolio allocation under proportional transaction costs for a mean-variance utility function.

See notebook for details.

# Dependencies
- scipy
- numpy
- cvxopt
- warnings 

# Usage
```python
import numpy as np
from PortfolioOptimization import OptP

# simulate some data
n = 10
mu = np.sort(1 + np.random.rand(n))
Q = np.random.uniform(-1,1,size=(n,n)) 
V = Q.T@Q 
V /= V[0,0]
x0 = np.array([1/n]*n)
k = np.random.rand(n)/10
g = 5 # risk aversion parameter

# optimize
op = OptP(mu,k,V,g,x0)
op.optimize_p(solver='qp')

print(op.solution)
```
