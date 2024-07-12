"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import sys, os
cur_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(cur_dir, "../.."))
sys.path.append(module_path)

import deepxde as dde
dde.config.set_default_float("float64")
import numpy as np

# ============================================================================ #
# Geometry parameters
x1 = 0.5
x2 = 0.75
w = 1.0
h = 0.5
geom = dde.geometry.Rectangle([0, 0], [w, h])

# ============================================================================ #
# PDE residuals
import torch # run in pytorch env
def pde(x, y):
    # Most backends
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)

    # Heterogeneous, with pytorch tensor
    ## x (n_pts, n_dim)
    ## y (n_pts, n_unknown)
    Ks = torch.tensor([1e-3, 1e-4], dtype=torch.float64)
    x1 = 0.5
    x2 = 0.75
    middle = torch.logical_and(x[:,0:1]>x1, x[:,0:1]<x2)
    K = torch.where(middle, Ks[0], Ks[1])
    
    # Backend jax
    # dy_xx, _ = dde.grad.hessian(y, x, i=0, j=0)
    # dy_yy, _ = dde.grad.hessian(y, x, i=1, j=1)
    # return -dy_xx - dy_yy
    return  - K * (dy_xx + dy_yy)

# ============================================================================ #
# Boundaries
def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.0)
def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], w)
def boundary_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], h)
def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.0)

# value for dirichlet values
def dirichlet_1(x):
    return 1.
def dirichlet_0_5(x):
    return 0.5
# value for Neumann 
def Neumann_0(x):
    return 0

Dirichlet_left = dde.icbc.DirichletBC(geom, dirichlet_1, boundary_left)
Dirichlet_right = dde.icbc.DirichletBC(geom, dirichlet_0_5, boundary_right)
Neumann_top = dde.icbc.NeumannBC(geom, Neumann_0, boundary_top)
Neumann_bottom = dde.icbc.NeumannBC(geom, Neumann_0, boundary_bottom)

data = dde.data.PDE(geom, pde, 
                    [Dirichlet_left, Dirichlet_right, Neumann_top, Neumann_bottom],
                    num_domain=2**14, num_boundary=2**10, num_test=1000) 

# ============================================================================ #
# Network
net = dde.nn.FNN([2] + [56]*3 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

# ============================================================================ #
# Train
## Adam
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=15000, display_every=1)
## LBFGS
dde.optimizers.config.set_LBFGS_options(
  maxcor=50,
  ftol= 1e-10,
  gtol=1e-12,
  maxiter=15000,
  maxls=50,
)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# ============================================================================ #
# Postprocess
x = np.load(cur_dir+"/fem_points.npy")
y = model.predict(x=x)
np.save(cur_dir + "/pinn_H.npy", y)
np.save(cur_dir + "/pinn_loss.npy", losshistory.loss_train)
