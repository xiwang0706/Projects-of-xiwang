"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import sys, os
cur_dir = os.path.dirname(__file__)
print(cur_dir)

module_path = os.path.abspath(os.path.join(cur_dir, "../../.."))
sys.path.append(module_path)

import deepxde as dde
import numpy as np
# Backend tensorflow.compat.v1 or tensorflow
# from deepxde.backend import tf
# Backend pytorch
import torch
# Backend jax
# import jax.numpy as jnp
# Backend paddle
# import paddle
import torch
import numpy as np
import random

dde.config.real.set_float64()

# 设置随机种子
# seed = 42
# seed = 50
seed = 60
torch.manual_seed(seed)  # PyTorch CPU
torch.cuda.manual_seed(seed)  # PyTorch GPU
torch.cuda.manual_seed_all(seed)  # 如果有多个GPU
np.random.seed(seed)  # NumPy
random.seed(seed)  # Python 内置随机库

# 如果使用 cuDNN，可以设置可复现性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

cv = 1.

def pde(x, y):
    # Most backends
    dy_t = dde.grad.jacobian(y, x, j=1) # 时间变量应该是 x[1]
    dy_xx = dde.grad.hessian(y, x, j=0)
    # Backend jax
    # dy_t, _ = dde.grad.jacobian(y, x, j=1)
    # dy_xx, _ = dde.grad.hessian(y, x, j=0)
    # Backend tensorflow.compat.v1 or tensorflow
    # return (
    #     dy_t
    #     - dy_xx
    #     + tf.exp(-x[:, 1:])
    #     * (tf.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0:1]))
    # )

    # Heterogeneous, with pytorch tensor
    ## x (n_pts, n_dim)
    ## y (n_pts, n_unknown)
    Ks = torch.tensor([0.24, 0.06], dtype=torch.float64)
    x_interface = 0.6
    cv = torch.where(x[:,0:1]<x_interface, Ks[0], Ks[1])

    # Backend pytorch
    return (
        dy_t
        - cv*dy_xx
    )

    # Backend jax
    # return (
    #     dy_t
    #     - dy_xx
    #     + jnp.exp(-x[:, 1:])
    #     * (jnp.sin(np.pi * x[..., 0:1]) - np.pi ** 2 * jnp.sin(np.pi * x[..., 0:1]))
    # )
    # Backend paddle
    # return (
    #     dy_t
    #     - dy_xx
    #     + paddle.exp(-x[:, 1:])
    #     * (paddle.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * paddle.sin(np.pi * x[:, 0:1]))
    # )

def ic_condition(x): 
    """x (n_pt, n_var)"""
    return np.ones_like(x[:,0:1],dtype=np.float64)

from postpre import*
def solution(x):
    return analytical(t=x[:,1:2], z=1-x[:,0:1], u0=1, cv=cv, h=1., n=1000)

def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1.)
def boundary_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.)

# value for dirichlet values
def dirichlet_0(x):
    return 0
# value for Neumann 
def Neumann_0(x):
    return 0

geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 3)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc_top = dde.icbc.DirichletBC(geomtime, dirichlet_0, boundary_top)
bc_down = dde.icbc.NeumannBC(geomtime, Neumann_0, boundary_bottom)
ic = dde.icbc.IC(geomtime, ic_condition, lambda _, on_initial: on_initial)
data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic, bc_down,bc_top],
    num_domain=10000,
    num_boundary=600,
    num_initial=200,
    solution=solution,
    num_test=10000,
)

layer_size = [2] + [60] * 5 + [1]
activation = "tanh"
initializer = "Glorot uniform"
# net = dde.nn.fnn.FNN(layer_size, activation, initializer)
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

## Adam
model.compile("adam", lr=0.001, metrics=["l2 relative error"], decay= ("step", 2000, 0.2)) # decay = ("exponential", gamma)
# iterations = 10
iterations = 0
losshistory, train_state = model.train(iterations=iterations)

## LBFGS
dde.optimizers.config.set_LBFGS_options(
  maxcor=50,
  ftol= 1e-10,
  gtol=1e-4,
  maxiter=4500,
  # maxiter=100,
  maxls=50,
)
dde.optimizers.set_LBFGS_options(lr=0.1)
model.compile("L-BFGS")
losshistory, train_state = model.train()

# dde.saveplot(losshistory, train_state, issave=True, isplot=True)
# ============================================================================ #
# Postprocess
cur_dir = os.path.dirname(__file__)
print(cur_dir)

from postpre import *
xt, nx, nt  = get_xt(xs=(0,1), dx=2e-2, ts=(0,3), dt=1e-3)
u = model.predict(x=xt).reshape(nx, nt)
np.save("pinn_u.npy", u)
np.save("pinn_loss.npy", losshistory.loss_train)
