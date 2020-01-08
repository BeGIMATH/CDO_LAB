import numpy as np
import random


def GD(x0, grad, prox, max_iter, n, L, mu):
    '''
    Parameters
    ----------
    x0: array, shape (nb_features,)
        Initialisation of the solver
    grad: function
        Gradient of the objective function
    max_iter: int
        Number of iterations (i.e. number of descent steps). Note that for GD or SVRG,
        one iteration is one epoch i.e, one pass through the data, while for SGD, SAG and SAGA,
        one iteration uses only one data point.
    n: int
        Dataset size
    L: float
        Smoothness constant of the objective function
    mu: float
        Strong convexity constant of the objective function

    Returns
    -------
    x: array, shape (nb_features,)
        final iterate of the solver
    x_tab: array, shape (nb_features, max_iter)
        table of all the iterates
    '''
    
    stepsize = 1.0 / L
    x = x0
    x_tab = np.copy(x)

    for k in range(max_iter):
        x = x - stepsize*grad(x)
        x_tab = np.vstack((x_tab, x))
    return x, x_tab


def GD_prox(x0, grad, prox, max_iter, n, L, mu):
    stepsize = 1.0 / L
    x = x0
    x_tab = np.copy(x)

    for k in range(max_iter):
        x = prox(x - stepsize*grad(x), stepsize)
        x_tab = np.vstack((x_tab, x))
    return x, x_tab


def SGD(x0, grad, prox, max_iter, n, L, mu):

    x = np.copy(x0)
    x_tab = np.copy(x)
    
    for k in range(max_iter):
        t = random.randrange(1, n)
        
        step = 1/(pow(k+1, 0.6))
        x = x - step*grad(x, t)
        if k % n == 0:  # each completed epoch
            x_tab = np.vstack((x_tab, x))

    return x, x_tab


def SGD_prox(x0, grad, prox, max_iter, n, L, mu):
    x = np.copy(x0)
    x_tab = np.copy(x)
    for k in range(max_iter):
        t = random.randrange(1, n)

        step = 1/(pow(k+1, 0.7))
        #step = mu / (k+1)
        x = prox(x - step*grad(x, t), step)
        if k % n == 0:  # each completed epoch
            x_tab = np.vstack((x_tab, x))

    return x, x_tab


def SAGA(x0, grad, prox, max_iter, n, L, mu):

    x = np.copy(x0)
    x_tab = np.copy(x)
    d = len(x)

    A = np.zeros([n, d])
    for i in range(n):
        A[i, :] = grad(x, i)

    step = 1 / (2*(mu*n + L))

    for k in range(max_iter):
        xprev = np.copy(x)

        j = random.randrange(1, n)
        f_grad_phi_k = A[j, :]
        alfa_bar = A.mean(0)
        A[j, :] = grad(x, j)
        x = x - step*(A[j, :] - f_grad_phi_k + alfa_bar)

        if (k % n == 0):  # each completed epoch-n
            x_tab = np.vstack((x_tab, x))
    
    return x, x_tab


def SAGA_prox(x0, grad, prox, max_iter, n, L, mu):

    x = np.copy(x0)
    x_tab = np.copy(x)
    d = len(x)

    A = np.zeros([n, d])
    for i in range(n):
        A[i, :] = grad(x, i)
    #step = 1/(3*(mu*n+L))
    step = 1 / (2*(mu*n + L))
    for k in range(max_iter):
        #step = 1/(pow(k+1, 0.7))

        j = random.randrange(1, n)
        f_grad_phi_k = A[j, :]
        alfa_bar = A.mean(0)
        A[j, :] = grad(x, j)
        x = prox(x*(1-step*mu) - step *
                 (A[j, :] - f_grad_phi_k + alfa_bar), step)

        if (k % n == 0):  # each completed epoch
            x_tab = np.vstack((x_tab, x))

    return x, x_tab


def SVRG(x0, grad, prox, max_iter, n, L, mu):
    x = np.copy(x0)
    x_tab = np.copy(x)
    d = len(x)

    step = 1/(2*(mu*n+L))
    for k in range(max_iter):
        #step = 1/(pow(k+1, 0.6))
        G = grad(x)
        y = x

        for i in range(2):
            e = random.randrange(1, n)
            g = grad(y, e)-grad(x, e)+G
            y = y-step*g
        x = y
        x_tab = np.vstack((x_tab, x))

    return x, x_tab


def SVRG_prox(x0, grad, prox, max_iter, n, L, mu):
    x = np.copy(x0)
    x_tab = np.copy(x)
    d = len(x)
    step = 1/(3*(mu*n+L))

    #step = 0.1 / L
    for k in range(max_iter):
        G = grad(x)
        y = x

        for i in range(2):
            e = random.randrange(1, n)
            g = grad(y, e)-grad(x, e)+G
            y = y-step*g
        x = prox(y, step)
        x_tab = np.vstack((x_tab, x))

    return x, x_tab
