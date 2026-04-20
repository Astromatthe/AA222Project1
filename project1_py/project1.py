#
# File: project1.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project1_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np


def optimize(f, g, x0, n, count, prob):
    """
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `g` costs twice of `f`
        count (function): takes no arguments are returns current count
        prob (str): Name of the problem. So you can use a different strategy
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
    Returns:
        x_best (np.array): best selection of variables found
    """
    strategy_map = {
        'simple1': "l_bfgs", # Rosenbrock
        'simple2': "l_bfgs", # Himmelblau
        'simple3': "l_bfgs", # Powell
        'secret1': "l_bfgs",
        'secret2': "l_bfgs"
    }
    strategy = strategy_map[prob]
    x_history = [x0.copy()]

    # ----- local descent -----
    if strategy == "local_descent":
        # print(f"Running local descent for {prob} with n={n}")
        x_history = local_descent(f, g, x0, n, count)
        x_best = x_history[-1]

    if strategy == "l_bfgs":
        x_history = l_bfgs(f, g, x0, n, count)
        x_best = x_history[-1]

    return x_best
    
def local_descent(f, g, x0, n, count):
    """
    Local descent with backtracking line search and adaptive step size.
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `g` costs twice of `f`
        count (function): takes no arguments are returns current count
    Returns:
        x_history (np.array): best selection of variables found"""
    # initial step size
    step_size = 1
    x = x0.copy()
    x_best = x0.copy()
    x_history = [x0.copy()]

    # 1. if termination condition is met, return x_best
    if count() + 1 > n:
        return x_best
    
    f_best = f(x_best)
    f_current = f_best
    alpha = step_size

    while True:
        if count() + 2 > n:
            break
        # 2. Determine descent direction
        grad = g(x)
        grad_norm = np.linalg.norm(grad)
        # safe guard for zero gradient
        if grad_norm < 1e-12:
            break
        
        # normalized descent direction
        descent_direction = -grad / (grad_norm + 1e-12)

        improved = False
        step = alpha

        # try a few backtracking steps
        for _ in range(4):
            if count() + 1 > n:
                break

            x_try = x + step * descent_direction
            f_try = f(x_try)

            if f_try < f_current:
                # print(f"Improved from {f_current:.4e} to {f_try:.4e} with step size {step:.2e} at count {count()}")
                x = x_try
                x_history.append(x.copy())
                f_current = f_try
                improved = True
                if f_try < f_best:
                    x_best = x_try
                    f_best = f_try
                break
            else:
                step *= 0.5
    
        # adapt step size
        if improved:
            alpha = min(alpha * 20, 1.0)
        else:
            alpha *= 0.5
            if alpha < 1e-10:
                break

    return x_history

def l_bfgs(f, g, x0, n, count):
    """
    L-BFGS optimization method.
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `g` costs twice of `f`
        count (function): takes no arguments are returns current count
    Returns:
        x_history (list of np.array): history of positions visited during optimization
    """
    x = x0.copy()
    x_history = [x.copy()]

    # L-BFGS parameters
    m_max = 10  # memory size
    deltas = []  # list to store delta_k = x_{k+1} - x_k
    gammas = []  # list to store gamma_k = grad_{k+1} - grad_k

    while True:
        # check termination condition
        if count() + 2 > n:
            break
        
        grad = g(x)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break

        # compute L-BFGS direction using two-loop recursion
        m = len(deltas)
        if m > 0:
            q = grad.copy()
            alphas = []
            # backward loop
            for delta, gamma in zip(reversed(deltas), reversed(gammas)):
                alpha = (delta @ q) / (gamma @ delta + 1e-8)
                alphas.append(alpha)
                q -= alpha * gamma
            
            z = (gammas[-1] @ deltas[-1]) / (gammas[-1] @ gammas[-1] + 1e-8) * q

            # forward loop
            for delta, gamma, alpha in zip(deltas, gammas, reversed(alphas)):
                z = z + delta * (alpha - (gamma @ z) / (gamma @ delta + 1e-8))
            d = -z
        else:
            d = -grad
        
        # line search to find step size
        alpha, f, pen_f = line_search(f, x, d, grad, count, n)
        x_new = x + alpha * d

        # update history
        x_history.append(x_new.copy())
    
        if count() + 2 > n:
            break
        # update memory
        delta_new = x_new - x
        grad_new = g(x_new)
        gamma_new = grad_new - grad
        if len(deltas) == m_max:
            deltas.pop(0)
            gammas.pop(0)
        deltas.append(delta_new)
        gammas.append(gamma_new)
        x = x_new
    return x_history


def line_search(f, x, d, grad, count, n):
    """
    Backtracking line search to find a suitable step size.
    Args:
        f (function): Function to be optimized
        x (np.array): Current point
        d (np.array): Descent direction
        grad (np.array): Gradient at current point
        count (function): takes no arguments are returns current count
        n (int): Number of evaluations allowed. Remember [g] costs twice of [f]
    Returns:
        step_size (float): step size found by line search
    """
    n_searches = 4
    alpha = 1.0
    beta = 0.5
    sigma = 1e-4

    if count() + 1 > n:
        return 0.0
    f_x = f(x)
    for _ in range(n_searches):
        if count() + 1 > n:
            break

        x_try = x + alpha * d
        f_try = f(x_try)

        if f_try <= f_x + sigma * alpha * grad @ d:
            break
        else:
            alpha *= beta

    return alpha