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
        'simple1': "local_descent", # Rosenbrock
        'simple2': "local_descent", # Himmelblau
        'simple3': "local_descent", # Powell
        'secret1': "local_descent",
        'secret2': "local_descent"
    }
    strategy = strategy_map[prob]

    # ----- local descent -----
    if strategy == "local_descent":
        # initial step size
        step_size = 1e-2
        x = x0.copy()
        x_best = x0.copy()

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
                    x = x_try
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

    return x_best