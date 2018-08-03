import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))

import numpy as np

if __name__ == "__main__":

    if len(sys.argv) != 4:
        raise Exception("Usage: cmd.py <problem> <params> <func>")

    _, par_problem, par_params, par_func = sys.argv

    # imports all problem in this framework
    from pymop import *

    try:
        clazz = eval(par_problem)
    except Exception as e:
        print("ERROR: Problem %s not found." % par_problem[0])
        print(e)
        exit(1)

    # no parameter given - just "[]"
    if len(par_params) <= 2:
        par_params = []
    else:
        par_params = par_params.replace("[", "").replace("]", "").split(",")

    if len(par_params) % 2 != 0:
        raise Exception("Parameters must have an even number of arguments. [key, value, key, value]. Currently: %s" % len(par_params))

    try:

        params_of_constructor = clazz.__init__.__code__.co_varnames[1:]
        default_values_of_params = clazz.__init__.__defaults__

        # if no params are defined just try without any
        if len(params_of_constructor) == 0:
            problem = clazz()

        # else parse the params as dict
        else:

            if len(params_of_constructor) != len(default_values_of_params):
                raise Exception("Each problem used externally must have provided default values for all constructor arguments.")

            params = {}
            for i in range(int(len(par_params)/2)):
                key = par_params[2*i].strip()
                val = par_params[2*i+1].strip()

                if key in params_of_constructor:
                    default_type = type(default_values_of_params[params_of_constructor.index(key)])
                    params[key] = default_type(val)

            problem = clazz(**params)

    except Exception as e:
        print("ERROR: Problem can not be initialized with params %s." % params)
        print(e)
        exit(1)

    # execute the method - use stdin and stdout if necessary
    if par_func == "evaluate":

        X = np.loadtxt(sys.stdin.readlines(), dtype=np.float)

        if len(X.shape) == 1:
            X = X[None, :]

        F, G = problem.evaluate(X)
        if problem.n_constr > 0:
            F = np.concatenate([F, G], axis=1)

        np.savetxt(sys.stdout, F, fmt='%.18e', delimiter=' ')

    elif par_func == 'front':
        np.savetxt(par_out, problem._calc_pareto_front(), fmt='%.18e', delimiter=' ')

    elif par_func == 'info':
        import json

        d = {
            'n_var': problem.n_var,
            'n_obj': problem.n_obj,
            'n_constr': problem.n_constr,
            'xl': problem.xl.tolist(),
            'xu': problem.xu.tolist()
        }
        print(json.dumps(d))

    else:
        print("ERROR: Method does not exist.")
        exit(1)

    sys.stdout.flush()