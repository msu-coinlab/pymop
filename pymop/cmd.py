import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))
from inspect import signature

import numpy as np

if __name__ == "__main__":

    if len(sys.argv) != 5:
        raise Exception("Usage: cmd.py <problem> <func> <in> <out>")

    _, par_problem, par_func, par_in, par_out = sys.argv

    # imports all problem in this framework
    from pymop import *

    # read the problem and initialize the class object
    str_problem = par_problem.split("(")

    try:
        clazz = eval(str_problem[0])
    except Exception as e:
        print("ERROR: Problem %s not found." % str_problem[0])
        print(e)
        exit(1)

    has_params = len(str_problem) > 1

    try:
        if has_params:
            params = [int(s) for s in str_problem[1][:-1].split(",")]
            print(clazz)
            print(params)
            print(len(params))
            problem = clazz(*params[:len(signature(clazz).parameters)])
        else:
            problem = clazz()
    except Exception as e:
        print("ERROR: Problem can not be initialized with params %s." % params)
        print(e)
        exit(1)

    # execute the method - use stdin and stdout if necessary
    if par_func == "evaluate":

        X = np.loadtxt(par_in, dtype=np.float)

        if len(X.shape) == 1:
            X = X[None, :]

        F, G = problem.evaluate(X)
        if problem.n_constr > 0:
            F = np.concatenate([F, G], axis=1)

        np.savetxt(par_out, F, fmt='%.18e', delimiter=' ')

    elif par_func == 'front':
        np.savetxt(par_out, problem.calc_pareto_front(), fmt='%.18e', delimiter=' ')

    elif par_func == 'info':
        import json

        d = {
            'n_var': problem.n_var,
            'n_constr': problem.n_constr,
            'xl': problem.xl.tolist(),
            'xu': problem.xu.tolist()
        }
        print(json.dumps(d))

    else:
        print("ERROR: Method does not exist.")
        exit(1)

    sys.stdout.flush()