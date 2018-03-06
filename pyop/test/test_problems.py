import os
import unittest


from pyop.problems.dtlz import *
from pyop.problems.zdt import *
from pyop.problems.tnk import TNK
from pyop.problems.rosenbrock import Rosenbrock
from pyop.problems.rastrigin import Rastrigin
from pyop.problems.griewank import Griewank
from pyop.problems.osy import OSY
from pyop.problems.kursawe import Kursawe
from pyop.problems.welded_beam import WeldedBeam
from pyop.problems.carside import Carside
from pyop.problems.bnh import BNH



def load(name):
    X = np.loadtxt('../resources/' + name + '.x')
    F = np.loadtxt('../resources/' + name + '.f')

    G = None
    if os.path.exists('../resources/' + name + '.g'):
        G = np.loadtxt('../resources/' + name + '.g')

    return X, F, G


problems = [
    ('DTLZ1', [10, 3]), ('DTLZ2', [10, 3]), ('DTLZ3', [10, 3]), ('DTLZ4', [10, 3]), ('DTLZ5', [10, 3]),
    ('DTLZ6', [10, 3]) ,('DTLZ7', [10, 3]),
    ('ZDT1', [10]), ('ZDT2', [10]), ('ZDT3', [10]), ('ZDT4', [10]), ('ZDT6', [10]),
    ('TNK', []), ('Rosenbrock', [10]), ('Rastrigin', [10]), ('Griewank', [10]), ('OSY', []), ('Kursawe', []),
    ('WeldedBeam', []), ('Carside', []), ('BNH', [])
]


class ProblemTest(unittest.TestCase):

    def test_problems(self):
        for entry in problems:
            name, params = entry
            print("Testing: " + name)

            X, F, CV = load(name)
            problem = globals()[name](*params)
            F_, G_ = problem.evaluate(X)

            if problem.n_obj == 1:
                F = F[:, None]

            if name == "WeldedBeam":
                pass


            self.assertTrue(np.all(np.abs(F_ - F) < 0.00001))

            if problem.n_constr > 0:
                G_[G_<0] = 0
                CV_ = np.sum(G_, axis=1)
                self.assertTrue(np.all(np.abs(CV_ - CV) < 0.00001))


def opt_problems(X):
    from optproblems import Individual
    from optproblems.dtlz import DTLZ2 as optDTLZ2

    T = np.zeros((100, 3))
    for i in range(100):
        ind = Individual(X[i, :])
        optDTLZ2(3, 10).evaluate(ind)
        T[i, :] = ind.objective_values
    return T


if __name__ == '__main__':
    unittest.main()
