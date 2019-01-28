import unittest

from pymop import *


def load(name):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")

    X = anp.loadtxt(os.path.join(path, "%s.x" % name))
    F = anp.loadtxt(os.path.join(path, "%s.f" % name))

    CV = None
    if os.path.exists(os.path.join(path, "%s.cv" % name)):
        CV = anp.loadtxt(os.path.join(path, "%s.cv" % name))

    return X, F, CV


problems = [
    ('C1DTLZ1', [14, 10]), ('C1DTLZ3', [12, 3]),
    ('DTLZ1', [10, 3]), ('DTLZ2', [10, 3]), ('DTLZ3', [10, 3]), ('DTLZ4', [10, 3]), ('DTLZ5', [10, 3]),
    ('DTLZ6', [10, 3]), ('DTLZ7', [10, 3]),
    ('ZDT1', [10]), ('ZDT2', [10]), ('ZDT3', [10]), ('ZDT4', [10]), ('ZDT6', [10]),
    ('TNK', []), ('Rosenbrock', [10]), ('Rastrigin', [10]), ('Griewank', [10]), ('OSY', []), ('Kursawe', []),
    ('WeldedBeam', []), ('Carside', []), ('BNH', []),
    ('G1', []), ('G2', []), ('G3', []), ('G4', []), ('G5', []), ('G6', []), ('G7', []), ('G8', []),
    ('G9', []), ('G10', []),

]


class CorrectnessTest(unittest.TestCase):

    def test_problems(self):
        for entry in problems:
            name, params = entry
            print("Testing: " + name)

            X, F, CV = load(name)
            problem = globals()[name](*params)
            _F, _G, _CV, _dF, _dG = problem.evaluate(X, return_values_of=["F", "G", "CV", "dF", "dG"])

            if problem.n_obj == 1:
                F = F[:, None]

            self.assertTrue(anp.all(anp.abs(_F - F) < 0.00001))

            if problem.n_constr > 0:
                self.assertTrue(anp.all(anp.abs(_CV[:, 0] - CV) < 0.0001))


if __name__ == '__main__':
    unittest.main()
