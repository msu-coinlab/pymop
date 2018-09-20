

def evaluate():
    import numpy as np

    # initialize it with the necessary parameters
    from pymop.problems.dtlz import DTLZ1
    problem = DTLZ1(n_var=10, n_obj=3)

    # evaluation function returns by default two numpy arrays - objective function values and constraints -
    # as input either provide a vector
    F, G = problem.evaluate(np.random.random(10))

    # or a whole matrix to evaluate several solutions at once
    F, G = problem.evaluate(np.random.random((100, 10)))

    # if no constraints should be returned
    F = problem.evaluate(np.random.random((100, 10)), return_constraint_violation=False)

    from pymop.problems.welded_beam import WeldedBeam
    F, CV = WeldedBeam().evaluate(np.random.random((100, 4)), return_constraint_violation=True)


def plot():
    from pymop import plot_problem_surface, Ackley
    plot_problem_surface(Ackley(n_var=1), 200)
    plot_problem_surface(Ackley(n_var=2), 200, plot_type="wireframe")
    plot_problem_surface(Ackley(n_var=2), 200, plot_type="contour")
