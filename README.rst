pymop - Optimization Test Problems
==================================


Installation
==================================

The test problems are uploaded to the PyPi Repository.

.. code:: bash

    pip install pymop

For the current development version:

.. code:: bash

    git clone https://github.com/msu-coinlab/pymop
    cd pymop
    python setup.py install

Problems
==================================

In this package single- as well as multi-objective test problems are
included:


-  Single-Objective:

   -  Ackley
   -  BNH
   -  Griewank
   -  Knapsack
   -  Schwefel
   -  Sphere
   -  Zakharov

-  Multi-Objective:

   -  ZDT 1-6 
   -  DTLZ 1-7 
   -  WFG 1-9 
   -  Carside Impact
   -  BNH
   -  Kursawe
   -  OSY`
   -  TNK
   -  Welded Beam

Usage
==================================
.. code:: python

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


Implementation
==================================

All problems are implemented to efficiently evaluate multiple input
points at a time. Therefore, the input can be a n x m dimensional
matrix, where n is the number of points to evaluate and m the number of
variables.


Contributors
==================================
| Julian Blank
| Yash Prasad


Contact
==================================
Feel free to contact me if you have any question:

| Julian Blank (blankjul [at] egr.msu.edu)
| Michigan State University
| Computational Optimization and Innovation Laboratory (COIN)
| East Lansing, MI 48824, USA



Changelog
==================================
`0.2.1`
---------------------------------------

* First official release providing a bunch of test problems
* Some redesign of classes compared to early versions
* Added trust_2d problems

