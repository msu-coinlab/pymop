# pymop - Optimization Test Problems



Installation
------

The test problems are uploaded to the PyPi Repository.

```bash
pip install pymop
```


Usage
------

```python
# numpy arrays are required as an input
import numpy as np

# first import the specific problem to be solved
from pymop.dtlz import DTLZ1

# initialize it with the necessary parameters
problem = DTLZ1(n_var=10, n_obj=3)

# evaluation function returns by default two numpy arrays - objective function values and constraints -
# as input either provide a vector
F, G = problem.evaluate(np.random.random(10))

# or a whole matrix to evaluate several solutions at once
F, G = problem.evaluate(np.random.random((100, 10)))

# if no constraints should be returned
F = problem.evaluate(np.random.random((100, 10)), return_constraints=0)

# if only the constraint violation should be returned - vector of zeros if no constraints exist
from pymop.welded_beam import WeldedBeam

problem = WeldedBeam()
F, CV = problem.evaluate(np.random.random((100, 4)), return_constraints=2)

```


Problems
------

In this package single- as well as multi-objective test problems are included.

* Single-Objective:

    * Ackley
    * BNH
    * Griewank
    * Knapsack
    * Schwefel
    * Sphere
    * Zakharov

* Multi-Objective:

    * DTLZ 1-7
    * ZDT 1-6
    * Carside Impact
    * BNH
    * Kursawe
    * OSY
    * TNK
    * Welded Beam



Implementation
------

All problems are implemented to efficiently evaluate multiple input points at a time.
Therefore, the input can be a n x m dimensional matrix, where n is the number of points to evaluate and m the number of variables.


Contact
------
Feel free to contact me if you have any question: 
blankjul@egr.msu.edu
