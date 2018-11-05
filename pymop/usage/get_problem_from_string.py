from pymop.factory import get_problem

# create a simple test problem from string
p = get_problem("Ackley")

# the input name is not case sensitive
p = get_problem("ackley")

# also input parameter can be provided directly
p = get_problem("dtlz1", n_var=20, n_obj=5)
