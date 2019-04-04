pymop - Multi-Objective Optimization Problems
==========================================================================


|gitlab| |python| |license|


.. |gitlab| image:: https://gitlab.msu.edu/blankjul/pymop/badges/master/pipeline.svg
   :alt: pipeline status
   :target: https://gitlab.msu.edu/blankjul/pymop/commits/master

.. |python| image:: https://img.shields.io/badge/python-3.6-blue.svg
   :alt: python 3.6

.. |license| image:: https://img.shields.io/badge/license-apache-orange.svg
   :alt: license apache
   :target: https://www.apache.org/licenses/LICENSE-2.0



This framework provides a collection of test problems in Python. The main features are:

- Most important multi-objective test function is one place
- Vectorized evaluation by using numpy matrices (no for loops)
- Gradients and Hessian matrices are available through automatic differentiation
- Easily new problems can be created using custom classes or functions


Here, you can find a detailed documentation and information about the framework:
https://www.egr.msu.edu/coinlab/blankjul/pymop/




Problems
==================================

In this package single- as well as multi-objective test problems are
included:


-  Single-Objective:

   -  Ackley
   -  Cantilevered Beam
   -  Griewank
   -  Himmelblau
   -  Knapsack
   -  Pressure Vessel
   -  Schwefel
   -  Sphere
   -  Zakharov
   -  G1-10

-  Multi-Objective:

   -  ZDT 1-6 
   -  CTP 
   -  Carside Impact
   -  BNH
   -  Kursawe
   -  OSY
   -  TNK
   -  Truss 2D
   -  Welded Beam

-  Many-Objective:

   -  DTLZ 1-7 
   -  CDTLZ 
