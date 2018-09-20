import os
import sys
import unittest

# add the path to be execute in the main directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

testmodules = [
    'tests.test_correctness',
    'tests.test_usage'
]

suite = unittest.TestSuite()

for t in testmodules:
    try:
        # If the module defines a suite() function, call it to get the suite.
        mod = __import__(t, globals(), locals(), ['suite'])
        suitefn = getattr(mod, 'suite')
        suite.addTest(suitefn())
    except (ImportError, AttributeError):
        # else, just load all the tests cases from the module.
        suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

unittest.TextTestRunner().run(suite)