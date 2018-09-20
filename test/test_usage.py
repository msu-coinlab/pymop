import unittest


class UsageTest(unittest.TestCase):

    def test_usage(self):
        from pymop.usage import evaluate
        evaluate()
        
        

if __name__ == '__main__':
    unittest.main()
