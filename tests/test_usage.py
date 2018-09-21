import os
import unittest


class UsageTest(unittest.TestCase):

    def test(self):

        USAGE_DIR = "../pymop/usage"

        for fname in os.listdir(USAGE_DIR):

            with open(os.path.join(USAGE_DIR, fname)) as f:
                s = f.read()

                try:
                    exec(s)
                except:
                    raise Exception("Usage %s failed." % fname)



if __name__ == '__main__':
    unittest.main()
