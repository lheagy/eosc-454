import unittest
import os
import testipynb

TESTDIR = os.path.abspath(__file__)

NBDIR = os.path.sep.join(TESTDIR.split(os.path.sep)[:-2] + ["lectures", "week-2"])
Test = testipynb.TestNotebooks(directory=NBDIR, timeout=2100)
TestWeek2 = Test.get_tests()

if __name__ == "__main__":
    unittest.main()
