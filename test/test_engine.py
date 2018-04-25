import unittest
from test import support

class HarmonicModelTest(unittest.TestCase):

    # Only use setUp() and tearDown() if necessary

    def setUp(self):
        #... code to execute in preparation for tests ...

    def tearDown(self):
        #... code to execute to clean up after tests ...

    def test_feature_one(self):
        # Test feature one.
        #... testing code ...

    def test_feature_two(self):
        # Test feature two.
        #... testing code ...

    #... more test methods ...

class MyTestCase2(unittest.TestCase):
    #... same structure as MyTestCase1 ...

#... more test classes ...

if __name__ == '__main__':
    unittest.main()