import unittest


class test_nothing(unittest.TestCase):
    def test_nothing(self):
        a = [1, 2, 3]
        b = [1, 2, 3]
        return self.assertEqual(a, b)
