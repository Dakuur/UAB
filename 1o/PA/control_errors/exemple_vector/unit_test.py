import unittest
import vector as v


class testVector(unittest.TestCase):
    def test_str_empty(self):
        v1 = Vector([])
        self.assertEqual("[]", str(v1))

    def test_str_single_element(self):
        v1 = Vector([1])
        self.assertEqual("[1]", str(v1))

    def test_str_multiple_element(self):
        v1 = Vector([1, 2, 3])
        self.assertEqual("[1,2,3]", str(v1))


"""
class testVector(unittest.TestCase):
    def test_str_empty(self):
        v1 = Vector([])
        self.assertEqual("[]", str(v1))
    def test_str_single_element(self):
        v1 = Vector([1])
        self.assertEqual("[1]", str(v1))
    def test_str_multiple_element(self):
        v1 = Vector([1, 2, 3])
        self.assertEqual("[1,2,3]", str(v1))
    def test_add_empty(self):
        v1 = Vector([])
        v2 = Vector([])
        resultat = v1 + v2
        self.assertEqual("[]", str(resultat))
    def test_add_not_equal_length(self):
        v1 = Vector([])
        v2 = Vector([1])
        with self.assertRaises(AssertionError):
            resultat = v1 + v2
"""

"""
class testVector(unittest.TestCase):
    def setUp(self):
        self.empty = Vector([])
        self.single_element = Vector([1])
        self.multiple_element = Vector([1,2,3])

    def test_str_empty(self):
         self.assertEqual("[]", str(self.empty))

    def test_str_single_element(self):
         self.assertEqual("[1]", str(self.single_element))

    def test_str_multiple_element(self):
         self.assertEqual("[1,2,3]", str(self.multiple_element))

    def test_add_empty(self):
         resultat = self.empty + self.empty
         self.assertEqual("[]", str(resultat))

    def test_add_not_equal_length(self):
         with self.assertRaises(AssertionError):
             self.empty + self.single_element
         with self.assertRaises(AssertionError):
             self.multiple_element + self.single_element
"""

"""
class testVector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.empty = Vector([])
        cls.single_element = Vector([1])
        cls.multiple_element = Vector([1,2,3])

    def test_str_empty(self):
         self.assertEqual("[]", str(self.empty))

    def test_str_single_element(self):
         self.assertEqual("[1]", str(self.single_element))

    def test_str_multiple_element(self):
         self.assertEqual("[1,2,3]", str(self.multiple_element))

    def test_add_empty(self):
         resultat = self.empty + self.empty
         self.assertEqual("[]", str(resultat))

    def test_add_not_equal_length(self):
         with self.assertRaises(AssertionError):
             self.empty + self.single_element
         with self.assertRaises(AssertionError):
             self.multiple_element + self.single_element
"""

import unittest
from vector import Vector

class testVector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.empty = Vector([])
        cls.single_element = Vector([1])
        cls.multiple_element = Vector([1, 2, 3])

    def test_str_empty(self):
        self.assertEqual("[]", str(self.empty))

    def test_str_single_element(self):
        self.assertEqual("[1]", str(self.single_element))

    def test_str_multiple_element(self):
        self.assertEqual("[1, 2, 3]", str(self.multiple_element))

    def test_sub_empty(self):
        result = self.empty - self.empty
        self.assertEqual("[]", str(result))

    def test_sub_not_equal_length(self):
        with self.assertRaises(AssertionError):
            self.empty - self.single_element
        with self.assertRaises(AssertionError):
            self.multiple_element - self.single_element

    def test_sub_valid(self):
        result = self.multiple_element - self.single_element
        self.assertEqual("[0, 1, 2]", str(result))
        result = self.single_element - self.multiple_element
        self.assertEqual("[-1, -2, -3]", str(result))
        result = self.multiple_element - self.multiple_element
        self.assertEqual("[0, 0, 0]", str(result))
        result = Vector([1, 2, 3, 4]) - Vector([4, 3, 2, 1])
        self.assertEqual("[-3, -1, 1, 3]", str(result))

    def test_mul_scalar(self):
        result = self.multiple_element * 2
        self.assertEqual("[2, 4, 6]", str(result))
        result = 2 * self.multiple_element
        self.assertEqual("[2, 4, 6]", str(result))
        result = self.single_element * -3
        self.assertEqual("[-3]", str(result))
        result = 0 * self.multiple_element
        self.assertEqual("[0, 0, 0]", str(result))

    def test_mul_invalid_type(self):
        with self.assertRaises(TypeError):
            self.multiple_element * "a"
        with self.assertRaises(TypeError):
            "a" * self.multiple_element
        with self.assertRaises(TypeError):
            self.multiple_element * [1, 2]
        with self.assertRaises(TypeError):
            [1, 2] * self.multiple_element

if __name__ == "__main__":
    unittest.main()
