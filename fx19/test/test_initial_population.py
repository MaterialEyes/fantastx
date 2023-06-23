"""
Tests for initial_population.py
"""

from unittest import TestCase
import unittest
import unittest.mock as mock
from fx19.initial_population import make_model_from_input
# from fx19.initial_population import make_model_from_input
# from pymatgen.util.testing import PymatgenTest


class TestMakeInputModels(TestCase):
    def test_nothing(self):
        a = [1, 2, 3]
        b = [1, 2, 3]
        return self.assertEqual(a, b)

    def test_set_up(self):
        with mock.patch('os.listdir') as mocked_listdir:
            mocked_listdir.return_value = ['POSCARone',
                                           'POSCAR_two',
                                           'test.cif',
                                           'trial.xyz']
            expected = ['POSCARone', 'POSCAR_two', 'test.cif']
            input_model_maker = make_model_from_input("")
            self.assertEqual(input_model_maker.all_files, expected)


# class TestMakeRandomMoleculeModel(TestCase):
#     """Test the MakeRandomMoleculeModel module"""

#     # def setUp(self):
#     #     self.random_molecule = make_random_molecule_model()

#     def test_get_cluster_in_box(self):
#         pass

#     def test_random_model(self):
#         pass

#     def test_get_n_species(self):
#         pass

#     def test_get_thickness(self):
#         pass

#     def test_add_vac(self):
#         pass

#     def test_get_n_coords_linear(self):
#         pass

#     def test_get_point_on_sphere(self):
#         pass

# class TestMakeRandomModel(TestCase):
#     """ Test the MakeRandomModel module"""


if __name__ == '__main__':
    unittest.main()
