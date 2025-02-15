# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020 D. Craig Brinck, SE; tamalone1
"""

import unittest
from PyNite import FEModel3D
from datetime import datetime

import numpy as np

class Test_Vector_Results(unittest.TestCase):
    """Tests for reaction results

    :param unittest: _description_
    :type unittest: _type_
    """

    # def setUp(self):
    #     # Suppress printed output temporarily
    #     sys.stdout = StringIO()

    # def tearDown(self):
    #     # Reset the print function to normal
    #     sys.stdout = sys.__stdout__

    def get_test_model(self):
        model = FEModel3D()

        model.add_node('N1', 0, 0, 0)
        model.add_node('N2', 7, 0, 0)
        model.add_node('N3', 12, 0, 0)
        model.add_auxnode('AN1', 3.5, 0, 1)

        model.add_material("steel", 200e9, 80e9, 0.2, 24000)
        model.add_section("sec", 1e-3, 1e-6, 1e-6, 1e-6)
        model.add_member("M1", "N1", "N2", "steel", "sec", aux_node="AN1")
        model.add_member("M2", "N2", "N3", "steel", "sec", aux_node="AN1")

        model.def_support('N1', True, True, True, True, True, True)
        model.def_support('N2', True, True, True)
        model.def_support('N3', True, True, True)

        model.add_load_combo("Combo 1",{"G":1.13, "Q":1.54})

        return model

    def test_shear_arrays(self):
        model = self.get_test_model()

        model.add_member_pt_load("M1", "Fy", 1000, 4.5, "G")
        model.add_member_pt_load("M1", "Fz", 1000, 6.5, "Q")
        model.add_member_pt_load("M1", "FZ", 1200, 3, "G")
        model.add_member_pt_load("M1", "FY", 1500, 5, "Q")

        model.add_member_dist_load("M1","Fy", 0.5,3, 1, 5, "Q")
        model.add_member_dist_load("M1","FZ", 5,3, 0, 5.2, "G")
        model.add_member_dist_load("M1","FY", 0.5,-3, 0, 7, "G")
        model.add_member_dist_load("M1","Fz", -9,-3, 2, 7, "Q")

        model.analyze_linear()

        mem = model.members['M1']

        xvals, shear_array = mem.shear_array('Fy', 'Combo 1', 50)
        ref_shears = np.array([mem.shear("Fy", x) for x in xvals])
        self.assertTrue(np.all(np.isclose(shear_array, ref_shears)))

        xvals, shear_array = mem.shear_array('Fz', 'Combo 1', 50)
        ref_shears = np.array([mem.shear("Fz", x) for x in xvals])
        self.assertTrue(np.all(np.isclose(shear_array, ref_shears)))

    def test_moment_arrays(self):
        model = self.get_test_model()

        model.add_member_pt_load("M1", "Fy", 1000, 4.5, "G")
        model.add_member_pt_load("M1", "Fz", 1000, 6.5, "Q")
        model.add_member_pt_load("M1", "FZ", 1200, 3, "G")
        model.add_member_pt_load("M1", "FY", 1500, 5, "G")
        model.add_member_pt_load("M1", "MZ", 1500, 5, "Q")
        model.add_member_pt_load("M1", "My", 900, 0.8, "G")
        model.add_member_pt_load("M1", "MY", -1200, 3, "Q")
        model.add_member_pt_load("M1", "Mz", 3800, 4.5, "G")
        model.add_member_pt_load("M1", "My", 750, 0, "Q")

        model.add_member_dist_load("M1","Fy", 3,3, 1, 5, "Q")
        model.add_member_dist_load("M1","FZ", -5,5, 2, 5.2, "G")
        model.add_member_dist_load("M1","FY", 1,-6, 0, 7, "G")
        model.add_member_dist_load("M1","Fz", 12,6, 0, 7, "G")
        model.add_member_dist_load("M1","Fz", -9,-9, 2, 7, "Q")

        model.add_member_dist_load("M1","Fy", 3,-5, 1, 5, "Q")
        model.add_member_dist_load("M1","FZ", 4,5, 2, 5.2, "G")
        model.add_member_dist_load("M1","FY", 0.5,6, 0, 7, "G")
        model.add_member_dist_load("M1","Fz", -9,-9, 0, 7, "Q")

        model.analyze_linear()

        mem = model.members['M1']

        xvals, mom_array = mem.moment_array('My', 'Combo 1', 50)
        ref_moments = np.array([mem.moment("My", x) for x in xvals])
        self.assertTrue(np.all(np.isclose(mom_array, ref_moments)))

        xvals, mom_array = mem.moment_array('Mz', 'Combo 1', 50)
        ref_moments = np.array([mem.moment("Mz", x) for x in xvals])
        self.assertTrue(np.all(np.isclose(mom_array, ref_moments)))

    def test_deflection_arrays(self):
        model = self.get_test_model()

        model.add_member_pt_load("M1", "Fz", 1000, 6.5, "Q")
        model.add_member_pt_load("M1", "FZ", 1200, 3, "G")
        model.add_member_pt_load("M1", "FY", 1500, 5, "G")
        model.add_member_pt_load("M1", "MZ", 1500, 5, "Q")
        model.add_member_pt_load("M1", "My", 900, 0.8, "G")
        model.add_member_pt_load("M1", "MY", -1200, 3, "Q")
        model.add_member_pt_load("M1", "Mz", 3800, 4.5, "G")
        model.add_member_pt_load("M1", "My", 750, 0, "Q")

        model.add_member_dist_load("M1","Fy", 3,3, 1, 5, "Q")
        model.add_member_dist_load("M1","FZ", -5,5, 2, 5.2, "G")
        model.add_member_dist_load("M1","FY", 1,-6, 0, 7, "G")
        model.add_member_dist_load("M1","Fz", 12,6, 0, 7, "G")
        model.add_member_dist_load("M1","Fz", -9,-9, 2, 7, "Q")

        model.add_member_dist_load("M1","Fy", 3,-5, 1, 5, "Q")
        model.add_member_dist_load("M1","FZ", 4,5, 2, 5.2, "G")
        model.add_member_dist_load("M1","FY", 0.5,6, 0, 7, "G")
        model.add_member_dist_load("M1","Fz", -9,-9, 0, 7, "Q")

        model.analyze_linear()

        mem = model.members['M1']

        xvals, def_array = mem.deflection_array('dy', 'Combo 1', 50)
        ref_deflections = np.array([mem.deflection("dy", x) for x in xvals])
        # print(def_array)
        # print(ref_deflections)
        self.assertTrue(np.all(np.isclose(def_array, ref_deflections)))

        xvals, def_array = mem.deflection_array('dz', 'Combo 1', 50)
        ref_deflections = np.array([mem.deflection("dz", x) for x in xvals])
        # print(def_array)
        # print(ref_deflections)
        self.assertTrue(np.all(np.isclose(def_array, ref_deflections)))

    def test_performance(self):
        model = self.get_test_model()

        model.add_member_pt_load("M1", "Fz", 1000, 6.5, "Q")
        model.add_member_pt_load("M1", "FZ", 1200, 3, "G")
        model.add_member_pt_load("M1", "FY", 1500, 5, "G")
        model.add_member_pt_load("M1", "MZ", 1500, 5, "Q")
        model.add_member_pt_load("M1", "My", 900, 0.8, "G")
        model.add_member_pt_load("M1", "MY", -1200, 3, "Q")
        model.add_member_pt_load("M1", "Mz", 3800, 4.5, "G")
        model.add_member_pt_load("M1", "My", 750, 0, "Q")

        model.add_member_dist_load("M1","Fy", 3,3, 1, 5, "Q")
        model.add_member_dist_load("M1","FZ", -5,5, 2, 5.2, "G")
        model.add_member_dist_load("M1","FY", 1,-6, 0, 7, "G")
        model.add_member_dist_load("M1","Fz", 12,6, 0, 7, "G")
        model.add_member_dist_load("M1","Fz", -9,-9, 2, 7, "Q")

        model.add_member_dist_load("M1","Fy", 3,-5, 1, 5, "Q")
        model.add_member_dist_load("M1","FZ", 4,5, 2, 5.2, "G")
        model.add_member_dist_load("M1","FY", 0.5,6, 0, 7, "G")
        model.add_member_dist_load("M1","Fz", -9,-9, 0, 7, "Q")

        model.analyze_linear()

        mem = model.members['M1']

        stime = datetime.now()
        for i in range(1000):
            # xvals, def_array = mem.deflection_array('dy', 'Combo 1', 50)
            xvals, def_array = mem.moment_array('My', 'Combo 1', 50)
        array_time = datetime.now() - stime

        stime = datetime.now()
        for i in range(1000):
            # ref_deflections = np.array([mem.deflection("dy", x) for x in xvals])
            ref_deflections = np.array([mem.moment("My", x) for x in xvals])
        segment_time = datetime.now() - stime

        print("Array extraction complete in",array_time)
        print("Segment extraction complete in",segment_time)
        print("Performance factor:", segment_time.microseconds/array_time.microseconds)

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromName("test_vector_results")
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

