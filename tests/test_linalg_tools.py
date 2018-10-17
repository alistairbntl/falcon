import pytest
import numpy as np

from falcon import linalg_tools as la

def test_global_matrix_assembler_1():
    global_matrix_assembler = la.GlobalMatrix()

    global_matrix_assembler.add_new_entry(0,0,3.2)
    global_matrix_assembler.add_new_entry(0,3,1.2)
    global_matrix_assembler.add_new_entry(1,1,4.4)
    
    global_matrix_assembler.set_sparse_arrays()

    assert np.array_equal(global_matrix_assembler._np_row, np.array([0,0,1]))
    assert np.array_equal(global_matrix_assembler._np_col, np.array([0,3,1]))
    assert np.array_equal(global_matrix_assembler._np_val, np.array([3.2,1.2,4.4]))

def test_dot_product_1():
    v1 = (1,0) ; v2 = (0,1)
    assert la.Operators.dot_product(v1,v2) == 0
    v1 = (1,0) ; v2 = (1,1)
    assert la.Operators.dot_product(v1,v2) == 1
    v1 = (1,1) ; v2 = (1,1)
    assert la.Operators.dot_product(v1,v2) == 2
    v1 = (0.1,0.5) ; v2 = (0.1, 0.5)
    assert la.Operators.dot_product(v1,v2) == 0.26
    v1 = (1,1,2) ; v2 = (0.1,0.1,2)
    assert la.Operators.dot_product(v1,v2) == 4.2
    
def test_deviatoric():
    sig1 = np.ones(shape=(2,2)) ; sig1 = la.Operators.deviatoric(sig1)
    assert np.allclose(sig1, np.array([[0., 1.],[1., 0.]]))
    sig1 = np.identity(3) 
    sig1 = la.Operators.deviatoric(sig1)
    assert np.allclose(sig1, np.array([[0., 0., 0.],
                                       [0., 0., 0.],
                                       [0., 0., 0.]]))
    sig1 = np.array([[1,2],[3,4]])
    sig1 = la.Operators.deviatoric(sig1)
    assert np.allclose(sig1, np.array([[-3./2., 2],
                                       [3, 3./2.]]))
    sig1 = np.array([[4,3],[2,1]])
    sig1 = la.Operators.deviatoric(sig1)
    assert np.allclose(sig1, np.array([[3./2., 3],
                                       [2, -3./2.]]))

@pytest.mark.bdm
def test_cartesian_elasticity_tens_tens():
    sig1 = np.ones(shape=(2,2)) ; sig2 = np.ones(shape=(2,2))
    output = la.Operators.cartesian_elasticity_tens_tens(sig1,
                                                         sig2)
    assert output == 3./2.
    sig1 = np.array([[1,2],[3,4]]) ; sig2 = np.array([[4,3],[2,1]])
    output = la.Operators.cartesian_elasticity_tens_tens(sig1,
                                                         sig2)
    assert output == 6.875

@pytest.mark.bdm
def test_cartesian_elasticity_weak_symmetry():
    sig1 = np.ones(shape=(2,2)) ; sig2 = np.array([[0,1.],[-1.,0]])
    output = la.Operators.weak_symmetry_dot_product(sig1,
                                                    sig2)
    assert output == 0.
    sig1 = np.array([[3.,2.],[1.,4.]])
    output = la.Operators.weak_symmetry_dot_product(sig1,
                                                    sig2)
    assert output == 1
