import math
import pytest
import numpy as np

from .context import falcon
from falcon import mesh_tools as mt
from falcon import bdm_basis as bdm
from falcon import dof_handler as dof

def test_dof_handler_1(basic_mesh_bdm1):
    dof_handler = basic_mesh_bdm1
    assert dof_handler.get_global_dof_idx() == tuple(range(10))
    assert dof_handler.get_num_node_dof() == 0
    assert dof_handler.get_num_edge_dof() == 10
    assert dof_handler.get_num_interior_dof() == 0
    
def test_dof_handler_2(basic_mesh_p1):
    dof_handler = basic_mesh_p1
    assert dof_handler.get_global_dof_idx() == tuple(range(4))
    assert dof_handler.get_num_node_dof() == 4
    assert dof_handler.get_num_edge_dof() == 0
    assert dof_handler.get_num_interior_dof() == 0

def test_dof_handler_3(basic_mesh_bdm1):
    dof_handler = basic_mesh_bdm1
    import pdb ; pdb.set_trace()
    assert dof_handler.get_local_2_global(0,0) == 1
    assert dof_handler.get_local_2_global(0,1) == 2
    assert dof_handler.get_local_2_global(0,2) == 0
    assert dof_handler.get_local_2_global(0,3) == 6
    assert dof_handler.get_local_2_global(0,4) == 7
    assert dof_handler.get_local_2_global(0,5) == 5

    assert dof_handler.get_local_2_global(1,0) == 4
    assert dof_handler.get_local_2_global(1,1) == 2
    assert dof_handler.get_local_2_global(1,2) == 3
    assert dof_handler.get_local_2_global(1,3) == 9
    assert dof_handler.get_local_2_global(1,4) == 7
    assert dof_handler.get_local_2_global(1,5) == 8

def test_dof_handler_4(basic_mesh_p1):
    dof_handler = basic_mesh_p1
    assert dof_handler.get_local_2_global(0,0) == 2
    assert dof_handler.get_local_2_global(0,1) == 0
    assert dof_handler.get_local_2_global(0,2) == 1

    assert dof_handler.get_local_2_global(1,0) == 1
    assert dof_handler.get_local_2_global(1,1) == 3
    assert dof_handler.get_local_2_global(1,2) == 2    

def test_dof_handler_5(basic_mesh_bdm2):
    dof_handler = basic_mesh_bdm2
    assert dof_handler.get_global_dof_idx() == tuple(range(21))
    assert dof_handler.get_num_node_dof() == 0
    assert dof_handler.get_num_edge_dof() == 15
    assert dof_handler.get_num_interior_dof() == 6

    E0_expected = [1,2,0,6,7,5,11,12,10,15,17,19]
    E1_expected = [4,2,3,9,7,8,14,12,13,16,18,20]

    for i in range(dof_handler.get_num_dof_per_element()):
        assert dof_handler.get_local_2_global(0,i) == E0_expected[i]
        assert dof_handler.get_local_2_global(1,i) == E1_expected[i]

def test_mixed_dof_handler_1(basis_mesh_bdm1_p0):
    dof_handler = basis_mesh_bdm1_p0
    assert dof_handler.get_global_dof_idx() == tuple(range(12))
    assert dof_handler.get_num_node_dof() == 0
    assert dof_handler.get_num_edge_dof() == 10
    assert dof_handler.get_num_interior_dof() == 2

def test_mixed_dof_handler_2(basis_mesh_bdm1_p0):
    dof_handler = basis_mesh_bdm1_p0
    assert np.array_equal(dof_handler._l2g_map[0],[1, 2, 0, 6, 7, 5, 10])
    assert np.array_equal(dof_handler._l2g_map[1],[4, 2, 3, 9, 7, 8, 11])

def test_mixed_dof_handler_3(basis_mesh_bdm2_p1):
    dof_handler = basis_mesh_bdm2_p1
    assert dof_handler.get_global_dof_idx() == tuple(range(25))
    assert dof_handler.get_num_node_dof() == 4
    assert dof_handler.get_num_edge_dof() == 15
    assert dof_handler.get_num_interior_dof() == 6

def test_mixed_dof_handler_4(basis_mesh_bdm2_p1):
    dof_handler = basis_mesh_bdm2_p1
    assert np.array_equal(dof_handler._l2g_map[0],
                          [1,2,0,6,7,5,11,12,10,15,17,19,23,21,22])
    assert np.array_equal(dof_handler._l2g_map[1],
                          [4,2,3,9,7,8,14,12,13,16,18,20,22,24,23])

@pytest.mark.bdm
def test_mixed_tens_dof_handler_1(basis_mesh_bdm1tens_p0vec):
    dof_handler = basis_mesh_bdm1tens_p0vec
    assert np.array_equal(dof_handler._l2g_map[0],
                          [2,0,1,7,5,6,12,10,11,17,15,16,20,22])
    assert np.array_equal(dof_handler._l2g_map[1],
                          [4,3,2,9,8,7,14,13,12,19,18,17,21,23])

@pytest.mark.bdm
def test_mixed_tens_dof_handler_1(basic_mesh_bdm1_p0_elasticity):
    basis, mesh, dof_handler = basic_mesh_bdm1_p0_elasticity
    assert np.array_equal(dof_handler._l2g_map[0],
                          [2,0,1,7,5,6,12,10,11,17,15,16,20,22,24])
    assert np.array_equal(dof_handler._l2g_map[1],
                          [4,3,2,9,8,7,14,13,12,19,18,17,21,23,25])

