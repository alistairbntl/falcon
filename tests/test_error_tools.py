import math
import pytest
import numpy as np

from .context import falcon
from falcon import error_tools as et
from falcon import mesh_tools as mt
from falcon import linalg_tools as lat
from falcon import function_tools as ft

@pytest.mark.error
def test_element_solution_1(basic_mesh_P0):
    basis, mesh, dof_handler = basic_mesh_P0
    dof_lst = [term.get_num_dof() for term in basis]
    num_dof = sum(dof_lst)
    sol = lat.DiscreteSolutionVector(num_dof)
    sol.set_solution(np.array([1.5,-1.]))
    error_handler = et.SolutionHandler(mesh,
                                       dof_handler,
                                       basis,
                                       sol)
    assert error_handler.get_num_bases() == 1
    assert error_handler.get_basis_dof_indices(0) == [0,1]

    for i, element in enumerate(mesh.get_elements()):        
        eval_point = element.get_center()
        quad_pt = mt.QuadraturePoint(eval_point[0],
                                     eval_point[1],
                                     1.)
        a = error_handler.get_element_solution_approx(element,
                                                      0,
                                                      quad_pt)
        if i==0:
            assert a==1.5
        if i==1:
            assert a==-1.

@pytest.mark.error
def test_element_solution_2(basis_mesh_bdm1_p0):
    basis, mesh, dof_handler = basis_mesh_bdm1_p0

    dof_lst = [term.get_num_dof() for term in basis]
    num_dof = sum(dof_lst)
    sol = lat.DiscreteSolutionVector(num_dof)
    sol.set_solution(np.array([0.,0.,0.,0.,0.,0.,
                               0.,0.,0.,0.,1.5,-1.]))
    error_handler = et.SolutionHandler(mesh,
                                       dof_handler,
                                       basis,
                                       sol)
    assert error_handler.get_num_bases() == 2
    assert np.array_equal(error_handler.get_basis_dof_indices(0), [0,6])
    assert np.array_equal(error_handler.get_basis_dof_indices(1), [6,7])

    for i, element in enumerate(mesh.get_elements()):        
        eval_point = element.get_center()
        quad_pt = mt.QuadraturePoint(eval_point[0],
                                     eval_point[1],
                                     1.)
        val = error_handler.get_element_solution_approx(element,
                                                        1,
                                                        quad_pt)
        if i==0:
            assert val==1.5
        if i==1:
            assert val==-1.

@pytest.mark.error
def test_element_solution_3(basic_mesh_p0vec):
    basis, mesh, dof_handler = basic_mesh_p0vec

    dof_lst = [term.get_num_dof() for term in basis]
    num_dof = sum(dof_lst)
    sol = lat.DiscreteSolutionVector(num_dof)
    sol.set_solution(np.array([0.,1.,2.,3.]))

    error_handler = et.SolutionHandler(mesh,
                                       dof_handler,
                                       basis,
                                       sol)

    assert error_handler.get_num_bases() == 1
    assert np.array_equal(error_handler.get_basis_dof_indices(0), [0, 2])

    for i, element in enumerate(mesh.get_elements()):
        eval_point = element.get_center()
        quad_pt = mt.QuadraturePoint(eval_point[0],
                                     eval_point[1],
                                     1.)
        val = error_handler.get_element_solution_approx(element,
                                                        0,
                                                        quad_pt)
        if i==0:
            assert np.array_equal(val, [0,2])
        if i==1:
            assert np.array_equal(val, [1,3])

@pytest.mark.error
def test_element_solution_4(basic_mesh_bdm1_p0_elasticity):
    basis, mesh, dof_handler = basic_mesh_bdm1_p0_elasticity

    dof_lst = [term.get_num_dof() for term in basis]
    num_dof = dof_handler.get_num_dof()
    sol = lat.DiscreteSolutionVector(num_dof)
    sol_array = np.zeros(num_dof)
    sol_array[20] = 1. ; sol_array[22] = 4.
    sol_array[21] = -1. ; sol_array[23] = -2.
    sol.set_solution(sol_array)

    error_handler = et.SolutionHandler(mesh,
                                       dof_handler,
                                       basis,
                                       sol)

    assert error_handler.get_num_bases() == 3
    assert np.array_equal(error_handler.get_basis_dof_indices(0), [0,12])
    assert np.array_equal(error_handler.get_basis_dof_indices(1), [12,14])
    assert np.array_equal(error_handler.get_basis_dof_indices(2), [14,15])

    for i, element in enumerate(mesh.get_elements()):
        eval_point = element.get_center()
        quad_pt = mt.QuadraturePoint(eval_point[0],
                                     eval_point[1],
                                     1.)
        val = error_handler.get_element_solution_approx(element,
                                                        1,
                                                        quad_pt)
        if i == 0:
            assert np.array_equal(val, [1.,4.])
        if i == 1:
            assert np.array_equal(val, [-1.,-2.])

@pytest.mark.error
def test_element_solution_5(basic_mesh_bdm1_p0_elasticity):
    basis, mesh, dof_handler = basic_mesh_bdm1_p0_elasticity

    dof_lst = [term.get_num_dof() for term in basis]
    num_dof = dof_handler.get_num_dof()

    q1 = 0.5 - math.sqrt(3) / 6. ; q2 = 0.5 + math.sqrt(3) / 6.
    const = -1. / (q2 - q1)
    true_stress = ft.TrueSolution([lambda x,y : const * (q2*x + y - q2),
                                   lambda x,y : const * (q2-1) * y,
                                   lambda x,y : 0.,
                                   lambda x,y : 0.])

    true_displacement = ft.TrueSolution([lambda x,y : 4.,
                                         lambda x,y : 0.])
    
    sol = lat.DiscreteSolutionVector(num_dof)
    sol_array = np.zeros(num_dof)
    sol_array[0] = 1.
    sol_array[20] = 4. ; sol_array[22] = 0.
    sol_array[21] = 4. ; sol_array[23] = 0.
    sol.set_solution(sol_array)

    error_handler = et.ElasticityErrorHandler(mesh,
                                              dof_handler,
                                              basis,
                                              sol)

    err_dis = error_handler.calculate_displacement_error(true_displacement)
    err_stress = error_handler.calculate_stress_error(true_stress)
    assert err_dis == 0.
    assert abs(err_stress - 0.589482328) < 1.0e-6  # The error comes from
    # the second element not the first which is what I expect given the
    # function so this is okay.    
