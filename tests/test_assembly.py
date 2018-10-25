import math
import pytest
import numpy as np
import scipy.sparse.linalg as sp_la

from .context import falcon
from falcon import mesh_tools as mt
from falcon import mapping_tools as mp
from falcon import quadrature as quad
from falcon import bdm_basis as bdm
from falcon import dof_handler as dof
from falcon import linalg_tools as la
from falcon import function_tools as ft
from falcon import error_tools as ec

def test_assembly_1(basic_mesh_bdm):
    bdm_basis, mesh, dof_handler = basic_mesh_bdm
    quadrature = quad.Quadrature(1)
    reference_element = mt.ReferenceElement()

    num_mesh_elements = mesh.get_num_mesh_elements()
    num_edges_per_element = mesh.get_num_edges_per_element()
    num_dof_per_edge = bdm_basis.get_num_dof_per_edge()

    for eN in range(num_mesh_elements):

        element = mesh.get_element(eN)

        for edge_dof in range(num_dof_per_edge):
            for edge_num, edge in enumerate(element.get_edges()):

                dof_num = edge_dof*num_edges_per_element + edge_num

                func = bdm_basis.get_edge_normal_func(edge_num,
                                                      dof_num=dof_num)
                edge_quad_pt = reference_element.get_lagrange_quad_point(edge_num,
                                                                         edge_dof)
                assert 1 == func(edge_quad_pt[0],edge_quad_pt[1])

@pytest.mark.test
def test_assembly_2(basic_mesh_P0):
    P0_basis, mesh, dof_handler = basic_mesh_P0

    P0_basis = P0_basis[0]
    quadrature = quad.Quadrature(1)
    reference_element = mt.ReferenceElement()

    num_mesh_elements = mesh.get_num_mesh_elements()
    num_interior_dof = P0_basis.get_num_interior_dof()

    for eN in range(num_mesh_elements):

        element = mesh.get_element(eN)
        element_area = 0.
        reference_map = mp.ReferenceElementMap(element)

        for interior_dof in range(num_interior_dof):
            func = P0_basis.get_func(interior_dof)

            for quad_pt in quadrature.get_element_quad_pts():
                dV = quad_pt.get_quad_weight()*reference_map.get_jacobian_det()
                element_area += func(quad_pt[0],quad_pt[1])*dV

        assert element_area == 0.5

@pytest.mark.test
def test_assembly_3(basic_mesh_2_P0):
    P0_basis, mesh, dof_handler = basic_mesh_2_P0

    quadrature = quad.Quadrature(1)
    reference_element = mt.ReferenceElement()

    num_mesh_elements = mesh.get_num_mesh_elements()
    num_interior_dof = P0_basis.get_num_interior_dof()

    for eN in range(num_mesh_elements):

        element = mesh.get_element(eN)
        element_area = 0.
        reference_map = mp.ReferenceElementMap(element)

        for interior_dof in range(num_interior_dof):
            func = P0_basis.get_func(interior_dof)

            for quad_pt in quadrature.get_element_quad_pts():
                dV = quad_pt.get_quad_weight()*reference_map.get_jacobian_det()
                element_area += func(quad_pt[0],quad_pt[1])*dV

        assert element_area == element.get_area()

@pytest.mark.test
def test_mixed_assembly_1(basic_mesh_bdm1_p0):
    basis, mesh, dof_handler = basic_mesh_bdm1_p0

    num_local_dof = sum([a.get_num_dof() for a in basis])

    global_matrix_assembler = la.GlobalMatrix(dof_handler.get_num_dof(),
                                              dof_handler.get_num_dof())

    local_matrix_assembler = la.LocalMatrixAssembler(dof_handler,
                                                     num_local_dof,
                                                     global_matrix_assembler)

    solution_vec = la.DiscreteSolutionVector(dof_handler.get_num_dof())

    global_rhs = la.GlobalRHS(dof_handler)

    fx = lambda x,y : x*y - y**2
    fy = lambda x,y : x + x**2 - 0.5*y**2
    p = lambda x,y : 2*x + 3*y - 3./2.
    true_solution = ft.TrueSolution([fx,fy,p])
    forcing_function = ft.Function((lambda x,y: x*y - y**2 + 2,
                                    lambda x,y: x + x**2 - 0.5*y**2 + 3))

    test_space = basis ; trial_space = basis
    quadrature = quad.Quadrature(3)
    reference_element = mt.ReferenceElement()

    num_mesh_elements = mesh.get_num_mesh_elements()

    for eN in range(num_mesh_elements):
        element = mesh.get_element(eN)
        reference_map = mp.ReferenceElementMap(element)
        piola_map = mp.PiolaMap(element)
        value_types_test = ['vals','Jt_vals','|Jt|','quad_wght']
        value_types_trial = ['vals', 'Jt_vals']
        for quad_pt in quadrature.get_element_quad_pts():
            for i in range(test_space[0].get_num_dof()):
                val_dic_test = test_space[0].get_element_vals(i,
                                                              quad_pt,
                                                              reference_map,
                                                              value_types_test)
                val_dic_test = piola_map.correct_div_space_vals(val_dic_test,
                                                                i,
                                                                test_space[0])
                # rhs construction
                inv_val = 0
                ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                             quad_pt)
                vals = forcing_function.get_f_eval(ele_quad_pt)
                hs = (la.Operators.dot_product(vals, val_dic_test['Jt_vals']) * (1./val_dic_test['|Jt|']))
                int_val = (la.Operators.dot_product(vals,
                                                    val_dic_test['Jt_vals'])
                           * (1./val_dic_test['|Jt|'])
                           * val_dic_test['quad_wght'])
                global_rhs.add_val(eN, i, int_val)

                for j in range(trial_space[0].get_num_dof()):
                    int_val = 0
                    # matrix construction
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    reference_map,
                                                                    value_types_trial)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     j,
                                                                     trial_space[0])
                    int_val = (la.Operators.dot_product(val_dic_trial['Jt_vals'],
                                                        val_dic_test['Jt_vals']) *
                               (1./val_dic_test['|Jt|']) * val_dic_test['quad_wght'])
                    local_matrix_assembler.add_val(i,j,int_val)

        value_types_test = ['div','quad_wght']
        value_types_trial = ['vals']
        for i in range(test_space[0].get_num_dof()):
            for j in range(trial_space[1].get_num_dof()):
                int_val = 0.
                for quad_pt in quadrature.get_element_quad_pts():
                    val_dic_trial = trial_space[1].get_element_vals(j,
                                                                    quad_pt,
                                                                    reference_map,
                                                                    value_types_trial)
                    val_dic_test = test_space[0].get_element_vals(i,
                                                                  quad_pt,
                                                                  reference_map,
                                                                  value_types_test)
                    val_dic_test = piola_map.correct_div_space_vals(val_dic_test,
                                                                    i,
                                                                    test_space[0])
                    int_val = -(val_dic_trial['vals']
                                *val_dic_test['div']
                                *val_dic_test['quad_wght'])

                    j_tmp = j + trial_space[0].get_num_dof()
                    local_matrix_assembler.add_val(i,j_tmp,int_val)

        value_types_test = ['vals', 'quad_wght']
        value_types_trial = ['div']
        for i in range(test_space[1].get_num_dof()):
            for j in range(trial_space[0].get_num_dof()):
                int_val = 0.
                for quad_pt in quadrature.get_element_quad_pts():
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    reference_map,
                                                                    value_types_trial)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     j,
                                                                     trial_space[0])                    
                    val_dic_test = test_space[1].get_element_vals(i,
                                                                  quad_pt,
                                                                  reference_map,
                                                                  value_types_test)
                    int_val = (val_dic_trial['div']
                               *val_dic_test['vals']
                               *val_dic_test['quad_wght'])
                    i_tmp = i + test_space[0].get_num_dof()
                    local_matrix_assembler.add_val(i_tmp,j,int_val)

        local_matrix_assembler.distribute_local_2_global(eN)
        local_matrix_assembler.reset_matrix()

    assert abs(global_rhs._rhs_vec[5] + 0.43769752) < 1.0E-8
    assert abs(global_rhs._rhs_vec[1] - 0.64455656) < 1.0E-8
    assert abs(global_rhs._rhs_vec[2] - 1.56919528) < 1.0E-8
    assert abs(global_rhs._rhs_vec[0] - 0.52936418) < 1.0E-8
    assert abs(global_rhs._rhs_vec[6] - 0.19711010) < 1.0E-8
    assert abs(global_rhs._rhs_vec[7] - 1.05888492) < 1.0E-8

    assert dof_handler.get_bdy_dof_dic('bdm_basis') == set([(0,'e',0),
                                                            (1,'e',1),
                                                            (3,'e',3),
                                                            (4,'e',4),
                                                            (5,'e',0),
                                                            (6,'e',1),
                                                            (8,'e',3),
                                                            (9,'e',4)])

    global_matrix_assembler.initialize_sparse_matrix()
    global_matrix_assembler.set_csr_rep()
    
    assert abs(global_matrix_assembler._coo.toarray()[2][2] - 2./3.) < 1.0E-12
    assert abs(global_matrix_assembler._coo.toarray()[2][10] + math.sqrt(2)/2.) < 1.0E-12
    assert abs(global_matrix_assembler._coo.toarray()[2][11] - math.sqrt(2)/2.) < 1.0E-12    
    assert abs(global_matrix_assembler._coo.toarray()[7][7] - 2./3.) < 1.0E-12
    assert abs(global_matrix_assembler._coo.toarray()[7][10] + math.sqrt(2)/2.) < 1.0E-12
    assert abs(global_matrix_assembler._coo.toarray()[7][11] - math.sqrt(2)/2.) < 1.0E-12

    # ******************* Pin down the error to make system non-singular ****************
    global_matrix_assembler.set_row_as_dirichlet_bdy(dof_handler.get_num_dof() - 1)

    for basis_ele in trial_space:
        for boundary_dof in dof_handler.get_bdy_dof_dic(basis_ele.get_name()):
            dof_idx, bdy_type, bdy_type_global_idx = boundary_dof
            global_matrix_assembler.set_row_as_dirichlet_bdy(dof_idx)

            global_edge = mesh.get_edge(bdy_type_global_idx)
            local_edge_dof = dof_handler.get_local_edge_dof_idx(dof_idx,
                                                                bdy_type_global_idx)            
            quad_pt = quadrature.find_one_quad_on_edge(global_edge, local_edge_dof)
            n = global_edge.get_unit_normal_vec()
            udotn = true_solution.get_normal_velocity_func(n)
            val = udotn(quad_pt[0],quad_pt[1])
            global_rhs.set_value(dof_idx,val)

    global_matrix_assembler.solve(global_rhs, solution_vec)

    # ARB TO DO - I'm not sure if the following are correct, so I'm not including these
    # tests

    # expected_sol = np.array([-0.62200847,  2.67863279,  2.28435787, -0.25598306,
    #                          0.9106836 ,  -0.0446582 ,  4.98803387,  2.6653896 ,
    #                          -1.4106836 , -0.24401694,  -0.65      ,  0.        ])

    # assert np.allclose(expected_sol,solution_vec.get_solution())

    error_calculator = ec.DarcyErrorHandler(mesh,dof_handler,[basis[0]],solution_vec)
    l2_vel_error = error_calculator.calculate_vel_error(true_solution)

    # assert abs(l2_vel_error - xxx) < 1.0E-6

@pytest.mark.test
def test_mixed_assembly_4(basic_mesh_s1_bdm1_partial_r1):
    basis, mesh, dof_handler = basic_mesh_s1_bdm1_partial_r1

    num_local_dof = sum([a.get_num_dof() for a in basis])
    
    global_matrix_assembler = la.GlobalMatrix(dof_handler.get_num_dof(),
                                              dof_handler.get_num_dof())

    local_matrix_assembler = la.LocalMatrixAssembler(dof_handler,
                                                     num_local_dof,
                                                     global_matrix_assembler)

    solution_vec = la.DiscreteSolutionVector(dof_handler.get_num_dof())
    global_rhs = la.GlobalRHS(dof_handler)

    fx = lambda x,y : y
    fy = lambda x,y : 0.

    p_dx = lambda x,y : -y
    p_dy = lambda x,y : 0.

    true_vel = ft.TrueSolution([p_dx,p_dy])

    test_space = basis ; trial_space = basis
    quadrature = quad.Quadrature(2)
    reference_element = mt.ReferenceElement()

    num_mesh_elements = mesh.get_num_mesh_elements()
    for eN in range(num_mesh_elements):
        element = mesh.get_element(eN)
        reference_map = mp.ReferenceElementMap(element)
        piola_map = mp.PiolaMap(element)
        for quad_pt in quadrature.get_element_quad_pts():
            value_types_test = ['vals','Jt_vals','|Jt|','quad_wght']
            value_types_trial = ['vals','Jt_vals']
            for i in range(test_space[0].get_num_dof()):
                val_dic_test = test_space[0].get_element_vals(i,
                                                              quad_pt,
                                                              reference_map,
                                                              value_types_test)
                val_dic_test = piola_map.correct_div_space_vals(val_dic_test,
                                                                i,
                                                                test_space[0])
                # rhs construction
                int_val = 0.
                ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                             quad_pt)
                vals = true_vel.get_f_eval(ele_quad_pt)
                int_val = (la.Operators.dot_product(vals,
                                                    val_dic_test['Jt_vals'])
                           * (1./val_dic_test['|Jt|'])
                           * ele_quad_pt.get_quad_weight())
                global_rhs.add_val(eN, i, int_val)

                for j in range(trial_space[0].get_num_dof()):
                    int_val = 0.
                    # matrix construction
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    reference_map,
                                                                    value_types_trial)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     j,
                                                                     trial_space[0])
                    int_val = (la.Operators.dot_product(val_dic_trial['Jt_vals'],
                                                        val_dic_test['Jt_vals']) *
                               (1./val_dic_test['|Jt|']) * val_dic_test['quad_wght'])
                    local_matrix_assembler.add_val(i,j,int_val)

        local_matrix_assembler.distribute_local_2_global(eN)
        local_matrix_assembler.reset_matrix()

    global_matrix_assembler.initialize_sparse_matrix()
    global_matrix_assembler.set_csr_rep()

    global_matrix_assembler.solve(global_rhs, solution_vec)

    error_calculator = ec.DarcyErrorHandler(mesh,dof_handler,[basis[0]],solution_vec)

    l2_vel_error = error_calculator.calculate_vel_error(true_vel)
    assert abs(l2_vel_error) < 1.0e-12

def test_mixed_assembly_2(basic_mesh_s1_bdm1_p0):
    basis, mesh, dof_handler = basic_mesh_s1_bdm1_p0
    assert np.array_equal(dof_handler._l2g_map[0], [1,2,0,9,10,8,16])
    assert np.array_equal(dof_handler._l2g_map[1], [5.,4.,3.,13.,12.,11.,17])
    assert np.array_equal(dof_handler._l2g_map[2], [1.,5.,6.,9.,13.,14.,18])
    assert np.array_equal(dof_handler._l2g_map[3], [2.,4.,7.,10.,12.,15.,19])

    num_local_dof = sum([a.get_num_dof() for a in basis])
    assert num_local_dof == 7
    
    global_matrix_assembler = la.GlobalMatrix(dof_handler.get_num_dof(),
                                              dof_handler.get_num_dof())
    assert global_matrix_assembler.get_shape() == (20, 20)

    local_matrix_assembler = la.LocalMatrixAssembler(dof_handler,
                                                     num_local_dof,
                                                     global_matrix_assembler)
    assert local_matrix_assembler.get_num_local_dof() == 7

    solution_vec = la.DiscreteSolutionVector(dof_handler.get_num_dof())
    assert solution_vec.get_num_dof() == 20

    global_rhs = la.GlobalRHS(dof_handler)

    fx = lambda x,y : x*y - y**2
    fy = lambda x,y : x + x**2 - 0.5*y**2
    p = lambda x,y : 2*x + 3*y - 3./2.
    true_solution = ft.TrueSolution([fx,fy,p])
    dirichlet_forcing_function = ft.Function((lambda x,y: x*y - y**2,
                                              lambda x,y: x + x**2 - 0.5*y**2))
    forcing_function = ft.Function((lambda x,y: x*y - y**2 + 2,
                                    lambda x,y: x + x**2 - 0.5*y**2 + 3))    

    test_space = basis ; trial_space = basis
    quadrature = quad.Quadrature(3)
    reference_element = mt.ReferenceElement()

    num_mesh_elements = mesh.get_num_mesh_elements()
    assert num_mesh_elements == 4

    for eN in range(num_mesh_elements):
        element = mesh.get_element(eN)
        reference_map = mp.ReferenceElementMap(element)
        piola_map = mp.PiolaMap(element)
        for quad_pt in quadrature.get_element_quad_pts():
            value_types_test = ['vals','Jt_vals','|Jt|','quad_wght']
            value_types_trial = ['vals', 'Jt_vals']
            for i in range(test_space[0].get_num_dof()):
                val_dic_test = test_space[0].get_element_vals(i,
                                                              quad_pt,
                                                              reference_map,
                                                              value_types_test)
                val_dic_test = piola_map.correct_div_space_vals(val_dic_test,
                                                                i,
                                                                test_space[0])
                # rhs construction
                int_val = 0.
                ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                             quad_pt)
                vals = forcing_function.get_f_eval(ele_quad_pt)
                int_val = (la.Operators.dot_product(vals,
                                                    val_dic_test['Jt_vals'])
                           * (1./val_dic_test['|Jt|'])
                           * ele_quad_pt.get_quad_weight())
                global_rhs.add_val(eN, i, int_val)
                # You could run another test to make sure this is correct.

                for j in range(trial_space[0].get_num_dof()):
                    int_val = 0.
                    # matrix construction
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    reference_map,
                                                                    value_types_trial)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     j,
                                                                     trial_space[0])
                    int_val = (la.Operators.dot_product(val_dic_trial['Jt_vals'],
                                                        val_dic_test['Jt_vals']) *
                               (1./val_dic_test['|Jt|']) * val_dic_test['quad_wght'])
                    local_matrix_assembler.add_val(i,j,int_val)
                    # Write another test for the local matrix

            value_types_test = ['div','quad_wght']
            value_types_trial = ['vals']
            for i in range(test_space[0].get_num_dof()):
                for j in range(trial_space[1].get_num_dof()):
                    int_val = 0.
                    val_dic_trial = trial_space[1].get_element_vals(j,
                                                                    quad_pt,
                                                                    reference_map,
                                                                    value_types_trial)
                    val_dic_test = test_space[0].get_element_vals(i,
                                                                  quad_pt,
                                                                  reference_map,
                                                                  value_types_test)
                    val_dic_test = piola_map.correct_div_space_vals(val_dic_test,
                                                                    i,
                                                                    test_space[0])
                    int_val = -(val_dic_trial['vals']
                                *val_dic_test['div']
                                *val_dic_test['quad_wght'])
                    
                    j_tmp = j + trial_space[0].get_num_dof()
                    local_matrix_assembler.add_val(i,j_tmp,int_val)
                    # check the non velocity-velocity parts
                     
            value_types_test = ['vals', 'quad_wght']
            value_types_trial = ['div']
            for i in range(test_space[1].get_num_dof()):
                for j in range(trial_space[0].get_num_dof()):
                    int_val = 0.
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    reference_map,
                                                                    value_types_trial)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     j,
                                                                     trial_space[0])                    
                    val_dic_test = test_space[1].get_element_vals(i,
                                                                  quad_pt,
                                                                  reference_map,
                                                                  value_types_test)
                    int_val = -(val_dic_trial['div']
                               *val_dic_test['vals']
                               *val_dic_test['quad_wght'])
                    i_tmp = i + test_space[0].get_num_dof()
                    local_matrix_assembler.add_val(i_tmp,j,int_val)

        local_matrix_assembler.distribute_local_2_global(eN)
        local_matrix_assembler.reset_matrix()

    assert abs(global_rhs.get_rhs_vec()[0] + 0.47479816) <= 1.0e-6

    global_matrix_assembler.initialize_sparse_matrix()
    global_matrix_assembler.set_csr_rep()

    assert abs(global_matrix_assembler.get_csr_rep().diagonal()[0] - 1./6.) <= 1.0e-8
    assert abs(global_matrix_assembler.get_csr_rep().getrow(0).toarray()[0][16] + 1./2.) <= 1.0e-8
    assert abs(global_matrix_assembler.get_csr_rep().getrow(16).toarray()[0][0] + 1./2.) <= 1.0e-8
    assert np.allclose(global_matrix_assembler.get_csr_rep().toarray(),
                       global_matrix_assembler.get_csr_rep().toarray().T)

    # ******************* Pin down the pressure to make system non-singular ****************
    global_matrix_assembler.set_row_as_dirichlet_bdy(dof_handler.get_num_dof() - 1)
    assert sp_la.norm(global_matrix_assembler.get_csr_rep().getrow(dof_handler.get_num_dof()-1)[0]) == 1.

    for basis_ele in trial_space:
        for boundary_dof in dof_handler.get_bdy_dof_dic(basis_ele.get_name()):
            dof_idx, bdy_type, bdy_type_global_idx = boundary_dof
            global_matrix_assembler.set_row_as_dirichlet_bdy(dof_idx)

            global_edge = mesh.get_edge(bdy_type_global_idx)
            local_edge_dof = dof_handler.get_local_edge_dof_idx(dof_idx,
                                                                bdy_type_global_idx)
            quad_pt = quadrature.find_one_quad_on_edge(global_edge, local_edge_dof)
            n = global_edge.get_unit_normal_vec()
            udotn = dirichlet_forcing_function.get_normal_velocity_func(n)
            val = udotn(quad_pt[0],quad_pt[1])
            global_rhs.set_value(dof_idx,val)

    global_matrix_assembler.solve(global_rhs, solution_vec)

    error_calculator = ec.DarcyErrorHandler(mesh,dof_handler,[basis[0]],solution_vec)
    l2_vel_error = error_calculator.calculate_vel_error(true_solution)
    #ARB TODO - let's extend this further.

@pytest.mark.skip
def test_mixed_assembly_3(basic_mesh_s1_bdm1_p0_tmp):
    basis, mesh, dof_handler = basic_mesh_s1_bdm1_p0_tmp

    num_local_dof = sum([a.get_num_dof() for a in basis])

    global_matrix_assembler = la.GlobalMatrix(dof_handler.get_num_dof(),
                                              dof_handler.get_num_dof())

    local_matrix_assembler = la.LocalMatrixAssembler(dof_handler,
                                                     num_local_dof,
                                                     global_matrix_assembler)

    solution_vec = la.DiscreteSolutionVector(dof_handler.get_num_dof())
    global_rhs = la.GlobalRHS(dof_handler)

    fx = lambda x,y : y
    fy = lambda x,y : 0.
    p = lambda x,y : 0.
    true_solution = ft.TrueSolution([fx,fy,p])
    dirichlet_forcing_function = ft.Function((lambda x,y: y,
                                              lambda x,y: 0.))
    forcing_function = ft.Function((lambda x,y: y,
                                    lambda x,y: 0.))

    test_space = basis ; trial_space = basis
    quadrature = quad.Quadrature(2)
    reference_element = mt.ReferenceElement()

    num_mesh_elements = mesh.get_num_mesh_elements()
    for eN in range(num_mesh_elements):
        element = mesh.get_element(eN)
        reference_map = mp.ReferenceElementMap(element)
        piola_map = mp.PiolaMap(element)
        for quad_pt in quadrature.get_element_quad_pts():
            value_types_test = ['vals','Jt_vals','|Jt|','quad_wght']
            value_types_trial = ['vals', 'Jt_vals']
            for i in range(test_space[0].get_num_dof()):
                val_dic_test = test_space[0].get_element_vals(i,
                                                              quad_pt,
                                                              reference_map,
                                                              value_types_test)
                val_dic_test = piola_map.correct_div_space_vals(val_dic_test,
                                                                i,
                                                                test_space[0])
                # rhs construction
                int_val = 0.
                ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                             quad_pt)
                vals = forcing_function.get_f_eval(ele_quad_pt)
                int_val = (la.Operators.dot_product(vals,
                                                    val_dic_test['Jt_vals'])
                           * (1./val_dic_test['|Jt|'])
                           * ele_quad_pt.get_quad_weight())
                global_rhs.add_val(eN, i, int_val)

                for j in range(trial_space[0].get_num_dof()):
                    int_val = 0.
                    # matrix construction
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    reference_map,
                                                                    value_types_trial)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     j,
                                                                     trial_space[0])
                    int_val = (la.Operators.dot_product(val_dic_trial['Jt_vals'],
                                                        val_dic_test['Jt_vals']) *
                               (1./val_dic_test['|Jt|']) * val_dic_test['quad_wght'])
                    local_matrix_assembler.add_val(i,j,int_val)

            value_types_test = ['div','quad_wght']
            value_types_trial = ['vals']
            for i in range(test_space[0].get_num_dof()):
                for j in range(trial_space[1].get_num_dof()):
                    int_val = 0.
                    val_dic_trial = trial_space[1].get_element_vals(j,
                                                                    quad_pt,
                                                                    reference_map,
                                                                    value_types_trial)
                    val_dic_test = test_space[0].get_element_vals(i,
                                                                  quad_pt,
                                                                  reference_map,
                                                                  value_types_test)
                    val_dic_test = piola_map.correct_div_space_vals(val_dic_test,
                                                                    i,
                                                                    test_space[0])
                    int_val = -(val_dic_trial['vals']
                                *val_dic_test['div']
                                *val_dic_test['quad_wght'])
                    j_tmp = j + trial_space[0].get_num_dof()
                    local_matrix_assembler.add_val(i,j_tmp,int_val)

            value_types_test = ['vals', 'quad_wght']
            value_types_trial = ['div']
            for i in range(test_space[1].get_num_dof()):
                for j in range(trial_space[0].get_num_dof()):
                    int_val = 0.
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    reference_map,
                                                                    value_types_trial)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     j,
                                                                     trial_space[0])                    
                    val_dic_test = test_space[1].get_element_vals(i,
                                                                  quad_pt,
                                                                  reference_map,
                                                                  value_types_test)
                    int_val = -(val_dic_trial['div']
                                *val_dic_test['vals']
                                *val_dic_test['quad_wght'])
                    i_tmp = i + test_space[0].get_num_dof()
                    local_matrix_assembler.add_val(i_tmp,j,int_val)


        local_matrix_assembler.distribute_local_2_global(eN)
        local_matrix_assembler.reset_matrix()

    global_matrix_assembler.initialize_sparse_matrix()
    global_matrix_assembler.set_csr_rep()

    assert abs(global_rhs._rhs_vec[5] - 0.09858439) < 1.0E-8
    assert abs(global_rhs._rhs_vec[1] - 0.01525105) < 1.0E-8
    assert abs(global_rhs._rhs_vec[0] - 0.02641560) < 1.0E-8
    assert abs(global_rhs._rhs_vec[6] + 0.05691772) < 1.0E-8
    assert abs(global_rhs._rhs_vec[2] - 0.01578905) < 1.0E-8
    assert abs(global_rhs._rhs_vec[7] - 0.21991320) < 1.0E-8
    
    # ******************* Pin down the pressure to make system non-singular ****************
    global_matrix_assembler.set_row_as_dirichlet_bdy(dof_handler.get_num_dof() - 1)

    for basis_ele in trial_space:
        for boundary_dof in dof_handler.get_bdy_dof_dic(basis_ele.get_name()):
            dof_idx, bdy_type, bdy_type_global_idx = boundary_dof
            global_matrix_assembler.set_row_as_dirichlet_bdy(dof_idx)

            global_edge = mesh.get_edge(bdy_type_global_idx)
            local_edge_dof = dof_handler.get_local_edge_dof_idx(dof_idx,
                                                                bdy_type_global_idx)
            quad_pt = quadrature.find_one_quad_on_edge(global_edge, local_edge_dof)
            n = global_edge.get_unit_normal_vec()
            udotn = dirichlet_forcing_function.get_normal_velocity_func(n)
            val = udotn(quad_pt[0],quad_pt[1])
            global_rhs.set_value(dof_idx,val)

    global_matrix_assembler.solve(global_rhs, solution_vec)

    error_calculator = ec.DarcyErrorHandler(mesh,dof_handler,[basis[0]],solution_vec)

    l2_vel_error = error_calculator.calculate_vel_error(true_solution)
    
    assert abs(l2_vel_error - 0) < 1e-12

def test_mixed_assembly_5(basic_mesh_s1_bdm1_partial_r2):
    basis, mesh, dof_handler = basic_mesh_s1_bdm1_partial_r2

    num_local_dof = sum([a.get_num_dof() for a in basis])

    global_matrix_assembler = la.GlobalMatrix(dof_handler.get_num_dof(),
                                              dof_handler.get_num_dof())

    local_matrix_assembler = la.LocalMatrixAssembler(dof_handler,
                                                     num_local_dof,
                                                     global_matrix_assembler)

    solution_vec = la.DiscreteSolutionVector(dof_handler.get_num_dof())
    global_rhs = la.GlobalRHS(dof_handler)

    fx = lambda x,y : y
    fy = lambda x,y : 0.

    p_dx = lambda x,y : -y
    p_dy = lambda x,y : 0.

    true_vel = ft.TrueSolution([p_dx,p_dy])

    test_space = basis ; trial_space = basis
    quadrature = quad.Quadrature(3)
    reference_element = mt.ReferenceElement()

    num_mesh_elements = mesh.get_num_mesh_elements()
    for eN in range(num_mesh_elements):
        element = mesh.get_element(eN)
        reference_map = mp.ReferenceElementMap(element)
        piola_map = mp.PiolaMap(element)
        for quad_pt in quadrature.get_element_quad_pts():
            value_types_test = ['vals','Jt_vals','|Jt|','quad_wght']
            value_types_trial = ['vals','Jt_vals']
            for i in range(test_space[0].get_num_dof()):
                val_dic_test = test_space[0].get_element_vals(i,
                                                              quad_pt,
                                                              reference_map,
                                                              value_types_test)
                val_dic_test = piola_map.correct_div_space_vals(val_dic_test,
                                                                i,
                                                                test_space[0])
                # rhs construction
                int_val = 0.
                ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                             quad_pt)
                vals = true_vel.get_f_eval(ele_quad_pt)
                int_val = (la.Operators.dot_product(vals,
                                                    val_dic_test['Jt_vals'])
                           * (1./val_dic_test['|Jt|'])
                           * ele_quad_pt.get_quad_weight())
                global_rhs.add_val(eN, i, int_val)

                for j in range(trial_space[0].get_num_dof()):
                    int_val = 0.
                    # matrix construction
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    reference_map,
                                                                    value_types_trial)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     j,
                                                                     trial_space[0])
                    val_of_interest = (la.Operators.dot_product(val_dic_trial['Jt_vals'],
                                                               val_dic_test['Jt_vals'])*
                                       1./val_dic_test['|Jt|'])
                    int_val = (la.Operators.dot_product(val_dic_trial['Jt_vals'],
                                                        val_dic_test['Jt_vals']) *
                               (1./val_dic_test['|Jt|']) * val_dic_test['quad_wght'])
                    if (eN==3 and i==0):
                        a0 = val_dic_test['Jt_vals'][0]*(1./val_dic_test['|Jt|'])
                        a1 = val_dic_test['Jt_vals'][1]*(1./val_dic_test['|Jt|'])
                    local_matrix_assembler.add_val(i,j,int_val)

        local_matrix_assembler.distribute_local_2_global(eN)
        local_matrix_assembler.reset_matrix()

    global_matrix_assembler.initialize_sparse_matrix()
    global_matrix_assembler.set_csr_rep()

    global_mat_array = global_matrix_assembler.get_array_rep()

    assert abs(global_matrix_assembler.get_array_rep()[0][0]-0.166666667) < 1.0E-8
    assert abs(global_matrix_assembler.get_array_rep()[0][1]-0.069709692) < 1.0E-8
    assert abs(global_mat_array[0][8]+0.08333333) < 1.0E-8
    assert abs(global_matrix_assembler.get_array_rep()[1][0]-0.069709692) < 1.0E-8
    assert abs(global_matrix_assembler.get_array_rep()[1][1]-0.280502117) < 1.0E-8
    assert abs(global_mat_array[8][8]-0.166666667) < 1.0E-8
    assert abs(global_mat_array[8][0]+0.08333333) < 1.0E-8

    rhs_vec = global_rhs._rhs_vec
    assert abs(rhs_vec[0] + 0.02362447) < 1.0E-8
    assert abs(rhs_vec[1] + 0.11390386) < 1.0E-8

    global_matrix_assembler.solve(global_rhs, solution_vec)

    error_calculator = ec.DarcyErrorHandler(mesh,dof_handler,[basis[0]],solution_vec)

    l2_vel_error = error_calculator.calculate_vel_error(true_vel)
    assert abs(l2_vel_error) < 1.0e-12
