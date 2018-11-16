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

def test_bdm_p0_elasticity_reference():
    reference_element = mt.ReferenceElement()
    basis = [bdm.BDMTensBasis(1),
             bdm.P0VecBasis_2D(),
             bdm.P0SkewTensBasis_2D()]
    piola_map = mp.PiolaMap(reference_element)
    quadrature = quad.Quadrature(2)
    val_types_bdm = ['vals', 'div','quad_wght']
    val_types_p0_vec = ['vals']
    val_types_p0_skewtens = ['vals']
    # A - block
    A_ref_block = np.zeros(shape=(basis[0].get_num_dof(),
                                  basis[0].get_num_dof()))
    for quad_pt in quadrature.get_element_quad_pts():
        for i in range(basis[0].get_num_dof()):
            val_dic_test = basis[0].get_element_vals(i,
                                                     quad_pt,
                                                     piola_map,
                                                     val_types_bdm)
            for j in range(basis[0].get_num_dof()):
                int_val = 0.
                val_dic_trial = basis[0].get_element_vals(j,
                                                          quad_pt,
                                                          piola_map,
                                                          val_types_bdm)
                int_val = (la.Operators.cartesian_elasticity_tens_tens(val_dic_trial['vals'],
                                                                       val_dic_test['vals'])
                           * val_dic_test['quad_wght'])

                A_ref_block[i][j] += int_val
    assert abs(A_ref_block[0][0] - 0.127791137) < 1.e-6 ; assert abs(A_ref_block[3][3] - 0.163875529) < 1.e-6
    assert abs(A_ref_block[0][3] - 0.005208333) < 1.e-6 ; assert abs(A_ref_block[3][0] - 0.005208333) < 1.e-6
    assert abs(A_ref_block[6][6] - 0.163875529) < 1.e-6 ; assert abs(A_ref_block[9][9] - 0.127791137) < 1.e-6
    assert abs(A_ref_block[6][9] - 0.005208333) < 1.e-6 ; assert abs(A_ref_block[6][9] - 0.005208333) < 1.e-6
    assert abs(A_ref_block[0][9] + 0.019437764) < 1.e-6 ; assert abs(A_ref_block[9][0] + 0.019437764) < 1.e-6    

    b_ref_block = np.zeros(shape=(basis[0].get_num_dof(),
                                  basis[1].get_num_dof()))
    for quad_pt in quadrature.get_element_quad_pts():
        for i in range(basis[0].get_num_dof()):
            val_dic_test = basis[0].get_element_vals(i,
                                                     quad_pt,
                                                     piola_map,
                                                     val_types_bdm)
            for j in range(basis[1].get_num_dof()):
                int_val = 0.
                val_dic_trial = basis[1].get_element_vals(j,
                                                          quad_pt,
                                                          piola_map,
                                                          val_types_p0_vec)
                int_val = (la.Operators.dot_product(val_dic_trial['vals'],
                                                    val_dic_test['div'])
                           * val_dic_test['quad_wght'])
                b_ref_block[i][j] += int_val
    assert abs(b_ref_block[0][0] - 0.70710678) < 1.e-6
    assert abs(b_ref_block[3][1] - 0.70710678) < 1.e-6

    c_ref_block = np.zeros(shape=(basis[0].get_num_dof(),
                                  basis[2].get_num_dof()))
    for quad_pt in quadrature.get_element_quad_pts():
        for i in range(basis[0].get_num_dof()):
            val_dic_test = basis[0].get_element_vals(i,
                                                     quad_pt,
                                                     piola_map,
                                                     val_types_bdm)
            for j in range(basis[2].get_num_dof()):
                int_val = 0.
                val_dic_trial = basis[2].get_element_vals(j,
                                                          quad_pt,
                                                          piola_map,
                                                          val_types_p0_skewtens)
                int_val = (la.Operators.weak_symmetry_dot_product(val_dic_trial['vals'],
                                                                  val_dic_test['vals'])
                           * val_dic_test['quad_wght'])
                c_ref_block[i][j] += int_val
    assert abs(c_ref_block[0] - 0.08627302) < 1.e-6
    assert abs(c_ref_block[3] - 0.32197528) < 1.e-6
    
def test_elasticity_2(basic_mesh_bdm1_p0_elasticity):
    basis, mesh, dof_handler = basic_mesh_bdm1_p0_elasticity

    test_space = basis ; trial_space = basis
    quadrature = quad.Quadrature(2)
    
    element = mesh.get_element(1)
    piola_map = mp.PiolaMap(element)
    val_types_bdm = ['vals', 'Jt_vals', 'div', '|Jt|', 'quad_wght']
    val_types_p0_vec = ['vals']
    val_types_p0_skewtens = ['vals']

    # A - block
    A_ref_block = np.zeros(shape=(basis[0].get_num_dof(),
                                  basis[0].get_num_dof()))
    # b - block
    b_ref_block = np.zeros(shape=(basis[0].get_num_dof(),
                                  basis[1].get_num_dof()))
    # c - block
    c_ref_block = np.zeros(shape=(basis[0].get_num_dof(),
                                  basis[2].get_num_dof()))
    
    for quad_pt in quadrature.get_element_quad_pts():
        ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                     quad_pt)
        for i in range(test_space[0].get_num_dof()):
            pi = i % 3
            ki = i / 6
            val_dic_test = basis[0].get_element_vals(i,
                                                     quad_pt,
                                                     piola_map,
                                                     val_types_bdm)
            val_dic_test = piola_map.correct_div_space_vals(val_dic_test,
                                                            3*ki + pi,
                                                            test_space[0])

            # A - block
            for j in range(trial_space[0].get_num_dof()):
                pj = j % 3
                kj = j / 6
                int_val = 0.
                val_dic_trial = basis[0].get_element_vals(j,
                                                          quad_pt,
                                                          piola_map,
                                                          val_types_bdm)
                val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                 3*kj + pj,
                                                                 test_space[0])
                int_val = (la.Operators.cartesian_elasticity_tens_tens(val_dic_trial['Jt_vals'],
                                                                       val_dic_test['Jt_vals'])
                           * (1./val_dic_test['|Jt|'])*val_dic_test['quad_wght'])
                A_ref_block[i][j] += int_val

            # b - block
            for j in range(trial_space[1].get_num_dof()):
                int_val = 0.
                val_dic_trial = basis[1].get_element_vals(j,
                                                          quad_pt,
                                                          piola_map,
                                                          val_types_p0_vec)
                int_val = (la.Operators.dot_product(val_dic_trial['vals'],
                                                    val_dic_test['div'])
                           * val_dic_test['quad_wght'])
                b_ref_block[i][j] += int_val

            for j in range(trial_space[2].get_num_dof()):
                int_val = 0.
                val_dic_trial = trial_space[2].get_element_vals(j,
                                                                quad_pt,
                                                                piola_map,
                                                                val_types_p0_skewtens)
                int_val = (la.Operators.weak_symmetry_dot_product(val_dic_trial['vals'],
                                                                  val_dic_test['Jt_vals'])
                           * 1./val_dic_test['|Jt|'] * val_dic_test['quad_wght'])
                c_ref_block[i][j] += int_val

    assert abs(A_ref_block[2][2] - 0.1638755292) < 1.e-6 ; assert abs(A_ref_block[5][5] - 0.1277911374) < 1.e-6
    assert abs(A_ref_block[8][8] - 0.1277911374) < 1.e-6 ; assert abs(A_ref_block[11][11] - 0.1638755292) < 1.e-6
    assert abs(A_ref_block[8][11] - 0.0052083333) < 1.e-6 ; assert abs(A_ref_block[11][8] - 0.0052083333) < 1.e-6
    assert abs(A_ref_block[2][5] - 0.0052083333) < 1.e-6 ; assert abs(A_ref_block[5][2] - 0.0052083333) < 1.e-6
    assert abs(A_ref_block[2][8] + 0.072916666) < 1.e-6 ; assert abs(A_ref_block[8][2] + 0.0729166666) < 1.e-6
    assert abs(A_ref_block[11][2] + 0.0013955687) < 1.e-6 ; assert abs(A_ref_block[2][11] + 0.0013955687) < 1.e-6
    assert abs(b_ref_block[0][0] - 0.5) < 1.e-6 ; assert abs(b_ref_block[0][1] - 0.) < 1.e-6
    assert abs(b_ref_block[2][0] + 0.70710678) < 1.e-6 ; assert abs(b_ref_block[5][1] + 0.70710678) < 1.e-6
    assert abs(c_ref_block[2] + 0.32197528) < 1.e-6 ; assert abs(c_ref_block[5] + 0.08627302) < 1.e-6
    assert abs(c_ref_block[8] - 0.08627302) < 1.e-6 ; assert abs(c_ref_block[11] - 0.32197528) < 1.e-6
    
def test_elasticity_2(basic_mesh_bdm1_p0_elasticity):
    basis, mesh, dof_handler = basic_mesh_bdm1_p0_elasticity

    num_local_dof = sum([a.get_num_dof() for a in basis])
    
    global_matrix_assembler = la.GlobalMatrix(dof_handler.get_num_dof(),
                                              dof_handler.get_num_dof())

    local_matrix_assembler = la.LocalMatrixAssembler(dof_handler,
                                                     num_local_dof,
                                                     global_matrix_assembler)

    solution_vec = la.DiscreteSolutionVector(dof_handler.get_num_dof())
    global_rhs = la.GlobalRHS(dof_handler)

    fx = lambda x,y : 4*x*(1-x) + 16*y*(1-y) + 6*(1-2*x)*(1-2*y)
    fy = lambda x,y : -16*x*(1-x) - 4*y*(1-y) - 6*(1-2*x)*(1-2*y)

    # fx = lambda x,y : x*y - y**2
    # fy = lambda x,y : x + x**2 - 0.5*y**2
    # p = lambda x,y : 2*x + 3*y - 3./2.
    # true_solution = ft.TrueSolution([fx,fy,p])
    # dirichlet_forcing_function = ft.Function((lambda x,y: x*y - y**2,
    #                                           lambda x,y: x + x**2 - 0.5*y**2))
    # forcing_function = ft.Function((lambda x,y: x*y - y**2 + 2,
    #                                 lambda x,y: x + x**2 - 0.5*y**2 + 3))

    test_space = basis ; trial_space = basis
    quadrature = quad.Quadrature(2)
    reference_element = mt.ReferenceElement()

    val_types_bdm = ['vals', 'Jt_vals', 'div', '|Jt|', 'quad_wght']
    val_types_p0_vec = ['vals', 'quad_wght', '|Jt|']
    val_types_p0_skewtens = ['vals', 'quad_wght', '|Jt|']    

    num_mesh_elements = mesh.get_num_mesh_elements()
    for eN in range(num_mesh_elements):
        element = mesh.get_element(eN)
        piola_map = mp.PiolaMap(element)
        
        for quad_pt in quadrature.get_element_quad_pts():
            ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                         quad_pt)            
            for i in range(test_space[0].get_num_dof()):
                pi = i % 3 ; ki = i / 6
                val_dic_test = test_space[0].get_element_vals(i,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_bdm)
                val_dic_test = piola_map.correct_div_space_vals(val_dic_test,
                                                                3*ki + pi,
                                                                test_space[0])

                for j in range(trial_space[0].get_num_dof()):
                    pj = j % 3 ; kj = j / 6
                    int_val = 0.
                    # matrix construction
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_bdm)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     3*kj + pj,
                                                                     trial_space[0])
                    int_val = (la.Operators.cartesian_elasticity_tens_tens(val_dic_trial['Jt_vals'],
                                                                           val_dic_test['Jt_vals'])
                               * (1./val_dic_test['|Jt|']) * val_dic_test['quad_wght'])
                    local_matrix_assembler.add_val(i,j,int_val)

                for j in range(trial_space[1].get_num_dof()):
                    int_val = 0.
                    val_dic_trial = basis[1].get_element_vals(j,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_vec)
                    int_val = (la.Operators.dot_product(val_dic_trial['vals'],
                                                        val_dic_test['div'])
                               * val_dic_test['quad_wght'])

                    j_tmp = j + trial_space[0].get_num_dof()
                    local_matrix_assembler.add_val(i,j_tmp,int_val)

                for j in range(trial_space[2].get_num_dof()):
                    int_val = 0.
                    val_dic_trial = basis[2].get_element_vals(j,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_skewtens)
                    int_val = (la.Operators.weak_symmetry_dot_product(val_dic_trial['vals'],
                                                                      val_dic_test['Jt_vals'])
                               * 1./val_dic_test['|Jt|']*val_dic_test['quad_wght'])
                    j_tmp = j + trial_space[0].get_num_dof() + trial_space[1].get_num_dof()
                    local_matrix_assembler.add_val(i,j_tmp,int_val)

            for i in range(test_space[1].get_num_dof()):
                i_tmp = i + test_space[0].get_num_dof()                
                val_dic_test = test_space[1].get_element_vals(i,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_vec)

                # compute and add RHS
                x = ele_quad_pt.vals[0] ; y = ele_quad_pt.vals[1]
                f_vec = np.array([fx(x,y),fy(x,y)])
                int_val = 0.

                int_val = (la.Operators.dot_product(f_vec,
                                                    val_dic_test['vals'])
                           * val_dic_test['quad_wght'])
                global_rhs.add_val(eN, i_tmp, int_val)

                for j in range(trial_space[0].get_num_dof()):
                    pj = j % 3 ; kj = j / 6
                    int_val = 0.
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_bdm)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     3*kj + pj,
                                                                     trial_space[0])
                    int_val = (la.Operators.dot_product(val_dic_trial['div'],
                                                        val_dic_test['vals'])
                               * val_dic_test['quad_wght'])
                    local_matrix_assembler.add_val(i_tmp, j, int_val)

            for i in range(test_space[2].get_num_dof()):
                val_dic_test = test_space[2].get_element_vals(i,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_skewtens)

                for j in range(trial_space[0].get_num_dof()):
                    pj = j % 3 ; kj = j / 6
                    int_val = 0.
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_bdm)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     3*kj + pj,
                                                                     trial_space[0])
                    int_val = (la.Operators.weak_symmetry_dot_product(val_dic_trial['Jt_vals'],
                                                                      val_dic_test['vals'])
                               * 1./val_dic_test['|Jt|'] * val_dic_test['quad_wght'])
                    i_tmp = i + test_space[0].get_num_dof() + test_space[1].get_num_dof()
                    local_matrix_assembler.add_val(i_tmp, j, int_val)

        local_matrix_assembler.distribute_local_2_global(eN)
        local_matrix_assembler.reset_matrix()

    global_matrix_assembler.initialize_sparse_matrix()
    global_matrix_assembler.set_csr_rep()

    global_mat_array = global_matrix_assembler.get_csr_rep().toarray()
    global_rhs_array = global_rhs._rhs_vec

    assert abs(global_mat_array[2][2] - 0.291666666) < 1.e-6 ; assert abs(global_mat_array[7][7] - 0.29166666) < 1.e-6
    assert abs(global_mat_array[12][12] - 0.29166666) < 1.e-6 ; assert abs(global_mat_array[17][17] - 0.29166666) < 1.e-6
    assert abs(global_mat_array[2][7] - 0.010416666) < 1.e-6 ; assert abs(global_mat_array[7][2] - 0.01041666) < 1.e-6
    assert abs(global_mat_array[2][17] + 0.02083333) < 1.e-6 ; assert abs(global_mat_array[17][2] + 0.020833333) < 1.e-6
    
    assert abs(global_mat_array[2][20] - 0.70710678) < 1.e-6 ; assert abs(global_mat_array[2][21] + 0.70710678) < 1.e-6
    assert abs(global_mat_array[7][22] - 0.70710678) < 1.e-6 ; assert abs(global_mat_array[7][23] + 0.70710678) < 1.e-6
    assert abs(global_mat_array[12][20] - 0.70710678) < 1.e-6 ; assert abs(global_mat_array[12][21] + 0.70710678) < 1.e-6
    assert abs(global_mat_array[17][22] - 0.70710678) < 1.e-6 ; assert abs(global_mat_array[17][23] + 0.70710678) < 1.e-6

    assert abs(global_mat_array[20][2] - 0.70710678) < 1.e-6 ; assert abs(global_mat_array[21][2] + 0.70710678) < 1.e-6
    assert abs(global_mat_array[22][7] - 0.70710678) < 1.e-6 ; assert abs(global_mat_array[23][7] + 0.70710678) < 1.e-6
    assert abs(global_mat_array[20][12] - 0.70710678) < 1.e-6 ; assert abs(global_mat_array[21][12] + 0.70710678) < 1.e-6
    assert abs(global_mat_array[22][17] - 0.70710678) < 1.e-6 ; assert abs(global_mat_array[23][17] + 0.70710678) < 1.e-6
    
    assert abs(global_mat_array[2][24] - 0.08627302) < 1.e-6 ; assert abs(global_mat_array[7][24] - 0.3219752754) < 1.e-6
    assert abs(global_mat_array[7][25] + 0.08627302) < 1.e-6 ; assert abs(global_mat_array[2][25] + 0.3219752754) < 1.e-6
    assert abs(global_mat_array[12][25] - 0.08627302) < 1.e-6 ; assert abs(global_mat_array[17][25] - 0.3219752754) < 1.e-6

    assert abs(global_mat_array[24][2] - 0.08627302) < 1.e-6 ; assert abs(global_mat_array[24][7] - 0.3219752754) < 1.e-6
    assert abs(global_mat_array[25][7] + 0.08627302) < 1.e-6 ; assert abs(global_mat_array[25][2] + 0.3219752754) < 1.e-6
    assert abs(global_mat_array[25][12] - 0.08627302) < 1.e-6 ; assert abs(global_mat_array[25][17] - 0.3219752754) < 1.e-6

    assert abs(global_rhs_array[20] - 1.66666666) < 1.e-6 ;     assert abs(global_rhs_array[21] - 1.66666666) < 1.e-6
    assert abs(global_rhs_array[22] + 1.66666666) < 1.e-6 ;     assert abs(global_rhs_array[23] + 1.66666666) < 1.e-6     
    
    # ******************* Pin down the pressure to make system non-singular ****************

    #In this example, we use a pure displacement boundary condition

    for boundary_dof in dof_handler.get_bdy_dof_dic(trial_space[1].get_name()):
        dof_idx, bdy_type, bdy_type_global_idx = boundary_dof
        if bdy_type == 'E':
            global_matrix_assembler.set_row_as_dirichlet_bdy(dof_idx)
            global_rhs.set_value(dof_idx,0.)
        else:
            pass

    global_matrix_assembler.solve(global_rhs, solution_vec)
    import pdb ; pdb.set_trace()

def test_elasticity_3(basic_mesh2_bdm1_p0_elasticity):
    basis, mesh, dof_handler = basic_mesh2_bdm1_p0_elasticity

    num_local_dof = sum([a.get_num_dof() for a in basis])
    
    global_matrix_assembler = la.GlobalMatrix(dof_handler.get_num_dof(),
                                              dof_handler.get_num_dof())

    local_matrix_assembler = la.LocalMatrixAssembler(dof_handler,
                                                     num_local_dof,
                                                     global_matrix_assembler)

    solution_vec = la.DiscreteSolutionVector(dof_handler.get_num_dof())
    global_rhs = la.GlobalRHS(dof_handler)

#    fx = lambda x,y : 4*x**2 + x*(8-24*y) + 16*y**2 - 4*y -6
#    fy = lambda x,y : -16*x**2 + 4*x*(6*y+1) - 4*y**2 - 8*y + 6
    fx = lambda x,y : 4*x*(1-x) + 16*y*(1-y) + 6*(1-2*x)*(1-2*y)
    fy = lambda x,y : -16*x*(1-x) - 4*y*(1-y) - 6*(1-2*x)*(1-2*y)

    sig11 = lambda x,y : 8*(1-2*x)*y*(1-y) - 4*x*(1-x)*(1-2*y)
    sig12 = lambda x,y : 2*x*(1-x)*(1-2*y) - 2*(1-2*x)*y*(1-y)
    sig22 = lambda x,y : 4*(1-2*x)*y*(1-y) - 8*x*(1-x)*(1-2*y)

    u1 = lambda x,y : 4*x*(1-x)*y*(1-y)
    u2 = lambda x,y : -4*x*(1-x)*y*(1-y)

    true_stress = ft.TrueSolution([sig11, sig12, sig12, sig22])
    true_displacement = ft.TrueSolution([u1, u2])
#    true_displacement = 

    # fx = lambda x,y : x*y - y**2
    # fy = lambda x,y : x + x**2 - 0.5*y**2
    # p = lambda x,y : 2*x + 3*y - 3./2.
    # true_solution = ft.TrueSolution([fx,fy,p])
    # dirichlet_forcing_function = ft.Function((lambda x,y: x*y - y**2,
    #                                           lambda x,y: x + x**2 - 0.5*y**2))
    # forcing_function = ft.Function((lambda x,y: x*y - y**2 + 2,
    #                                 lambda x,y: x + x**2 - 0.5*y**2 + 3))

    test_space = basis ; trial_space = basis
    quadrature = quad.Quadrature(2)
    reference_element = mt.ReferenceElement()

    val_types_bdm = ['vals', 'Jt_vals', 'div', '|Jt|', 'quad_wght']
    val_types_p0_vec = ['vals', 'quad_wght', '|Jt|']
    val_types_p0_skewtens = ['vals', 'quad_wght', '|Jt|']

    num_mesh_elements = mesh.get_num_mesh_elements()
    for eN in range(num_mesh_elements):
        element = mesh.get_element(eN)
        piola_map = mp.PiolaMap(element)
        
        for quad_pt in quadrature.get_element_quad_pts():
            ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                         quad_pt)            
            for i in range(test_space[0].get_num_dof()):
                pi = i % 3 ; ki = i / 6
                val_dic_test = test_space[0].get_element_vals(i,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_bdm)
                val_dic_test = piola_map.correct_div_space_vals(val_dic_test,
                                                                3*ki + pi,
                                                                test_space[0])

                for j in range(trial_space[0].get_num_dof()):
                    pj = j % 3 ; kj = j / 6
                    int_val = 0.
                    # matrix construction
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_bdm)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     3*kj + pj,
                                                                     trial_space[0])
                    int_val = (la.Operators.cartesian_elasticity_tens_tens(val_dic_trial['Jt_vals'],
                                                                           val_dic_test['Jt_vals'],
                                                                           mu=0.5)
                               * (1./val_dic_test['|Jt|']) * val_dic_test['quad_wght'])
                    local_matrix_assembler.add_val(i,j,int_val)

                for j in range(trial_space[1].get_num_dof()):
                    int_val = 0.
                    val_dic_trial = basis[1].get_element_vals(j,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_vec)
                    int_val = (la.Operators.dot_product(val_dic_trial['vals'],
                                                        val_dic_test['div'])
                               * val_dic_test['quad_wght'])

                    j_tmp = j + trial_space[0].get_num_dof()
                    local_matrix_assembler.add_val(i,j_tmp,int_val)

                for j in range(trial_space[2].get_num_dof()):
                    int_val = 0.
                    val_dic_trial = basis[2].get_element_vals(j,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_skewtens)
                    int_val = (la.Operators.weak_symmetry_dot_product(val_dic_trial['vals'],
                                                                      val_dic_test['Jt_vals'])
                               * 1./val_dic_test['|Jt|']*val_dic_test['quad_wght'])
                    j_tmp = j + trial_space[0].get_num_dof() + trial_space[1].get_num_dof()
                    local_matrix_assembler.add_val(i,j_tmp,int_val)

            for i in range(test_space[1].get_num_dof()):
                i_tmp = i + test_space[0].get_num_dof()                
                val_dic_test = test_space[1].get_element_vals(i,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_vec)

                # compute and add RHS
                x = ele_quad_pt.vals[0] ; y = ele_quad_pt.vals[1]
                f_vec = np.array([fx(x,y),fy(x,y)])
                int_val = 0.

                int_val = (la.Operators.dot_product(f_vec,
                                                    val_dic_test['vals'])
                           * val_dic_test['quad_wght'])
                global_rhs.add_val(eN, i_tmp, int_val)

                for j in range(trial_space[0].get_num_dof()):
                    pj = j % 3 ; kj = j / 6
                    int_val = 0.
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_bdm)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     3*kj + pj,
                                                                     trial_space[0])
                    int_val = (la.Operators.dot_product(val_dic_trial['div'],
                                                        val_dic_test['vals'])
                               * val_dic_test['quad_wght'])
                    local_matrix_assembler.add_val(i_tmp, j, int_val)

            for i in range(test_space[2].get_num_dof()):
                val_dic_test = test_space[2].get_element_vals(i,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_skewtens)

                for j in range(trial_space[0].get_num_dof()):
                    pj = j % 3 ; kj = j / 6
                    int_val = 0.
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_bdm)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     3*kj + pj,
                                                                     trial_space[0])
                    int_val = (la.Operators.weak_symmetry_dot_product(val_dic_trial['Jt_vals'],
                                                                      val_dic_test['vals'])
                               * 1./val_dic_test['|Jt|'] * val_dic_test['quad_wght'])
                    i_tmp = i + test_space[0].get_num_dof() + test_space[1].get_num_dof()
                    local_matrix_assembler.add_val(i_tmp, j, int_val)

        local_matrix_assembler.distribute_local_2_global(eN)
        local_matrix_assembler.reset_matrix()

    global_matrix_assembler.initialize_sparse_matrix()
    global_matrix_assembler.set_csr_rep()

    global_mat_array = global_matrix_assembler.get_csr_rep().toarray()
    global_rhs_array = global_rhs._rhs_vec
    
    # ******************* Pin down the pressure to make system non-singular ****************

    # In this example, we use a pure displacement boundary condition
    
    for boundary_dof in dof_handler.get_bdy_dof_dic(trial_space[1].get_name()):
        dof_idx, bdy_type, bdy_type_global_idx = boundary_dof
        if bdy_type == 'E':
            global_matrix_assembler.set_row_as_dirichlet_bdy(dof_idx)
            global_rhs.set_value(dof_idx,0.)
        else:
            pass

    global_matrix_assembler.solve(global_rhs, solution_vec)

    error_handler = ec.ElasticityErrorHandler(mesh,
                                              dof_handler,
                                              basis,
                                              solution_vec)
    err_stress = error_handler.calculate_stress_error(true_stress)
    err_displ = error_handler.calculate_displacement_error(true_displacement)
    ### FYI - elements 7 and 12 are the interior elements.  I'm still not sure
    ### what the debugging strategy should be moving forward, but this is
    ### something to look into.  Also, the dof_handler class has a new
    ### method get_global_basis_dof_rng that returns the dof ranges for each
    ### basis in the FE space.

@pytest.mark.skip
def test_elasticity_4(basic_mesh2_bdm1_partial_elasticity):
    basis, mesh, dof_handler = basic_mesh2_bdm1_partial_elasticity

    num_local_dof = sum([a.get_num_dof() for a in basis])

    global_matrix_assembler = la.GlobalMatrix(dof_handler.get_num_dof(),
                                              dof_handler.get_num_dof())

    local_matrix_assembler = la.LocalMatrixAssembler(dof_handler,
                                                     num_local_dof,
                                                     global_matrix_assembler)

    solution_vec = la.DiscreteSolutionVector(dof_handler.get_num_dof())
    global_rhs = la.GlobalRHS(dof_handler)

    sig11 = lambda x,y : 4*y*(1-x)*(1-y) - 4*x*y*(1-y)
    sig12 = lambda x,y : 2*x*(1-x)*(1-2*y)-2*y*(1-y)*(1-2*x)
    sig22 = lambda x,y : 4*x*y*(1-x)-4*x*(1-x)*(1-y)

    u1 = lambda x,y : 4*x*(1-x)*y*(1-y)
    u2 = lambda x,y : -4*x*(1-x)*y*(1-y)

    true_stress = ft.TrueSolution([sig11, sig12, sig12, sig22])
    true_displacement = ft.TrueSolution([u1, u2])

    test_space = basis ; trial_space = basis
    quadrature = quad.Quadrature(2)
    reference_element = mt.ReferenceElement()

    val_types_bdm = ['vals', 'Jt_vals', 'div', '|Jt|', 'quad_wght']
    val_types_p0_skewtens = ['vals', 'quad_wght', '|Jt|']
    
    num_mesh_elements = mesh.get_num_mesh_elements()
    for eN in range(num_mesh_elements):
        element = mesh.get_element(eN)
        piola_map = mp.PiolaMap(element)
        
        for quad_pt in quadrature.get_element_quad_pts():
            ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                         quad_pt)            
            for i in range(test_space[0].get_num_dof()):
                pi = i % 3 ; ki = i / 6
                val_dic_test = test_space[0].get_element_vals(i,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_bdm)
                val_dic_test = piola_map.correct_div_space_vals(val_dic_test,
                                                                3*ki + pi,
                                                                test_space[0])

                for j in range(trial_space[0].get_num_dof()):
                    pj = j % 3 ; kj = j / 6
                    int_val = 0.
                    # matrix construction
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_bdm)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     3*kj + pj,
                                                                     trial_space[0])
                    int_val = (la.Operators.cartesian_elasticity_tens_tens(val_dic_trial['Jt_vals'],
                                                                           val_dic_test['Jt_vals'],
                                                                           mu=0.5,
                                                                           lam=0.0)
                               * (1./val_dic_test['|Jt|']) * val_dic_test['quad_wght'])
                    local_matrix_assembler.add_val(i,j,int_val)

                for j in range(trial_space[1].get_num_dof()):
                    int_val = 0.
                    val_dic_trial = basis[1].get_element_vals(j,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_skewtens)
                    int_val = (la.Operators.weak_symmetry_dot_product(val_dic_trial['vals'],
                                                                      val_dic_test['Jt_vals'])
                               * 1./val_dic_test['|Jt|']*val_dic_test['quad_wght'])
                    j_tmp = j + trial_space[0].get_num_dof()
                    local_matrix_assembler.add_val(i,j_tmp,int_val)

                # Compute the reduced RHS
                x = ele_quad_pt.vals[0] ; y = ele_quad_pt.vals[1]
                f_vec = np.array([u1(x,y), u2(x,y)])
                int_val = 0.
                int_val = -(la.Operators.dot_product(f_vec,
                                                     val_dic_test['div'])
                              * val_dic_test['quad_wght'])
                global_rhs.add_val(eN, i, int_val)

            for i in range(test_space[1].get_num_dof()):
                val_dic_test = test_space[1].get_element_vals(i,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_skewtens)

                for j in range(trial_space[0].get_num_dof()):
                    pj = j % 3 ; kj = j / 6
                    int_val = 0.
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_bdm)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     3*kj + pj,
                                                                     trial_space[0])
                    int_val = (la.Operators.weak_symmetry_dot_product(val_dic_trial['Jt_vals'],
                                                                      val_dic_test['vals'])
                               * 1./val_dic_test['|Jt|'] * val_dic_test['quad_wght'])
                    i_tmp = i + test_space[0].get_num_dof()
                    local_matrix_assembler.add_val(i_tmp, j, int_val)
    
        local_matrix_assembler.distribute_local_2_global(eN)
        local_matrix_assembler.reset_matrix()

    global_matrix_assembler.initialize_sparse_matrix()
    global_matrix_assembler.set_csr_rep()

    global_mat_array = global_matrix_assembler.get_csr_rep().toarray()
    global_rhs_array = global_rhs._rhs_vec

    global_matrix_assembler.solve(global_rhs, solution_vec)

    error_handler = ec.ElasticityErrorHandler(mesh,
                                              dof_handler,
                                              basis,
                                              solution_vec)
    err_stress = error_handler.calculate_stress_error(true_stress)

@pytest.mark.skip
def test_elasticity_5(basic_mesh2_bdm1_partial_elasticity):
    basis, mesh, dof_handler = basic_mesh2_bdm1_partial_elasticity

    num_local_dof = sum([a.get_num_dof() for a in basis])

    global_matrix_assembler = la.GlobalMatrix(dof_handler.get_num_dof(),
                                              dof_handler.get_num_dof())

    local_matrix_assembler = la.LocalMatrixAssembler(dof_handler,
                                                     num_local_dof,
                                                     global_matrix_assembler)

    solution_vec = la.DiscreteSolutionVector(dof_handler.get_num_dof())
    global_rhs = la.GlobalRHS(dof_handler)

    sig11 = lambda x,y : 8*(1-2*x)*y*(1-y) - 4*(x-x**2)*(1-2*y)
    sig12 = lambda x,y : 2*x*(1-x)*(1-2*y) - 2*(1-2*x)*y*(1-y)
    sig22 = lambda x,y : 4*(1-2*x)*y*(1-y) - 8*x*(1-x)*(1-2*y)

    u1 = lambda x,y : 4*x*(1-x)*y*(1-y)
    u2 = lambda x,y : -4*x*(1-x)*y*(1-y)

    true_stress = ft.TrueSolution([sig11, sig12, sig12, sig22])
    true_displacement = ft.TrueSolution([u1, u2])

    test_space = basis ; trial_space = basis
    quadrature = quad.Quadrature(2)
    reference_element = mt.ReferenceElement()

    val_types_bdm = ['vals', 'Jt_vals', 'div', '|Jt|', 'quad_wght']
    val_types_p0_skewtens = ['vals', 'quad_wght', '|Jt|']
    
    num_mesh_elements = mesh.get_num_mesh_elements()
    for eN in range(num_mesh_elements):
        element = mesh.get_element(eN)
        piola_map = mp.PiolaMap(element)
        
        for quad_pt in quadrature.get_element_quad_pts():
            ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                         quad_pt)            
            for i in range(test_space[0].get_num_dof()):
                pi = i % 3 ; ki = i / 6
                val_dic_test = test_space[0].get_element_vals(i,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_bdm)
                val_dic_test = piola_map.correct_div_space_vals(val_dic_test,
                                                                3*ki + pi,
                                                                test_space[0])

                for j in range(trial_space[0].get_num_dof()):
                    pj = j % 3 ; kj = j / 6
                    int_val = 0.
                    # matrix construction
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_bdm)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     3*kj + pj,
                                                                     trial_space[0])
                    int_val = (la.Operators.cartesian_elasticity_tens_tens(val_dic_trial['Jt_vals'],
                                                                           val_dic_test['Jt_vals'],
                                                                           mu=0.5)
                                                                           * (1./val_dic_test['|Jt|']) * val_dic_test['quad_wght'])
                    local_matrix_assembler.add_val(i,j,int_val)

                for j in range(trial_space[1].get_num_dof()):
                    int_val = 0.
                    val_dic_trial = basis[1].get_element_vals(j,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_skewtens)
                    int_val = (la.Operators.weak_symmetry_dot_product(val_dic_trial['vals'],
                                                                      val_dic_test['Jt_vals'])
                               * 1./val_dic_test['|Jt|']*val_dic_test['quad_wght'])
                    j_tmp = j + trial_space[0].get_num_dof()
                    local_matrix_assembler.add_val(i,j_tmp,int_val)

                # Compute the reduced RHS
                x = ele_quad_pt.vals[0] ; y = ele_quad_pt.vals[1]
                f_vec = np.array([u1(x,y), u2(x,y)])
                int_val = 0.
                int_val = -(la.Operators.dot_product(f_vec,
                                                     val_dic_test['div'])
                              * val_dic_test['quad_wght'])
                global_rhs.add_val(eN, i, int_val)

            for i in range(test_space[1].get_num_dof()):
                val_dic_test = test_space[1].get_element_vals(i,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_skewtens)

                for j in range(trial_space[0].get_num_dof()):
                    pj = j % 3 ; kj = j / 6
                    int_val = 0.
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_bdm)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     3*kj + pj,
                                                                     trial_space[0])
                    int_val = (la.Operators.weak_symmetry_dot_product(val_dic_trial['Jt_vals'],
                                                                      val_dic_test['vals'])
                               * 1./val_dic_test['|Jt|'] * val_dic_test['quad_wght'])
                    i_tmp = i + test_space[0].get_num_dof()
                    local_matrix_assembler.add_val(i_tmp, j, int_val)
    
        local_matrix_assembler.distribute_local_2_global(eN)
        local_matrix_assembler.reset_matrix()

    global_matrix_assembler.initialize_sparse_matrix()
    global_matrix_assembler.set_csr_rep()

    global_mat_array = global_matrix_assembler.get_csr_rep().toarray()
    global_rhs_array = global_rhs._rhs_vec

    global_matrix_assembler.solve(global_rhs, solution_vec)

    error_handler = ec.ElasticityErrorHandler(mesh,
                                              dof_handler,
                                              basis,
                                              solution_vec)
    err_stress = error_handler.calculate_stress_error(true_stress)

