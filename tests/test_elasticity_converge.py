import math
import pytest
import numpy as np
import scipy.sparse.linalg as sp_la
from fractions import Fraction

from .context import falcon
from falcon import mesh_tools as mt
from falcon import mapping_tools as mp
from falcon import quadrature as quad
from falcon import bdm_basis as bdm
from falcon import dof_handler as dof
from falcon import linalg_tools as la
from falcon import function_tools as ft
from falcon import error_tools as ec

def test_elasticity_partial_bdm1_p0(elasticity_bdm1_partial):
    basis, mesh, dof_handler = elasticity_bdm1_partial
    error = elasticity_partial_converge_script(basis,mesh,dof_handler)
#    import pdb ; pdb.set_trace()

@pytest.mark.tnow
def test_elasticity_bdm2_p1(elasticity_bdm2):
    basis, mesh, dof_handler = elasticity_bdm2
    error = elasticity_convergence_script(basis,mesh,dof_handler)
#    import pdb ; pdb.set_trace()

@pytest.mark.cnow
def test_elasticity_bdm1_p0(elasticity_bdm1_p0_structured):
    basis, mesh, dof_handler = elasticity_bdm1_p0_structured
    error = elasticity_convergence_script(basis, mesh, dof_handler)
#    import pdb ; pdb.set_trace()

def elasticity_convergence_script(basis, mesh, dof_handler):
    num_local_dof = sum([a.get_num_dof() for a in basis])
    
    global_matrix_assembler = la.GlobalMatrix(dof_handler.get_num_dof(),
                                              dof_handler.get_num_dof())

    local_matrix_assembler = la.LocalMatrixAssembler(dof_handler,
                                                     num_local_dof,
                                                     global_matrix_assembler)

    solution_vec = la.DiscreteSolutionVector(dof_handler.get_num_dof())
    global_rhs = la.GlobalRHS(dof_handler)

    # Example 1
    # mu = 0.5 ; lam = 0.
    
    # u1 = lambda x,y : 4*x*(1-x)*y*(1-y)
    # u2 = lambda x,y : -4*x*(1-x)*y*(1-y)

    # sig11 = lambda x,y : 4*y*(1-x)*(1-y) - 4*x*y*(1-y)
    # sig12 = lambda x,y : 2*x*(1-x)*(1-2*y)-2*y*(1-y)*(1-2*x)
    # sig22 = lambda x,y : 4*x*y*(1-x)-4*x*(1-x)*(1-y)

    # fx = lambda x,y : 4*x**2 - 8*x*y + 8*y**2 - 4*y - 2
    # fy = lambda x,y : -8*x**2 + x*(8*y+4) - 4*y**2 + 2

    # Example 2
    mu = 0.5 ; lam = 1.
    fx = lambda x,y : 4*x**2 + x*(8-24*y) + 16*y**2 - 4*y -6
    fy = lambda x,y : -16*x**2 + 4*x*(6*y+1) - 4*y**2 - 8*y + 6
    
    sig11 = lambda x,y : 8*(1-2*x)*y*(1-y) - 4*x*(1-x)*(1-2*y)
    sig12 = lambda x,y : 2*x*(1-x)*(1-2*y) - 2*(1-2*x)*y*(1-y)
    sig22 = lambda x,y : 4*(1-2*x)*y*(1-y) - 8*x*(1-x)*(1-2*y)

    u1 = lambda x,y : 4*x*(1-x)*y*(1-y)
    u2 = lambda x,y : -4*x*(1-x)*y*(1-y)

    true_stress = ft.TrueSolution([sig11, sig12, sig12, sig22])
    true_displacement = ft.TrueSolution([u1, u2])
    true_divergence = ft.TrueSolution([fx,fy])

    test_space = basis ; trial_space = basis
    quadrature = quad.Quadrature(5)
    reference_element = mt.ReferenceElement()

    val_types_bdm = ['vals', 'Jt_vals', 'div', '|Jt|', 'quad_wght']
    val_types_p0_vec = ['vals', 'quad_wght', '|Jt|']
    val_types_p0_skewtens = ['vals', 'quad_wght', '|Jt|']

    num_mesh_elements = mesh.get_num_mesh_elements()
    for eN in range(num_mesh_elements):
        element = mesh.get_element(eN)
        piola_map = mp.PiolaMap(element)
        p = 0
        for quad_pt in quadrature.get_element_quad_pts():
            ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                         quad_pt)            
            for i in range(test_space[0].get_num_dof()):
                val_dic_test = test_space[0].get_element_vals(i,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_bdm)
                val_dic_test = piola_map.correct_div_space_vals(val_dic_test,
                                                                i,
                                                                test_space[0])

                for j in range(trial_space[0].get_num_dof()):
                    int_val = 0.
                    # matrix construction
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_bdm)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     j,
                                                                     trial_space[0])
                    int_val = (la.Operators.cartesian_elasticity_tens_tens(val_dic_trial['Jt_vals'],
                                                                           val_dic_test['Jt_vals'],
                                                                           mu=mu,
                                                                           lam=lam)
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
                               * val_dic_test['quad_wght'])
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
                           * val_dic_test['|Jt|'] * val_dic_test['quad_wght'])
                global_rhs.add_val(eN, i_tmp, int_val)

                for j in range(trial_space[0].get_num_dof()):
                    int_val = 0.
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_bdm)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     j,
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
                    int_val = 0.
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_bdm)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     j,
                                                                     trial_space[0])
                    int_val = (la.Operators.weak_symmetry_dot_product(val_dic_trial['Jt_vals'],
                                                                      val_dic_test['vals'])
                               * val_dic_test['quad_wght'])
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
    global_matrix_assembler.solve(global_rhs, solution_vec)
    A = global_matrix_assembler.get_array_rep()
    import numpy.linalg as np_la

    error_handler = ec.ElasticityErrorHandler(mesh,
                                              dof_handler,
                                              basis,
                                              solution_vec)
    err_stress = error_handler.calculate_stress_error(true_stress)
    err_div = error_handler.calculate_divergence_error(true_divergence)
    err_displ = error_handler.calculate_displacement_error(true_displacement)
    err_skew = error_handler.calculate_skew_symmetry_error()
    return err_stress, err_div, err_displ, err_skew
    
def elasticity_partial_converge_script(basis,
                                       mesh,
                                       dof_handler):
    num_local_dof = sum([a.get_num_dof() for a in basis])

    global_matrix_assembler = la.GlobalMatrix(dof_handler.get_num_dof(),
                                              dof_handler.get_num_dof())

    local_matrix_assembler = la.LocalMatrixAssembler(dof_handler,
                                                     num_local_dof,
                                                     global_matrix_assembler)

    solution_vec = la.DiscreteSolutionVector(dof_handler.get_num_dof())
    global_rhs = la.GlobalRHS(dof_handler)

    # Example 1
    # mu = 0.5 ; lam = 0.
    
    # u1 = lambda x,y : 4*x*(1-x)*y*(1-y)
    # u2 = lambda x,y : -4*x*(1-x)*y*(1-y)

    # sig11 = lambda x,y : 4*y*(1-x)*(1-y) - 4*x*y*(1-y)
    # sig12 = lambda x,y : 2*x*(1-x)*(1-2*y)-2*y*(1-y)*(1-2*x)
    # sig22 = lambda x,y : 4*x*y*(1-x)-4*x*(1-x)*(1-y)

    # fx = lambda x,y : 4*x**2 - 8*x*y + 8*y**2 - 4*y - 2
    # fy = lambda x,y : -8*x**2 + x*(8*y+4) - 4*y**2 + 2

    # Example 2
    mu = 0.5 ; lam = 1.
    fx = lambda x,y : 4*x*(1-x) + 16*y*(1-y) + 6*(1-2*x)*(1-2*y)
    fy = lambda x,y : -16*x*(1-x) - 4*y*(1-y) - 6*(1-2*x)*(1-2*y)

    sig11 = lambda x,y : 8*(1-2*x)*y*(1-y) - 4*x*(1-x)*(1-2*y)
    sig12 = lambda x,y : 2*x*(1-x)*(1-2*y) - 2*(1-2*x)*y*(1-y)
    sig22 = lambda x,y : 4*(1-2*x)*y*(1-y) - 8*x*(1-x)*(1-2*y)

    u1 = lambda x,y : 4*x*(1-x)*y*(1-y)
    u2 = lambda x,y : -4*x*(1-x)*y*(1-y)    

    true_stress = ft.TrueSolution([sig11, sig12, sig12, sig22])
    true_displacement = ft.TrueSolution([u1, u2])

    test_space = basis ; trial_space = basis
    quadrature = quad.Quadrature(5)
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
                val_dic_test = test_space[0].get_element_vals(i,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_bdm)
                val_dic_test = piola_map.correct_div_space_vals(val_dic_test,
                                                                i,
                                                                test_space[0])
                for j in range(trial_space[0].get_num_dof()):
                    int_val = 0.
                    # matrix construction
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_bdm)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     j,
                                                                     trial_space[0])
                    int_val = (la.Operators.cartesian_elasticity_tens_tens(val_dic_trial['Jt_vals'],
                                                                           val_dic_test['Jt_vals'],
                                                                           mu = mu,
                                                                           lam = lam)
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
                               * val_dic_test['quad_wght'])
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
                    int_val = 0.
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_bdm)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     j,
                                                                     trial_space[0])
                    int_val = (la.Operators.weak_symmetry_dot_product(val_dic_trial['Jt_vals'],
                                                                      val_dic_test['vals'])
                               * val_dic_test['quad_wght'])
                    i_tmp = i + test_space[0].get_num_dof()
                    local_matrix_assembler.add_val(i_tmp, j, int_val)
    
        local_matrix_assembler.distribute_local_2_global(eN)
        local_matrix_assembler.reset_matrix()

    global_matrix_assembler.initialize_sparse_matrix()
    global_matrix_assembler.set_csr_rep()

    global_mat_array = global_matrix_assembler.get_csr_rep().toarray()
    global_rhs_array = global_rhs._rhs_vec

    global_matrix_assembler.solve(global_rhs, solution_vec)
    A = global_matrix_assembler.get_array_rep()
    error_handler = ec.ElasticityErrorHandler(mesh,
                                              dof_handler,
                                              basis,
                                              solution_vec)
    err_stress = error_handler.calculate_stress_error(true_stress)
    err_skew = error_handler.calculate_skew_symmetry_error()
    return err_stress, err_skew

if __name__ == "__main__":
    h_lst = [1./4, 1./6, 1./8, 1./10, 1./12]
    for h in h_lst:
        mesh = mt.StructuredMesh([1.,1.], h)
        basis = [bdm.BDMTensBasis(3),
                 bdm.P2VecBasis_2D(),
                 bdm.P2SkewTensBasis_2D()]
        # basis = [bdm.BDMTensBasis(2),
        #          bdm.P1VecBasis_2D(),
        #          bdm.P1SkewTensBasis_2D()]                 
        # basis = [bdm.BDMTensBasis(1),
        #          bdm.P0VecBasis_2D(),
        #          bdm.P0SkewTensBasis_2D()]         
        dof_handler = dof.DOFHandler(mesh,
                                     basis)
        l2_error = elasticity_convergence_script(basis,
                                                 mesh,
                                                 dof_handler)
        print 'h : ' + `str(Fraction(h).limit_denominator())` + ' error : ' + `l2_error[0]` + '  ' + `l2_error[1]` + '  ' + `l2_error[2]` + ' ' + `l2_error[3]`
