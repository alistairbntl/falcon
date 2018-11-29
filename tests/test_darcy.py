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

#@pytest.mark.hip
def test_darcy_converge_1(darcy_bdm1_p0_structured_converge_1):
    basis, mesh, dof_handler = darcy_bdm1_p0_structured_converge_1
    l2_error = darcy_convergence_script(basis, mesh, dof_handler)
    import pdb ; pdb.set_trace()
    
#@pytest.mark.hip
def test_darcy_converge_2(darcy_bdm1_p0_converge_2):
    basis, mesh, dof_handler = darcy_bdm1_p0_converge_2
    l2_error = darcy_convergence_script(basis, mesh, dof_handler)
    import pdb ; pdb.set_trace()
#    assert abs(l2_error - 0.268586032) < 1.E-08

#@pytest.mark.hip
def test_darcy_converge_3(darcy_bdm1_p0_converge_3):
    basis, mesh, dof_handler = darcy_bdm1_p0_converge_3
    l2_error = darcy_convergence_script(basis, mesh, dof_handler)
    import pdb ; pdb.set_trace()
#    assert abs(l2_error - 0.0964585802) < 1.E-08

#@pytest.mark.hip
def test_darcy_converge_4(darcy_bdm1_p0_converge_4):
    basis, mesh, dof_handler = darcy_bdm1_p0_converge_4
    l2_error = darcy_convergence_script(basis, mesh, dof_handler)
    import pdb ; pdb.set_trace()
#    assert abs(l2_error - 0.035090596) < 1.E-08

#@pytest.mark.hip
def test_darcy_converge_5(darcy_bdm1_p0_converge_5):
    # ARB - Need to investigate these convergence rates more carefully
    basis, mesh, dof_handler = darcy_bdm1_p0_converge_5
    l2_error = darcy_convergence_script(basis, mesh, dof_handler)
    import pdb ; pdb.set_trace()
#    assert abs(l2_error - 0.025631359) < 1.E-08

#@pytest.mark.hip
def test_darcy_bdm2_converge_2(darcy_bdm2_p1_converge_2):
    basis, mesh, dof_handler = darcy_bdm2_p1_converge_2
    l2_error = darcy_convergence_script(basis, mesh, dof_handler)
    assert abs(l2_error - 0.025631359) < 1.E-08    

#@pytest.mark.hip
def test_darcy_bdm2_converge_2(darcy_bdm2_p1_structured_converge_1):
    basis, mesh, dof_handler = darcy_bdm2_p1_structured_converge_1
    l2_error = darcy_convergence_script(basis, mesh, dof_handler)

#@pytest.mark.darcy_bdm2
def test_darcy_partial_bdm2_p1(darcy_bdm2_partial_one_element):
    basis, mesh, dof_handler = darcy_bdm2_partial_one_element
    error = darcy_convergence_script(basis,mesh,dof_handler)
    import pdb ; pdb.set_trace()

@pytest.mark.bdm3_now
def test_darcy_partial_bdm3_p2(darcy_bdm3_partial_one_element):
    basis, mesh, dof_handler = darcy_bdm3_partial_one_element
    error = darcy_convergence_script(basis,mesh,dof_handler)
    
def darcy_convergence_script(basis, mesh, dof_handler):

    num_local_dof = sum([a.get_num_dof() for a in basis])
    global_matrix_assembler = la.GlobalMatrix(dof_handler.get_num_dof(),
                                              dof_handler.get_num_dof())
    local_matrix_assembler = la.LocalMatrixAssembler(dof_handler,
                                                     num_local_dof,
                                                     global_matrix_assembler)
    num_pressure_dof = mesh.get_num_mesh_elements()*basis[1].get_num_interior_dof()

    solution_vec = la.DiscreteSolutionVector(dof_handler.get_num_dof())
    global_rhs = la.GlobalRHS(dof_handler)
    
    # Example 1
    # fx = lambda x,y : x*y - y**2
    # fy = lambda x,y : x + x**2 - 0.5*y**2
    # axi_div_u = lambda x,y : 0.0
    # p = lambda x,y : 2*x + 3*y - 3./2.
    # true_solution = ft.TrueSolution([fx,fy,p],
    #                                 [axi_div_u])
    # dirichlet_forcing_function = ft.Function((lambda x,y: x*y - y**2,
    #                                           lambda x,y: x + x**2 - 0.5*y**2))
    # forcing_function = ft.Function((lambda x,y: x*y - y**2 + 2,
    #                                 lambda x,y: x + x**2 - 0.5*y**2 + 3))    

    # Example 2
    cos = math.cos ; sin = math.sin ; pi = math.pi
    fx = lambda x,y : sin(pi*x)*cos(pi*y)
    fy = lambda x,y : -cos(pi*x)*sin(pi*y)
    axi_div_u = lambda x,y : 0.0
    p = lambda x,y : x*y
    true_solution = ft.TrueSolution([fx,fy,p],
                                    [axi_div_u])
    dirichlet_forcing_function = ft.Function((fx,
                                              fy))
    forcing_function = ft.Function((lambda x,y: fx(x,y) + y,
                                    lambda x,y: fy(x,y) + x))    
    
    test_space = basis ; trial_space = basis
    quadrature = quad.Quadrature(5)
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
#                    if i==0 and j==15:
#                        import pdb ; pdb.set_trace()
                    local_matrix_assembler.add_val(i_tmp,j,int_val)

        local_matrix_assembler.distribute_local_2_global(eN)
        local_matrix_assembler.reset_matrix()

    global_matrix_assembler.set_average_pressure_val(num_pressure_dof)

    global_matrix_assembler.initialize_sparse_matrix()
    global_matrix_assembler.set_csr_rep()

    A = global_matrix_assembler.get_array_rep()
    # ******************* Pin down the pressure to make system non-singular ****************
#    global_matrix_assembler.set_row_as_dirichlet_bdy(dof_handler.get_num_dof() - 1)
    for boundary_dof in dof_handler.get_bdy_dof_dic('bdm_basis'):
        dof_idx, bdy_type, bdy_type_global_idx = boundary_dof
        if bdy_type=='E':
            continue        
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
    A = global_matrix_assembler.get_array_rep()
    import numpy.linalg as npla
    q, r = npla.qr(A)
    
    error_calculator = ec.DarcyErrorHandler(mesh,dof_handler,[basis[0],basis[1]],solution_vec)
    l2_vel_error = error_calculator.calculate_l2_vel_error(true_solution)
    hdiv_vel_error = error_calculator.calculate_hdiv_vel_error(true_solution)
    l2_pressure_error = error_calculator.calculate_l2_pressure_error(true_solution)
    return l2_vel_error, hdiv_vel_error, l2_pressure_error

if __name__ =="__main__":
    h_lst = [1./4.,1./6,1./8,1./10,1./12]
#    h_lst = [1./4.]
    for h in h_lst:
        mesh = mt.StructuredMesh([1.,1.], h)
        basis = [bdm.BDMBasis(3), bdm.P2Basis_2D()]
        dof_handler = dof.DOFHandler(mesh,
                                     basis)
        l2_error = darcy_convergence_script(basis,
                                            mesh,
                                            dof_handler)
        print 'h : ' + `str(Fraction(h).limit_denominator())` + ' error : ' + `l2_error[0]` + '  ' + `l2_error[1]` + '  ' + `l2_error[2]`
