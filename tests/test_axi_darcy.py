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

@pytest.mark.hip
def test_axidarcy_converge_1(darcy_bdm1_p0_structured_converge_1):
    basis, mesh, dof_handler = darcy_bdm1_p0_structured_converge_1
    l2_error = axidarcy_convergence_script(basis, mesh, dof_handler)
    import pdb ; pdb.set_trace()

@pytest.mark.hip
def test_axidarcy_converge_1(darcy_bdm1_p0_structured_converge_1):
    basis, mesh, dof_handler = darcy_bdm1_p0_structured_converge_1
    l2_error = axidarcy_convergence_script(basis, mesh, dof_handler)
    import pdb ; pdb.set_trace()
    
def axidarcy_convergence_script(basis, mesh, dof_handler):
    num_local_dof = sum([a.get_num_dof() for a in basis])

    global_matrix_assembler = la.GlobalMatrix(dof_handler.get_num_dof(),
                                              dof_handler.get_num_dof())

    local_matrix_assembler = la.LocalMatrixAssembler(dof_handler,
                                                     num_local_dof,
                                                     global_matrix_assembler)
#    num_pressure_dof = mesh.get_num_mesh_elements()*basis[1].get_num_interior_dof()    

    solution_vec = la.DiscreteSolutionVector(dof_handler.get_num_dof())
    global_rhs = la.GlobalRHS(dof_handler)

    # Example 1
    # fx = lambda x,y : 0.
    # fy = lambda x,y : x**2
    # axi_div_f = lambda x,y : 0.
    # p = lambda x,y : 0.
    # true_solution = ft.TrueSolution([fx,fy,p],
    #                                 [axi_div_f])
    # dirichlet_forcing_function = ft.Function((lambda x,y: 0.,
    #                                           lambda x,y: x**2))
    # forcing_function = ft.Function((lambda x,y: 0.,
    #                                 lambda x,y: x**2))

    #Example 2
    cos = math.cos ; sin = math.sin ; pi = math.pi
    
    ur = lambda r,z : -r*cos(pi*r)*sin(pi*z)
    uz = lambda r,z : -(2. / pi)*cos(pi*r)*cos(pi*z) + r*sin(pi*r)*cos(pi*z)
    axi_div_u = lambda r, z: 0.
    
    p = lambda r,z : sin(pi*z)*(-cos(pi*r)+2*pi*r*sin(pi*r))
    pr = lambda r,z : pi*sin(pi*z)*(3*sin(pi*r)+2*pi*r*cos(pi*r))
    pz = lambda r,z : pi*cos(pi*z)*(2*pi*r*sin(pi*r) - cos(pi*r))

    ff1 = lambda r,z : ur(r,z) + pr(r,z)
    ff2 = lambda r,z : uz(r,z) + pz(r,z)

    true_solution = ft.TrueSolution([ur,uz,p], [axi_div_u])
    dirichlet_forcing_function = ft.Function((lambda r,z: ur(r,z),
                                              lambda r,z: uz(r,z)))
    forcing_function = ft.Function((lambda r,z: ff1(r,z),
                                    lambda r,z: ff2(r,z)))

    
    test_space = basis ; trial_space = basis
#    bdm_basis_quad_pt = quad.Quadrature(basis[0].get_degree()+1)

    quadrature = quad.Quadrature(5)
    bdm_basis_quad_pt = quad.Quadrature(5)
    reference_element = mt.ReferenceElement()

    num_mesh_elements = mesh.get_num_mesh_elements()
    for eN in range(num_mesh_elements):
        element = mesh.get_element(eN)
        reference_map = mp.ReferenceElementMap(element)
        piola_map = mp.PiolaMap(element)
        for quad_pt in quadrature.get_element_quad_pts():
            ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                         quad_pt)
            value_types_test = ['vals','Jt_vals','|Jt|','quad_wght','div']
            value_types_trial = ['vals', 'Jt_vals','div']
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
                vals = forcing_function.get_f_eval(ele_quad_pt)
                int_val = (la.Operators.dot_product(vals,
                                                    val_dic_test['Jt_vals'])
                           * ele_quad_pt.vals[0]   # r - part of axisymmetric IP
                           * quad_pt.get_quad_weight() )

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
                    int_val = (la.Operators.dot_product(val_dic_trial['Jt_vals'] ,
                                                        val_dic_test['Jt_vals']) *
                               (1./val_dic_test['|Jt|']) * quad_pt.get_quad_weight() ) * ele_quad_pt.vals[0] 
                    # This addition is the r scaling for the axisymmetric IP

                    # grad-div stabilization
                    r_val = ele_quad_pt.vals[0]
                    if r_val !=0:
                        p1_val = 0. ; q1_val = 0.
                        p1_val = 1./r_val * val_dic_trial['Jt_vals'][0] + val_dic_trial['div']
                        q1_val = 1./r_val * val_dic_test['Jt_vals'][0] + val_dic_test['div']
                        int_val += 10.*(p1_val * q1_val * r_val * (1./val_dic_test['|Jt|'])) * quad_pt.get_quad_weight()
                    # end grad-div stabilization

                    local_matrix_assembler.add_val(i,j,int_val)

        value_types_test = ['vals', 'Jt_vals', '|Jt|', 'quad_wght', 'div']
        value_types_trial = ['vals']
        for i in range(test_space[0].get_num_dof()):
            for j in range(trial_space[1].get_num_dof()):
                int_val = 0.
                for quad_pt in quadrature.get_element_quad_pts():                    
                    ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                                 quad_pt)

                    val_dic_test = test_space[0].get_element_vals(i,
                                                                  quad_pt,
                                                                  reference_map,
                                                                  value_types_test)
                    val_dic_test = piola_map.correct_div_space_vals(val_dic_test,
                                                                    i,
                                                                    test_space[0])
                    val_dic_trial = test_space[1].get_element_vals(j,
                                                                   quad_pt,
                                                                   reference_map,
                                                                   value_types_trial)
                    int_val = - val_dic_trial['vals'] * (val_dic_test['div'] * ele_quad_pt.vals[0]
                                                         + val_dic_test['Jt_vals'][0] ) * (1./val_dic_test['|Jt|']) * ele_quad_pt.get_quad_weight()

                    j_tmp = j + trial_space[0].get_num_dof()
                    local_matrix_assembler.add_val(i,j_tmp,int_val)

        value_types_test = ['vals', 'quad_wght']
        value_types_trial = ['vals', 'div', 'Jt_vals', '|Jt|']
        for i in range(test_space[1].get_num_dof()):
            for j in range(trial_space[0].get_num_dof()):
                int_val = 0.
                for quad_pt in quadrature.get_element_quad_pts():
                    ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                                 quad_pt)

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
                    int_val =  val_dic_test['vals'] * (val_dic_trial['div'] * ele_quad_pt.vals[0]
                                                       + val_dic_trial['Jt_vals'][0]) *(1./val_dic_trial['|Jt|']) * quad_pt.get_quad_weight()
                    i_tmp = i + test_space[0].get_num_dof()
                    local_matrix_assembler.add_val(i_tmp,j,int_val)

        local_matrix_assembler.distribute_local_2_global(eN)
        local_matrix_assembler.reset_matrix()

#    global_matrix_assembler.set_average_pressure_val(num_pressure_dof)        

    global_matrix_assembler.initialize_sparse_matrix()
    global_matrix_assembler.set_csr_rep()

    global_matrix_assembler.set_row_as_dirichlet_bdy(dof_handler.get_num_dof() - 1)
#    global_rhs.set_value(dof_handler.get_num_dof()-1, 1.0)

    for basis_ele in trial_space:
        for boundary_dof in dof_handler.get_bdy_dof_dic('bdm_basis'):
            dof_idx, bdy_type, bdy_type_global_idx = boundary_dof
            if bdy_type == 'E':
                continue
            global_matrix_assembler.set_row_as_dirichlet_bdy(dof_idx)

            global_edge = mesh.get_edge(bdy_type_global_idx)
            local_edge_dof = dof_handler.get_local_edge_dof_idx(dof_idx,
                                                                bdy_type_global_idx)            
            quad_pt = bdm_basis_quad_pt.find_one_quad_on_edge(global_edge, local_edge_dof)
            n = global_edge.get_unit_normal_vec()
            udotn = true_solution.get_normal_velocity_func(n)
            val = udotn(quad_pt[0],quad_pt[1])
            global_rhs.set_value(dof_idx,val)

    global_matrix_assembler.solve(global_rhs, solution_vec)
    A = global_matrix_assembler.get_array_rep()    

    error_calculator = ec.DarcyErrorHandler(mesh,dof_handler,[basis[0],basis[1]],solution_vec)
    l2_vel_error = error_calculator.calculate_l2_vel_error(true_solution,r=1)
    hdiv_vel_error = error_calculator.calculate_hdiv_vel_error(true_solution,r=1)
    l2_pressure_error = error_calculator.calculate_l2_pressure_error(true_solution,r=1)
    return l2_vel_error, hdiv_vel_error, l2_pressure_error
    
if __name__ == "__main__":
    h_lst = [1./4, 1./6, 1./8, 1./10, 1./12]
    for h in h_lst:
        mesh = mt.StructuredMesh([1.,1.], h)
        basis = [bdm.BDMBasis(3), bdm.P2Basis_2D()]
        dof_handler = dof.DOFHandler(mesh,
                                     basis)
        l2_error = axidarcy_convergence_script(basis,
                                               mesh,
                                               dof_handler)
        print 'h : ' + `str(Fraction(h).limit_denominator())` + ' error : ' + `l2_error[0]` + '  ' + `l2_error[1]` + '  ' + `l2_error[2]`
        
