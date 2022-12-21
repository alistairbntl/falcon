from math import cos, sin, pi
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

@pytest.mark.axitest
def test_axi_elasticity_bdm1_p0(elasticity_bdm1_p0_structured):
    basis, mesh, dof_handler = elasticity_bdm1_p0_structured
    error = axi_elasticity_convergence_script(basis,mesh,dof_handler)

def axi_elasticity_convergence_script(basis, mesh, dof_handler):
    num_local_dof = sum([a.get_num_dof() for a in basis])

    global_matrix_assembler = la.GlobalMatrix(dof_handler.get_num_dof(),
                                              dof_handler.get_num_dof())

    local_matrix_assembler = la.LocalMatrixAssembler(dof_handler,
                                                     num_local_dof,
                                                     global_matrix_assembler)

    solution_vec = la.DiscreteSolutionVector(dof_handler.get_num_dof())
    global_rhs = la.GlobalRHS(dof_handler)

    # Example 1
#     mu = 0.5 ; lam = 0.
    
#     u1 = lambda x,y : 4*x**3*(1-x)*y*(1-y)
#     u2 = lambda x,y : -4*x**3*(1-x)*y*(1-y)

#     sig11 = lambda x,y : 4*x**2*(4*x-3)*(y-1)*y
# #    sig12 = lambda x,y : 2*x**2*(2*x**2*y -x**2 - 4*x*y**2 + 2*x*y + x + 3*y**2 - 3*y)
#     sig12 = lambda x,y : 2*x**2*(x*(x-1)*(2*y-1)-(4*x-3)*(y-1)*y)
#     sig22 = lambda x,y : -4*x**3*(x-1)*(2*y-1)

#     sig_scalar = lambda x,y : 4*x**2*(1-x)*y*(1-y)

# #    fx = lambda x,y : 4*x*(4*x-3)*(y-1)*y + 2*x*(2*x**3 - 2*x**2*(4*y-1) + 3*x*(8*y**2-6*y-1) - 12*(y-1)*y)
#     fx = lambda x,y : 4*x*(4*x-3)*(y-1)*y + 8*x*(6*x-3)*(y-1)*y + 2*x**2*(2*x**2 + x*(2-8*y)+6*y-3) - 4*x*(1-x)*y*(1-y)
# #    fy = lambda x,y : 2*x**2*(x*(x-1)*(2*y-1)-(4*x-3)*(y-1)*y) - 2*x*(4*x**3 - 8*x**2*y + 3*x*(4*y**2 - 2*y - 1) - 6*(y-1)*y )
#     fy = lambda x,y : 2*x*(x*(x-1)*(2*y-1)-(4*x-3)*(y-1)*y) + 2*x*(x**2*(8*y-4)+x*(-12*y**2+6*y+3) +6*(y-1)*y) - 8*(x-1)*x**3

#     divx = lambda x,y : 4*x*(4*x-3)*(y-1)*y + 8*x*(6*x-3)*(y-1)*y + 2*x**2*(2*x**2 + x*(2-8*y)+6*y-3)
#     divy = lambda x,y : 2*x*(x*(x-1)*(2*y-1)-(4*x-3)*(y-1)*y) + 2*x*(x**2*(8*y-4)+x*(-12*y**2+6*y+3) +6*(y-1)*y) - 8*(x-1)*x**3    
    
    # Example 2
    # mu = 0.5 ; lam = 1.

    # u1 = lambda x,y : 4*x**3*(1-x)*y*(1-y)
    # u2 = lambda x,y : -4*x**3*(1-x)*y*(1-y)

    # sig11 = lambda x,y : 6*x**2*(4*x-3)*(y-1)*y + 2*x**3*(x-1)*(1-2*y)
    # sig12 = lambda x,y : 2*x**2*(x*(x-1)*(2*y-1) - (4*x-3)*(y-1)*y)
    # sig22 = lambda x,y : 6*x**3*(x-1)*(1-2*y) + 2*x**2*(4*x-3)*(y-1)*y

    # fx = lambda x,y : 6*x*(4*x-3)*(y-1)*y + 2*x**2*(x-1)*(1-2*y) + 6*(12*x**2-6*x)*(y-1)*y +2*(4*x**3-3*x**2)*(1-2*y) + 2*x**2*(x*(x-1)*2-(4*x-3)*(2*y-1))
    # fy = lambda x,y : 2*x*(x*(x-1)*(2*y-1)-(4*x-3)*(y**2-y)) + 2*x*(x**2*(8*y-4) + x*(-12*y**2+6*y+3) + 6*(y-1)*y) + 6*x**3*(x-1)*(-2)+2*x**2*(4*x-3)*(2*y-1)

    # Example 2
#    mu = 0.5 ; lam = 1.
#    u1_ = lambda x,y : 4*x**3*(1-x)*y*(1-y)
#    u2_ = lambda x,y : -4*x**3*(1-x)*y*(1-y)

#    sig11_ = lambda x,y : 4*x**3*(1-x)*(2*y-1) + 4*x**2*y*(1-x)*(1-y) + 8*x**2*y*(4*x-3)*(y-1)
#    sig12_ = lambda x,y : 2*x**2*(x*(x-1)*(2*y-1) - (4*x-3)*(y-1)*y)
#    sig22_ = lambda x,y : 8*x**3*(1-x)*(2*y-1) + 4*x**2*y*(1-x)*(1-y) + 4*x**2*y*(4*x-3)*(y-1)
#    sig_scalar_ = lambda x,y: 4*x**3*(1-x)*(2*y-1)+8*x**2*y*(1-x)*(1-y) + 4*x**2*y*(4*x-3)*(y-1)

#    fx_ = lambda x,y : -4*x*(x**2*(8*y-4)+x*(-27*y**2+21*y+3)+14*(y-1)*y) + ( 4*x**2*(1-x)*(2*y-1) + 4*x*y*(1-x)*(1-y) + 8*x*y*(4*x-3)*(y-1) ) + 2*x**2*(2*x**2+x*(2-8*y)+6*y-3) - (4*x**2*(1-x)*(2*y-1)+8*x*y*(1-x)*(1-y) + 4*x*y*(4*x-3)*(y-1))
#    fy_ = lambda x,y : 2*x*(x**2*(8*y-4)+x*(-12*y**2+6*y+3)+6*(y-1)*y) + 2*x*(x*(x-1)*(2*y-1) - (4*x-3)*(y-1)*y) - 4*x**2*(4*x**2-10*x*y+x+8*y-4)
    
#    divx_ = lambda x,y : -4*x*(x**2*(8*y-4)+x*(-27*y**2+21*y+3)+14*(y-1)*y) + ( 4*x**2*(1-x)*(2*y-1) + 4*x*y*(1-x)*(1-y) + 8*x*y*(4*x-3)*(y-1) ) + 2*x**2*(2*x**2+x*(2-8*y)+6*y-3)
#    divy_ = lambda x,y : 2*x*(x**2*(8*y-4)+x*(-12*y**2+6*y+3)+6*(y-1)*y) + 2*x*(x*(x-1)*(2*y-1) - (4*x-3)*(y-1)*y) - 4*x**2*(4*x**2-10*x*y+x+8*y-4)
    
    mu = 0.5 ; lam = 1

#    u1 = lambda r,z: r**2 * sin(r*pi) * z**2 * cos((z-0.5)*pi)
#    u2 = lambda r,z: -r**2 * sin(r*pi) * z**2 * cos((z-0.5)*pi)

    u1 = lambda r,z : r**3 * sin(r*pi) * cos((z-0.5)*pi)
    u2 = lambda r,z : -u1(r,z)

    u1_r = lambda r,z : r**2 * cos( pi * (z-0.5) ) * (3 * sin(pi *r ) + pi * r * cos(pi*r))
    u1_z = lambda r,z : pi * r**3 * sin(pi * r) * cos(pi * z)
    u1_rr = lambda r,z : r* cos(pi * (z-0.5) ) * ((6-pi**2*r**2)*sin(pi*r) + 6*pi*r*cos(pi*r) )
    u1_rz =  lambda r,z : pi * r**2 * cos(pi *z) * (3 * sin(pi*r) + pi * r * cos(pi*r)) 
    u1_zz = lambda r,z : -pi**2 * r**3 * sin(pi*r) * sin(pi*z)

    u2_r = lambda r,z : -u1_r(r,z)
    u2_z = lambda r,z : -u1_z(r,z)
    u2_rr = lambda r,z : -u1_rr(r,z)
    u2_rz = lambda r,z : -u1_rz(r,z)
    u2_zz = lambda r,z : -u1_zz(r,z)
        
    sig11 = lambda r,z : (2*mu + lam) * u1_r(r,z) + lam * u2_z(r,z) + lam / r * u1(r,z)
    sig11_r = lambda r,z : (2*mu + lam) * u1_rr(r,z) + lam * u2_rz(r,z) + lam / r * (u1_r(r,z) - (1 / r) * u1(r,z) )
    
    sig22 = lambda r,z : (2*mu + lam) * u2_z(r,z) + lam * u1_r(r,z) + lam / r * u1(r,z)    
    sig22_z = lambda r,z : (2*mu + lam) * u2_zz(r,z) +  lam * u1_rz(r,z) + lam / r * u1_z(r,z)
    
    sig12 = lambda r,z : mu * (u1_z(r,z) + u2_r(r,z))
    sig12_r = lambda r,z : mu * ( u1_rz(r,z) + u2_rr(r,z) ) 
    sig12_z = lambda r,z : mu * ( u1_zz(r,z) + u2_rz(r,z) )
    
    sig_scalar = lambda r,z : (2*mu + lam)* (1. / r) * u1(r,z) + lam * u1_r(r,z) + lam * u2_z(r,z)

    fx = lambda r,z : divx(r,z) - (1. / r) * sig_scalar(r,z)
    fy = lambda r,z : divy(r,z)
    
    divx = lambda r, z: (1. / r) * sig11(r,z) + sig11_r(r,z) + sig12_z(r,z) 
    divy = lambda r, z: (1. / r) * sig12(r,z) +  sig12_r(r,z) + sig22_z(r,z)


    true_stress = ft.TrueSolution([sig11, sig12, sig12, sig22])
    true_stress_scalar = ft.TrueSolution([sig_scalar])
    true_displacement = ft.TrueSolution([u1, u2])
    true_divergence = ft.TrueSolution([divx,divy])

    test_space = basis ; trial_space = basis
    quadrature = quad.Quadrature(5)
    reference_element = mt.ReferenceElement()

    val_types_bdm = ['vals', 'Jt_vals', 'div', '|Jt|', 'quad_wght']
    val_types_p0_scalar = ['vals', '|Jt|', 'quad_wght']
    val_types_p0_vec = ['vals', 'quad_wght', '|Jt|']
    val_types_p0_skewtens = ['vals', 'quad_wght', '|Jt|']

    gamma = 1.
    
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

                r_val = ele_quad_pt.vals[0] ; y = ele_quad_pt.vals[1]
                f_vec = np.array([fx(r_val,y),fy(r_val,y)])
                int_val = 0.

                q1 = 1./r_val * val_dic_test['Jt_vals'][:,0] + val_dic_test['div']
                int_val = gamma*(la.Operators.dot_product(f_vec,
                                                          q1)
                                 * val_dic_test['quad_wght']) * ele_quad_pt.vals[0]
                global_rhs.add_val(eN, i, int_val)

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
                               * (1./val_dic_test['|Jt|']) * val_dic_test['quad_wght']) * ele_quad_pt.vals[0]
                    r_val = ele_quad_pt.vals[0]
                    if r_val != 0.:
                        p1_val = 0. ; q1_val = 0.
                        p1_val = 1./r_val * val_dic_trial['Jt_vals'][:,0] + val_dic_trial['div']
                        q1_val = 1./r_val * val_dic_test['Jt_vals'][:,0] + val_dic_test['div']
                        int_val += gamma * ( la.Operators.dot_product(q1_val, p1_val)
                                             * r_val * (1./val_dic_test['|Jt|']) * val_dic_test['quad_wght'] )
                    
                    local_matrix_assembler.add_val(i,j,int_val)

                for j in range(trial_space[1].get_num_dof()):
                    j_tmp = j + trial_space[0].get_num_dof()                    
                    r_val = ele_quad_pt.vals[0]
                    
                    int_val = 0.
                    val_dic_trial = basis[1].get_element_vals(j,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_scalar)
                    int_val = -( (lam / (2*mu*(2*mu+3*lam))) * val_dic_trial['vals']
                                 * val_dic_test['Jt_vals'].trace() * val_dic_test['quad_wght'] ) * r_val

                    int_val -= gamma * la.Operators.first_row_axi_divergence_scalar_product(val_dic_test,
                                                                                            r_val,
                                                                                            val_dic_trial['vals']) * val_dic_test['quad_wght']

                    local_matrix_assembler.add_val(i,j_tmp,int_val)

                for j in range(trial_space[2].get_num_dof()):
                    int_val = 0.
                    val_dic_trial = basis[2].get_element_vals(j,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_vec)
                    div_vals = val_dic_test['div'] * ele_quad_pt.vals[0] + val_dic_test['Jt_vals'][:,0]
                    int_val = (la.Operators.dot_product(val_dic_trial['vals'],
                                                        div_vals)
                               * (1./val_dic_test['|Jt|']) * ele_quad_pt.get_quad_weight() )

                    j_tmp = j + trial_space[0].get_num_dof() + trial_space[1].get_num_dof()
                    local_matrix_assembler.add_val(i,j_tmp,int_val)

                for j in range(trial_space[3].get_num_dof()):
                    int_val = 0.
                    val_dic_trial = basis[3].get_element_vals(j,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_skewtens)
                    # weak symmetry part
                    int_val = (la.Operators.weak_symmetry_dot_product(val_dic_trial['vals'],
                                                                      val_dic_test['Jt_vals'])
                               * val_dic_test['quad_wght']) * ele_quad_pt.vals[0]
                    # div part
                    div_vals = val_dic_test['div'] * ele_quad_pt.vals[0] + val_dic_test['Jt_vals'][:,0]
                    x_perp = np.array([ele_quad_pt[1], -ele_quad_pt[0]])
                    int_val += (la.Operators.dot_product(div_vals, x_perp) * val_dic_trial['vals'][1][0]
                                * (1./val_dic_test['|Jt|']) * ele_quad_pt.get_quad_weight() )

                    j_tmp = j + trial_space[0].get_num_dof() + trial_space[1].get_num_dof() + trial_space[2].get_num_dof()
                    local_matrix_assembler.add_val(i,j_tmp,int_val)

            for i in range(test_space[1].get_num_dof()):
                int_val = 0.
                i_tmp = i + test_space[0].get_num_dof()
                val_dic_test = test_space[1].get_element_vals(i,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_scalar)
                r_val = ele_quad_pt.vals[0] ; y = ele_quad_pt.vals[1]
                fx_val = fx(r_val, y)
                int_val = -gamma * fx_val * val_dic_test['vals'] * val_dic_test['quad_wght'] * val_dic_test['|Jt|']
#                int_val = -gamma * fx_val * val_dic_test['vals'] * val_dic_test['quad_wght']
                global_rhs.add_val(eN, i_tmp, int_val)

                for j in range(trial_space[0].get_num_dof()):
                    r_val = ele_quad_pt.vals[0]
                    int_val = 0.
                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_bdm)
                    val_dic_trial = piola_map.correct_div_space_vals(val_dic_trial,
                                                                     j,
                                                                     trial_space[0])

                    int_val = -( (lam / (2*mu*(2*mu+3*lam))) * val_dic_trial['Jt_vals'].trace()
                                 * val_dic_test['vals'] * val_dic_test['quad_wght'] ) * r_val

                    int_val -= gamma * la.Operators.first_row_axi_divergence_scalar_product(val_dic_trial,
                                                                                            r_val,
                                                                                            val_dic_test['vals']) * val_dic_test['quad_wght']

                    local_matrix_assembler.add_val(i_tmp, j, int_val)

                for j in range(trial_space[1].get_num_dof()):
                    int_val = 0.
                    j_tmp = j + trial_space[0].get_num_dof()

                    val_dic_trial = trial_space[1].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_p0_scalar)
                    
                    int_val = ( (1. / (2*mu)) * val_dic_test['vals'] * val_dic_trial['vals']
                                - (1. / (2*mu) * ( lam / (2 * mu + 3 * lam)) * val_dic_test['vals'] * val_dic_trial['vals'] )
                                 ) * ele_quad_pt.vals[0] * val_dic_test['quad_wght'] * val_dic_test['|Jt|']

                    int_val += gamma * ( val_dic_trial['vals'] * val_dic_test['vals'] * ( 1. / ele_quad_pt.vals[0] )
                                         * val_dic_test['quad_wght'] ) * val_dic_test['|Jt|']

                    local_matrix_assembler.add_val(i_tmp, j_tmp, int_val)

                for j in range(trial_space[2].get_num_dof()):
                    int_val = 0.
                    j_tmp = j + trial_space[0].get_num_dof() + trial_space[1].get_num_dof()

                    val_dic_trial = trial_space[2].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_p0_vec)
                    int_val = -val_dic_trial['vals'][0] * val_dic_test['vals'] * ele_quad_pt.get_quad_weight()
                    local_matrix_assembler.add_val(i_tmp, j_tmp, int_val)

                for j in range(trial_space[3].get_num_dof()):
                    int_val = 0.
                    j_tmp = j + trial_space[0].get_num_dof() + trial_space[1].get_num_dof() + trial_space[2].get_num_dof()

                    val_dic_trial = trial_space[3].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_p0_skewtens)

                    x_perp = np.array([ele_quad_pt[1], -ele_quad_pt[0]])

                    int_val =   -(val_dic_trial['vals'][1][0] * val_dic_test['vals']
                                  * val_dic_test['|Jt|'] * val_dic_test['quad_wght'] ) * x_perp[0]

                    local_matrix_assembler.add_val(i_tmp, j_tmp, int_val)

            for i in range(test_space[2].get_num_dof()):
                i_tmp = i + test_space[0].get_num_dof() + test_space[1].get_num_dof()
                val_dic_test = test_space[2].get_element_vals(i,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_vec)

                # compute and add RHS
                x = ele_quad_pt.vals[0] ; y = ele_quad_pt.vals[1]
                f_vec = np.array([fx(x,y),fy(x,y)])
                int_val = 0.

                int_val = (la.Operators.dot_product(f_vec,
                                                    val_dic_test['vals'])
                           * val_dic_test['|Jt|'] * val_dic_test['quad_wght']) * ele_quad_pt.vals[0]
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
                    div_vals = val_dic_trial['div'] * ele_quad_pt.vals[0] + val_dic_trial['Jt_vals'][:,0] 
                    int_val = (la.Operators.dot_product(div_vals,
                                                        val_dic_test['vals'])
                               * (1./val_dic_test['|Jt|']) * ele_quad_pt.get_quad_weight() )
                    local_matrix_assembler.add_val(i_tmp, j, int_val)

                for j in range(trial_space[1].get_num_dof()):
                    j_tmp = j + trial_space[0].get_num_dof()
                    int_val = 0.

                    val_dic_trial = trial_space[1].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_p0_scalar)
                    int_val = -val_dic_trial['vals'] * val_dic_test['vals'][0] * ele_quad_pt.get_quad_weight()
                    local_matrix_assembler.add_val(i_tmp, j_tmp, int_val)

            for i in range(test_space[3].get_num_dof()):
                i_tmp = (i + test_space[0].get_num_dof()
                         + test_space[1].get_num_dof() + test_space[2].get_num_dof() )
                val_dic_test = test_space[3].get_element_vals(i,
                                                              quad_pt,
                                                              piola_map,
                                                              val_types_p0_skewtens)

                # compute and add RHS
                x = ele_quad_pt.vals[0] ; y = ele_quad_pt.vals[1]
                x_perp = np.array([y, -x])
                f_vec = np.array([fx(x,y),fy(x,y)])
                int_val = 0.

                int_val = (la.Operators.dot_product(f_vec,
                                                    x_perp) * val_dic_test['vals'][1][0]
                           * val_dic_test['|Jt|'] * val_dic_test['quad_wght']) * ele_quad_pt.vals[0]
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
                    int_val = (la.Operators.weak_symmetry_dot_product(val_dic_trial['Jt_vals'],
                                                                      val_dic_test['vals'])
                               * val_dic_test['quad_wght']) * ele_quad_pt.vals[0]
                    #div part
                    div_vals = val_dic_trial['div'] * ele_quad_pt.vals[0] + val_dic_trial['Jt_vals'][:,0]
                    x_perp = np.array([ele_quad_pt[1], -ele_quad_pt[0]])
                    int_val += (la.Operators.dot_product(div_vals, x_perp) * val_dic_test['vals'][1][0]
                                * (1./val_dic_test['|Jt|']) * ele_quad_pt.get_quad_weight() )
                    local_matrix_assembler.add_val(i_tmp, j, int_val)

                for j in range(trial_space[1].get_num_dof()):
                    j_tmp = j + trial_space[0].get_num_dof()
                    int_val = 0.
                    val_dic_trial = trial_space[1].get_element_vals(j,
                                                                    quad_pt,
                                                                    piola_map,
                                                                    val_types_p0_scalar)
                    x_perp = np.array([ele_quad_pt[1], -ele_quad_pt[0]])

                    int_val =  -(val_dic_trial['vals'] * val_dic_test['vals'][1][0]
                                 * val_dic_test['|Jt|'] * val_dic_test['quad_wght'] ) * x_perp[0]
                    local_matrix_assembler.add_val(i_tmp, j_tmp, int_val)
                                                                

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

    for boundary_dof in dof_handler.get_bdy_dof_dic(trial_space[0].get_name()):
        dof_idx, bdy_type, bdy_type_global_idx, bdy_flag, vector_idx = boundary_dof
        if bdy_flag == 1 and bdy_type=='e':
            global_matrix_assembler.set_row_as_dirichlet_bdy(dof_idx)
            global_rhs.set_value(dof_idx,0.)
    
    error_handler = ec.ElasticityErrorHandler(mesh,
                                              dof_handler,
                                              basis,
                                              solution_vec)
    err_stress = error_handler.calculate_stress_error(true_stress,r=1)
    err_div = error_handler.calculate_divergence_error(true_divergence,r=1)
    err_displ = error_handler.calculate_displacement_error(true_displacement,r=1)
    err_skew = error_handler.calculate_skew_symmetry_error(r=1)
    return err_stress, err_div, err_displ, err_skew
    
if __name__ == "__main__":
    h_lst = [1./4, 1./6, 1./8, 1./10, 1./12]
    for h in h_lst:
        mesh = mt.StructuredMesh([1.,1.], h)
#        basis = [bdm.BDMTensBasis(3),
#                 bdm.P3Basis_2D(),
#                 bdm.P2VecBasis_2D(),
#                 bdm.P1SkewTensBasis_2D()]
#        basis = [bdm.BDMTensBasis(3),
#                 bdm.P2Basis_2D(),
#                 bdm.P2VecBasis_2D(),
#                 bdm.P1SkewTensBasis_2D()]
#        basis = [bdm.BDMTensBasis(3),
#                 bdm.P3Basis_2D(),
#                 bdm.P2VecBasis_2D(),
#                 bdm.P2SkewTensBasis_2D()]
#        basis = [bdm.BDMTensBasis(2),
#                 bdm.P2Basis_2D(),
#                 bdm.P1VecBasis_2D(),
#                 bdm.P1SkewTensBasis_2D()]
        basis = [bdm.BDMTensBasis(1),
                 bdm.P1Basis_2D(),
                 bdm.P0VecBasis_2D(),
                 bdm.P0SkewTensBasis_2D()]         
        dof_handler = dof.DOFHandler(mesh,
                                     basis)
        l2_error = axi_elasticity_convergence_script(basis,
                                                     mesh,
                                                     dof_handler)
        print('h : ' + str(Fraction(h).limit_denominator()) + ' error : ' + str(l2_error[0]) + '  ' + str(l2_error[1]) + '  ' + str(l2_error[2]) + ' ' + str(l2_error[3]))
