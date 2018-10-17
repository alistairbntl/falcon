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

def poisson_fe_code(sim_info):
    mesh, basis, dof_handler = sim_info

    num_local_dof = sum([a.get_num_dof() for a in basis])

    global_matrix_assembler = la.GlobalMatrix(dof_handler.get_num_dof(),
                                              dof_handler.get_num_dof())
    local_matrix_assembler = la.LocalMatrixAssembler(dof_handler,
                                                     num_local_dof,
                                                     global_matrix_assembler)

    solution_vec = la.DiscreteSolutionVector(dof_handler.get_num_dof())
    global_rhs = la.GlobalRHS(dof_handler)

    forcing_function = ft.Function((lambda x,y: 1.,))

    test_space = basis ; trial_space = basis
    quadrature = quad.Quadrature(2)
    reference_element = mt.ReferenceElement()

    num_mesh_elements = mesh.get_num_mesh_elements()

    for eN in range(num_mesh_elements):
        element = mesh.get_element(eN)
        reference_map = mp.ReferenceElementMap(element)

        for quad_pt in quadrature.get_element_quad_pts():
            value_types_test = ['dvals','vals','|Jt|','quad_wght']
            value_types_trial = ['dvals','|Jt|','quad_wght']

            for i in range(test_space[0].get_num_dof()):
                val_dic_test = test_space[0].get_element_vals(i,
                                                              quad_pt,
                                                              reference_map,
                                                              value_types_test)

                # rhs construction
                int_val = 0.
                ele_quad_pt = quadrature.get_quad_on_element(reference_map,
                                                             quad_pt)
                vals = forcing_function.get_f_eval(ele_quad_pt)
                int_val = vals[0] * val_dic_test['vals'] * val_dic_test['|Jt|'] * quad_pt.get_quad_weight()
                global_rhs.add_val(eN, i, int_val)

                for j in range(trial_space[0].get_num_dof()):
                    int_val = 0.

                    val_dic_trial = trial_space[0].get_element_vals(j,
                                                                    quad_pt,
                                                                    reference_map,
                                                                    value_types_trial)

                    int_val = la.Operators.dot_product(val_dic_trial['dvals'],val_dic_test['dvals']) * val_dic_test['|Jt|'] * quad_pt.get_quad_weight()
                    local_matrix_assembler.add_val(i,j,int_val)

        local_matrix_assembler.distribute_local_2_global(eN)
        local_matrix_assembler.reset_matrix()

    global_matrix_assembler.initialize_sparse_matrix()
    global_matrix_assembler.set_csr_rep()

    csr_rep = global_matrix_assembler.get_csr_rep()
    from scipy.sparse.linalg import norm
    
    for basis_ele in trial_space:
        for boundary_dof in dof_handler.get_bdy_dof_dic(basis_ele.get_name()):
            dof_idx, bdy_type, bdy_type_global_idx = boundary_dof
            global_matrix_assembler.set_row_as_dirichlet_bdy(dof_idx)
            vals = mesh.get_node(dof_idx).vals
            val = forcing_function.get_f_eval(vals)
            global_rhs.set_value(dof_idx,0.)

    global_matrix_assembler.solve(global_rhs, solution_vec)

    solu_plot = ec.VisualizationHandler(mesh,dof_handler,basis[0],solution_vec)
    solu_plot.output_nodal_solution('plot_test_1')

def test_poisson():
    mesh = mt.StructuredMesh([2,2],0.0625)
    basis = [bdm.P1Basis_2D()]
    dof_handler = dof.DOFHandler(mesh, basis)
    mesh_1 = (mesh, basis, dof_handler)
    poisson_fe_code(mesh_1)

