import math
import csv
import numpy as np
import sys

import quadrature as quad
import mesh_tools as mt
import mapping_tools as mpt

class SolutionHandler(object):

    """
    The SolutionHandler class organizes the information and classes
    needed to calculate solution approximations, generate visual data
    and find errors.

    Arguments
    ---------
    mesh : :class: mesh
        Finite element discretization.
    dof_handler : :class: dof_handler
        Finite element dof_handler class
    basis : lst
        A list of the finite element bases used in the problem.
    solution :
    """

    def __init__(self,
                 mesh,
                 dof_handler,
                 basis,
                 solution):
        self.set_mesh(mesh)
        self.set_dof_handler(dof_handler)
        self.set_basis(basis)
        self.set_solution(solution)
        self.set_basis_dof_indices()

    def set_mesh(self,mesh):
        self._mesh = mesh

    def get_mesh(self):
        return self._mesh

    def set_dof_handler(self,dof_handler):
        self._dof_handler = dof_handler

    def get_dof_handler(self):
        return self._dof_handler

    def set_basis(self,basis):
        self._basis = basis

    def get_basis(self):
        return self._basis

    def get_sub_basis(self,num):
        return self._basis[num]

    def get_num_bases(self):
        """
        Return the number of finite element bases used in the
        simulation.

        Return
        ------
        num_bases : int
        """
        return len(self.get_basis())

    def set_basis_dof_indices(self):
        dof_lst = [basis.get_num_dof() for basis in self.get_basis()]
        self._dof_lst = np.append([0], np.cumsum(dof_lst))

    def get_basis_dof_indices(self, basis_idx):
        """ Return the dof indices for the basis with basis_idx.

        Arguments
        ---------
        basis_idx : int
            Index of the desired basis element

        Return
        ------
        dof_indices : lst
            List of the degree-of-freedom range
        """
        return [self._dof_lst[basis_idx],
                self._dof_lst[basis_idx + 1]]

    def get_num_preceeding_local_dof(self,basis_idx):
        """
        Return the number of dof indices preceeding the current basis
        on a single element.

        Arguments
        ---------
        basis_idx : int
            Index of the desired basis element

        Return
        ------
        num_dofs : int
            The number of dof indices preceeding the current basis
            on a single element.
        """
        return self._dof_lst[basis_idx]

    def set_solution(self,solution):
        self._solution = solution

    def get_solution(self):
        return self._solution

    def get_pressure_solution_approx(self,
                                     element,
                                     reference_map,
                                     piola_map,
                                     quad_pt):
        """
        This function uses the finite element solution and basis to get the
        complete solution approximation.

        Arguments
        ---------
        element :
        reference_map :
        piola_map :
        quad_pt : quad_pt
        """
        basis = self.get_sub_basis(1)
        translation_factor = self.get_sub_basis(0).get_num_dof()
        # TODO - ARB, I think the basis would be better provided as a function call
        # rather than assuming a specific basis ordering.
        value_types_solution = ['vals', 'quad_wght']
        sol_val = 0

        for i in range(basis.get_num_dof()):
            basis_val = basis.get_element_vals(i,
                                               quad_pt,
                                               reference_map,
                                               value_types_solution)

            j = self.get_dof_handler().get_local_2_global(element.get_global_idx(),
                                                          translation_factor + i)
            basis_val = self.get_solution().scale_basis_by_solution(j,
                                                                    basis_val['vals'])
            sol_val += basis_val
        return sol_val

    def get_bdm_solution_approx(self,
                                element,
                                reference_map,
                                piola_map,
                                quad_pt):
        """
        This function uses the finite element solution and basis to get the
        complete solution approximation.

        Arguments
        ---------
        element :
        reference_map :
        piola_map :
        quad_pt : quad_pt
        """
        basis = self.get_sub_basis(0)
        # TODO - ARB, I think the basis would be better provided as a function call
        # rather than assuming a specific basis ordering.
        value_types_solution = ['vals', 'Jt_vals', '|Jt|', 'quad_wght']
        sol_val = 0

        for i in range(basis.get_num_dof()):
            basis_val = basis.get_element_vals(i,
                                               quad_pt,
                                               reference_map,
                                               value_types_solution)
            Jt_det = basis_val['|Jt|']
            basis_val = piola_map.correct_div_space_vals(basis_val,
                                                         i,
                                                         basis)

            j = self.get_dof_handler().get_local_2_global(element.get_global_idx(),
                                                          i)
            basis_val = self.get_solution().scale_basis_by_solution(j,
                                                                    basis_val['Jt_vals'])
            sol_val += 1./Jt_det * basis_val
        return sol_val

    def get_bdm_tens_div_solution_approx(self,
                                         element,
                                         reference_map,
                                         piola_map,
                                         quad_pt):
        """
        This function uses the finite element solution and basis to get the
        complete solution approximation.

        Arguments
        ---------
        element :
        reference_map :
        piola_map :
        quad_pt : quad_pt
        """
        basis = self.get_sub_basis(0)
        ele_quad_pt = reference_map.apply_affine_map(quad_pt.vals[0],
                                                     quad_pt.vals[1])
        # TODO - ARB, I think the basis would be better provided as a function call
        # rather than assuming a specific basis ordering.
        value_types_solution = ['vals', 'Jt_vals', 'div', '|Jt|', 'quad_wght']
        sol_val = 0

        for i in range(basis.get_num_dof()):
            basis_val = basis.get_element_vals(i,
                                               quad_pt,
                                               reference_map,
                                               value_types_solution)
            Jt_det = basis_val['|Jt|']
            basis_val = piola_map.correct_div_space_vals(basis_val,
                                                         i,
                                                         basis)
            div_val = basis_val['div'] + (1./ele_quad_pt[0])*basis_val['Jt_vals'][:,0]

            j = self.get_dof_handler().get_local_2_global(element.get_global_idx(),
                                                          i)
            div = self.get_solution().scale_basis_by_solution(j,
                                                              div_val)
            sol_val += 1./Jt_det * div
        return sol_val

    def get_bdm_axi_div_solution_approx(self,
                                        element,
                                        reference_map,
                                        piola_map,
                                        quad_pt):
        """
        This function uses the finite element solution and basis to get the
        complete solution approximation.

        Arguments
        ---------
        element :
        reference_map :
        piola_map :
        quad_pt : quad_pt
        """
        basis = self.get_sub_basis(0)
        ele_quad_pt = reference_map.apply_affine_map(quad_pt.vals[0],
                                                     quad_pt.vals[1])
        # TODO - ARB, I think the basis would be better provided as a function call
        # rather than assuming a specific basis ordering.
        value_types_solution = ['vals', 'Jt_vals', 'div', '|Jt|', 'quad_wght']
        sol_val = 0

        for i in range(basis.get_num_dof()):
            basis_val = basis.get_element_vals(i,
                                               quad_pt,
                                               reference_map,
                                               value_types_solution)
            Jt_det = basis_val['|Jt|']
            basis_val = piola_map.correct_div_space_vals(basis_val,
                                                         i,
                                                         basis)
            axi_div = 1./ele_quad_pt[0] * basis_val['Jt_vals'][0] + basis_val['div']

            j = self.get_dof_handler().get_local_2_global(element.get_global_idx(),
                                                          i)
            axi_div = self.get_solution().scale_basis_by_solution(j,
                                                                  axi_div)
            sol_val += 1./Jt_det * axi_div
        return sol_val

    def get_element_solution_approx(self,
                                    element,
                                    basis_idx,
                                    quad_pt):
        """
        Finds the solution approximation for an element based finite
        element.

        Arguments
        ---------
        basis : :class:`basis`
            The specific element basis used in the finite element
            calculation.
        """
        num_preceeding_dof = self.get_num_preceeding_local_dof(basis_idx)
        basis = self.get_sub_basis(basis_idx)
        value_types_solution = ['vals']
        sol_val = 0.
        for i in range(basis.get_num_dof()):
            basis_val = basis.get_element_vals(i,
                                               quad_pt,
                                               None,
                                               value_types_solution)
            dof_idx = self.get_dof_handler().get_local_2_global(element.get_global_idx(),
                                                                num_preceeding_dof+i)
            basis_val = self.get_solution().scale_basis_by_solution(dof_idx,
                                                                    basis_val['vals'])
            sol_val += basis_val
        return sol_val

    def get_nodal_solution(self,
                           node):
        basis = self.get_basis()
        value_types_solution = ['vals']
        sol_val = 0
        quad_pt = mt.QuadraturePoint(node.vals[0],
                                     node.vals[1],
                                     1.)
        for i in range(basis.get_num_dof()):
            basis_val = basis.get_element_vals(i,
                                               quad_pt,
                                               None,
                                               value_types_solution)
            basis_val = self.get_solution().scale_basis_by_solution(node.get_global_idx(),
                                                                    basis_val['vals'])
            sol_val += basis_val
        return sol_val

class DarcyErrorHandler(SolutionHandler):

    def __init__(self,
                 mesh,
                 dof_handler,
                 basis,
                 solution):
        super(DarcyErrorHandler, self).__init__(mesh,
                                                dof_handler,
                                                basis,
                                                solution)

    def calculate_hdiv_vel_error(self,
                                 true_solution,
                                 r=0):
        """
        Calculates the velocity error.

        Arguments
        ---------
        r :
            Power of the r for the inner-product
        """

        mesh = self.get_mesh()
        k = self.get_basis()[0].get_degree()
        quadrature = quad.Quadrature(4)
        # the choice of quadrature point is a bit trickier when r \neq 0
        # i think this should work for many cases, but you do need to be
        # careful

        num_mesh_elements = mesh.get_num_mesh_elements()
        element_errors = np.zeros(num_mesh_elements)

        for eN in range(num_mesh_elements):
            element = mesh.get_element(eN)
            reference_map = mpt.ReferenceElementMap(element)
            piola_map = mpt.PiolaMap(element)

            for quad_pt in quadrature.get_element_quad_pts():

                ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                             quad_pt)

                sol_val = self.get_bdm_solution_approx(element,
                                                       reference_map,
                                                       piola_map,
                                                       quad_pt)

                div_val = self.get_bdm_axi_div_solution_approx(element,
                                                               reference_map,
                                                               piola_map,
                                                               quad_pt)
                r_val = ele_quad_pt[0]**r

                true_solution_vals = true_solution.get_f_vel_eval(ele_quad_pt)
                true_solution_div_vals = true_solution.get_div_f_vel_eval(ele_quad_pt)

                u_error_sq = (true_solution_vals[0]-sol_val[0])**2
                v_error_sq = (true_solution_vals[1]-sol_val[1])**2
                div_error_sq = (true_solution_div_vals[0]-div_val)**2
                element_errors[eN] += (u_error_sq + v_error_sq + div_error_sq)*r_val*ele_quad_pt.get_quad_weight()

        return math.sqrt(element_errors.sum())
        
    def calculate_l2_vel_error(self,
                               true_solution,
                               r=0):
        """
        Calculates the velocity error.

        Arguments
        ---------
        r :
            Power of the r for the inner-product
        """

        mesh = self.get_mesh()
        k = self.get_basis()[0].get_degree()
#        quadrature = quad.Quadrature(k+1+r)
        quadrature = quad.Quadrature(4)
        # the choice of quadrature point is a bit trickier when r \neq 0
        # i think this should work for many cases, but you do need to be
        # careful

        num_mesh_elements = mesh.get_num_mesh_elements()
        element_errors = np.zeros(num_mesh_elements)

        for eN in range(num_mesh_elements):
            element = mesh.get_element(eN)
            reference_map = mpt.ReferenceElementMap(element)
            piola_map = mpt.PiolaMap(element)

            for quad_pt in quadrature.get_element_quad_pts(): 
                
                ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                             quad_pt)

                sol_val = self.get_bdm_solution_approx(element,
                                                       reference_map,
                                                       piola_map,
                                                       quad_pt)
                r_val = ele_quad_pt[0]**r

                true_solution_vals = true_solution.get_f_vel_eval(ele_quad_pt)

                u_error_sq = (true_solution_vals[0]-sol_val[0])**2
                v_error_sq = (true_solution_vals[1]-sol_val[1])**2
                element_errors[eN] += (u_error_sq + v_error_sq)*r_val*ele_quad_pt.get_quad_weight()

        return math.sqrt(element_errors.sum())

    def calculate_l2_pressure_error(self,
                                    true_solution,
                                    r=0):
        """
        Calculates the velocity error.

        Arguments
        ---------
        r :
            Power of the r for the inner-product
        """
        mesh = self.get_mesh()
        k = self.get_basis()[1].get_degree()
        quadrature = quad.Quadrature(4)

        num_mesh_elements = mesh.get_num_mesh_elements()
        element_errors = np.zeros(num_mesh_elements)

        for eN in range(num_mesh_elements):
            element = mesh.get_element(eN)
            reference_map = mpt.ReferenceElementMap(element)
            piola_map = mpt.PiolaMap(element)

            for quad_pt in quadrature.get_element_quad_pts():

                ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                             quad_pt)

                sol_val = self.get_pressure_solution_approx(element,
                                                            reference_map,
                                                            piola_map,
                                                            quad_pt)
                r_val = ele_quad_pt[0]**r

                true_solution_vals = true_solution.get_p_eval(ele_quad_pt)

                p_error_sq = (true_solution_vals-sol_val)**2
                element_errors[eN] += (p_error_sq)*r_val*ele_quad_pt.get_quad_weight()

        return math.sqrt(element_errors.sum())

class ElasticityErrorHandler(SolutionHandler):

    def __init__(self,
                 mesh,
                 dof_handler,
                 basis,
                 solution):
        super(ElasticityErrorHandler,self).__init__(mesh,
                                                    dof_handler,
                                                    basis,
                                                    solution)
        self.quad_degree = 4

    def calculate_stress_error(self,
                               true_solution,
                               r=0):
        # ARB Note - I have no idea if get_bdm_solution_approx
        # works correctly in this context
        mesh = self.get_mesh()
        k = self.get_basis()[0].get_degree()
        quadrature = quad.Quadrature(self.quad_degree)

        num_mesh_elements = mesh.get_num_mesh_elements()
        element_errors = np.zeros(num_mesh_elements)

        for eN in range(num_mesh_elements):
            element = mesh.get_element(eN)
            reference_map = mpt.ReferenceElementMap(element)
            piola_map = mpt.PiolaMap(element)

            for quad_pt in quadrature.get_element_quad_pts():
                sol_val = self.get_bdm_solution_approx(element,
                                                       reference_map,
                                                       piola_map,
                                                       quad_pt)
                ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                             quad_pt)
                r_val = ele_quad_pt[0]**r

                true_solution_vals = true_solution.get_f_eval(ele_quad_pt)
                sig11_error_sq = (true_solution_vals[0]-sol_val[0][0])**2
                sig12_error_sq = (true_solution_vals[1]-sol_val[0][1])**2
                sig21_error_sq = (true_solution_vals[2]-sol_val[1][0])**2
                sig22_error_sq = (true_solution_vals[3]-sol_val[1][1])**2
                element_errors[eN] += (sig11_error_sq +
                                       sig12_error_sq +
                                       sig21_error_sq +
                                       sig22_error_sq) * r_val* ele_quad_pt.get_quad_weight()

        return math.sqrt(element_errors.sum())

    def calculate_skew_symmetry_error(self,
                                      r=0):
        mesh = self.get_mesh()
        k = self.get_basis()[0].get_degree()
        quadrature = quad.Quadrature(self.quad_degree)

        num_mesh_elements = mesh.get_num_mesh_elements()
        element_errors = np.zeros(num_mesh_elements)

        for eN in range(num_mesh_elements):
            element = mesh.get_element(eN)
            reference_map = mpt.ReferenceElementMap(element)
            piola_map = mpt.PiolaMap(element)

            for quad_pt in quadrature.get_element_quad_pts():
                sol_val = self.get_bdm_solution_approx(element,
                                                       reference_map,
                                                       piola_map,
                                                       quad_pt)
                ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                             quad_pt)
                r_val = ele_quad_pt[0]**r

                skew_error = (sol_val[1][0] - sol_val[0][1])**2
                element_errors[eN] += skew_error * r_val* ele_quad_pt.get_quad_weight()

        return math.sqrt(element_errors.sum())

    def calculate_skew_symmetry_element_integral(self,
                                                 eN,
                                                 r=0):
        mesh = self.get_mesh()
        k = self.get_basis()[0].get_degree()
        quadrature = quad.Quadrature(4)

        element = mesh.get_element(eN)
        reference_map = mpt.ReferenceElementMap(element)
        piola_map = mpt.PiolaMap(element)

        int_val = 0.

        for quad_pt in quadrature.get_element_quad_pts():
            sol_val = self.get_bdm_solution_approx(element,
                                                   reference_map,
                                                   piola_map,
                                                   quad_pt)
            ele_quad_pt = quadrature.get_quad_on_element(piola_map,
                                                         quad_pt)
            r_val = ele_quad_pt[0]**r
            sig_error = (sol_val[0][1] - sol_val[1][0])
            int_val += sig_error * r_val * ele_quad_pt.get_quad_weight()

        return int_val

    def calculate_divergence_error(self,
                                   true_solution,
                                   r=0):
        mesh = self.get_mesh()
        k = self.get_basis()[0].get_degree()
        quadrature = quad.Quadrature(self.quad_degree)

        num_mesh_elements = mesh.get_num_mesh_elements()
        element_errors = np.zeros(num_mesh_elements)
        basis = self.get_sub_basis(0)

        for eN in range(num_mesh_elements):
            element = mesh.get_element(eN)
            reference_map = mpt.ReferenceElementMap(element)
            piola_map = mpt.PiolaMap(element)

            for quad_pt in quadrature.get_element_quad_pts():
                sol_val = self.get_bdm_tens_div_solution_approx(element,
                                                                reference_map,
                                                                piola_map,
                                                                quad_pt)

                ele_quad_pt = quadrature.get_quad_on_element(reference_map,
                                                             quad_pt)
                r_val = ele_quad_pt[0]**r

                true_solution_vals = true_solution.get_f_eval(ele_quad_pt)
                t1_error = (true_solution_vals[0] - sol_val[0])**2
                t2_error = (true_solution_vals[1] - sol_val[1])**2
                element_errors[eN] += (t1_error + t2_error) * r_val * ele_quad_pt.get_quad_weight()
        return math.sqrt(element_errors.sum())
        
    def calculate_displacement_error(self,
                                     true_solution,
                                     r=0):
        mesh = self.get_mesh()
        k = self.get_basis()[1].get_degree()
        quadrature = quad.Quadrature(self.quad_degree)

        num_mesh_elements = mesh.get_num_mesh_elements()
        element_errors = np.zeros(num_mesh_elements)
        basis = self.get_sub_basis(1)

        for eN in range(num_mesh_elements):
            element = mesh.get_element(eN)
            reference_map = mpt.ReferenceElementMap(element)

            for quad_pt in quadrature.get_element_quad_pts():
                sol_val = self.get_element_solution_approx(element=element,
                                                           basis_idx=1,
                                                           quad_pt=quad_pt)

                ele_quad_pt = quadrature.get_quad_on_element(reference_map,
                                                             quad_pt)
                r_val = ele_quad_pt[0]**r

                true_solution_vals = true_solution.get_f_eval(ele_quad_pt)
                u1_error_sq = (true_solution_vals[0] - sol_val[0])**2
                u2_error_sq = (true_solution_vals[1] - sol_val[1])**2
                element_errors[eN] += (u1_error_sq + u2_error_sq) * r_val * ele_quad_pt.get_quad_weight()
        return math.sqrt(element_errors.sum())

# class VisualizationHandler(SolutionHandler):

#     def __init__(self,
#                  mesh,
#                  dof_handler,
#                  basis,
#                  solution):
#         super(VisualizationHandler, self).__init__(mesh,
#                                                    dof_handler,
#                                                    basis,
#                                                    solution)

#     def save_mesh_info(self,
#                        file_name):
#         mesh = self.get_mesh()
#         mesh_info = mesh.get_mesh_plt_info()
#         np.save(file_name+'_x',mesh_info[0])
#         np.save(file_name+'_y',mesh_info[1])
#         np.save(file_name+'_triangles',mesh_info[2])

#     def save_solu_info(self,
#                        file_name,
#                        solu_info):
#         np.save(file_name+'_u', solu_info)

#     def output_nodal_solution(self,
#                               file_name,
#                               true_sol=None):
#         """
#         Calculates the nodal solution values for a scalar valued
#         problem.

#         Arguments
#         ---------
#         filename : str
#             Name for output data file

#         true_sol : func
#             Function handle for true solution.
#             Note - currently not implemented
#         """
#         mesh = self.get_mesh()
#         mesh_info = mesh.get_mesh_plt_info()
#         k = self.get_basis().get_degree()

#         num_mesh_elements = mesh.get_num_mesh_elements()
#         num_mesh_nodes = mesh.get_num_mesh_nodes()

#         node_val = np.zeros(num_mesh_nodes)
#         num_mesh_nodes = mesh.get_num_mesh_nodes()

#         for nN in range(num_mesh_nodes):
#             node = mesh.get_node(nN)
#             global_node_idx = node.get_global_idx()
#             node_quad_pt = mt.QuadraturePoint(node.vals[0],
#                                               node.vals[1],
#                                               1.)
#             sol_val = self.get_nodal_solution(node)
#             node_val[global_node_idx] = sol_val

#         self.save_mesh_info(file_name)
#         self.save_solu_info(file_name,
#                             node_val)


    # def output_solution(self,
    #                     file_name,
    #                     num_components,
    #                     true_sol=None,):

    #     mesh = self.get_mesh()
    #     mesh_info = mesh.get_mesh_plt_info()
    #     k = self.get_basis().get_degree()
    #     quadrature = quad.Quadrature(k+1)

    #     num_mesh_elements = mesh.get_num_mesh_elements()
    #     element_errors = np.zeros(num_mesh_elements)

    #     mesh_node_eval_indicator = dict([(i,False) for i in range(mesh.get_num_mesh_nodes())])
    #     node_val = [] ; true_node_val = [] ; error = []
    #     num_nodes = mesh.get_num_mesh_nodes()

    #     for i in range(num_components):
    #         node_val.append(np.zeros(num_nodes))
    #         true_node_val.append(np.zeros(num_nodes))
    #         error.append(np.zeros(num_nodes))

    #     for eN in range(num_mesh_elements):
    #         element = mesh.get_element(eN)
    #         reference_map = mpt.ReferenceElementMap(element)
    #         piola_map = mpt.PiolaMap(element)
    #         for node in element.get_nodes():
    #             global_node_idx = node.get_global_idx()
    #             if mesh_node_eval_indicator[global_node_idx] is False:
    #                 node_quad_pt = mt.QuadraturePoint(node.vals[0],
    #                                                   node.vals[1],
    #                                                   1.)
    #                 sol_val = self.get_solution_approx(element,
    #                                                    reference_map,
    #                                                    piola_map,
    #                                                    node_quad_pt)
    #                 xi = node.vals[0]
    #                 yi = node.vals[1]
    #                 for i in range(num_components-1):
    #                     node_val[i][global_node_idx] = sol_val[i]
    #                     true_node_val[i][global_node_idx] = true_sol[i](xi,yi)
    #                     error[i][global_node_idx] = sol_val[i] - true_sol[i](xi,yi)
    #                 mesh_node_eval_indicator[global_node_idx]==True

    #     self.save_solution_output(file_name,
    #                               mesh_info,
    #                               node_val,
    #                               true_node_val,
    #                               error)

    # def save_solution_output(self,
    #                          file_name,
    #                          mesh_info,
    #                          solu_info,
    #                          true_solu_info,
    #                          error_info):
    #     np.save(file_name+'_x',mesh_info[0])
    #     np.save(file_name+'_y',mesh_info[1])
    #     np.save(file_name+'_triangles',mesh_info[2])
    #     np.save(file_name+'_u',solu_info[0])
    #     np.save(file_name+'_v',solu_info[1])
    #     np.save(file_name+'_utrue',true_solu_info[0])
    #     np.save(file_name+'_vtrue',true_solu_info[1])
    #     np.save(file_name+'_uerror',error_info[0])
    #     np.save(file_name+'_verror',error_info[1])

        # with open('file_name'+'.csv', 'w') as csvfile:
        #     solu_file_x = csv.writer(csvfile, delimiter=',')
        #     solu_file_y = csv.writer(csvfile, delimiter=',')
        #     solu_file_x.writerow(['x_coord', 'y_coord','z_coord','scalar'])
        #     for eN in range(num_mesh_elements):
        #         element = mesh.get_element(eN)
        #         reference_map = mpt.ReferenceElementMap(element)
        #         piola_map = mpt.PiolaMap(element)
        #         for quad_pt in quadrature.get_element_quad_pts():
        #             sol_val = self.get_solution_approx(element,
        #                                                reference_map,
        #                                                piola_map,
        #                                                quad_pt)
        #             ele_quad_pt = quadrature.get_quad_on_element(piola_map,
        #                                                          quad_pt)
        #             solu_file_x.writerow([ele_quad_pt[0],
        #                                   ele_quad_pt[1],
        #                                   0,
        #                                   sol_val[0]])
        #             # solu_file_y.writerow([ele_quad_pt[0],
        #             #                       ele_quad_pt[1],
        #             #                       sol_val[1]])
