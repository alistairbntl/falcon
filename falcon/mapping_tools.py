import numpy as np
import math
import copy

from falcon import mesh_tools as mt

class ReferenceElementMap(object):

    def __init__(self,element):
        self.element = element
        self._initialize_affine_map()

    def _initialize_affine_map(self):
        p1 = self.element.get_point(0)
        p2 = self.element.get_point(1)
        p3 = self.element.get_point(2)

        self._jacobian_mat = np.array([[p2[0]-p1[0], p3[0]-p1[0]],
                                       [p2[1]-p1[1], p3[1]-p1[1]] ])

        self._jacobian_mat_t = [[p2[0]-p1[0], p2[1]-p1[1]],
                                [p3[0]-p1[0], p3[1]-p1[1]]]
        
        self._jacobian_det = ( self._jacobian_mat[0][0]*self._jacobian_mat[1][1]
                                 - self._jacobian_mat[1][0]*self._jacobian_mat[0][1] )
        
        self._jacobian_mat_t_inv = np.array([[p3[1]-p1[1], -(p2[1]-p1[1])],
                                             [-(p3[0]-p1[0]), p2[0]-p1[0]] ])

        self._jacobian_mat_t_inv *= 1./self._jacobian_det
        

        self._affine_map = [lambda xi, eta: p1[0] + (p2[0]-p1[0])*xi + (p3[0]-p1[0])*eta,
                            lambda xi, eta: p1[1] + (p2[1]-p1[1])*xi + (p3[1]-p1[1])*eta]

        self._inv_affine_map = [lambda x, y: 1./self._jacobian_det * ((p3[1]-p1[1])*(x-p1[0]) - (p3[0] - p1[0])*(y - p1[1])) ,
                                lambda x, y: 1./self._jacobian_det * (-(p2[1]-p1[1])*(x-p1[0]) + (p2[0] - p1[0])*(y - p1[1]))]

    def get_affine_map(self):
        return self._affine_map

    def apply_jacobian_mat(self,xi,eta):
        if xi.shape == (2,):
            x_val = self._jacobian_mat.dot(xi)
            y_val = self._jacobian_mat.dot(eta)
        else:
            x_val, y_val = self._jacobian_mat.dot([xi,eta])
        return np.array([x_val,y_val])

    def apply_affine_map(self,xi,eta):
        x_val = self._affine_map[0](xi,eta)
        y_val = self._affine_map[1](xi,eta)
        return mt.Point(x_val, y_val)

    def apply_inverse_affine_map(self,x,y):
        xi_val = self._inv_affine_map[0](x,y)
        eta_val = self._inv_affine_map[1](x,y)
        return (xi_val, eta_val)

    def apply_inv_transpose_jacobian_mat(self,xi,eta):
        x_val , y_val = self._jacobian_mat_t_inv.dot([xi,eta])
        return x_val, y_val
    
    def get_jacobian_mat(self):
        return self._jacobian_mat

    def get_jacobian_det(self):
        return abs(self._jacobian_det)

class PiolaMap(ReferenceElementMap):

    def __init__(self,element):
        super(PiolaMap,self).__init__(element)

    def apply_piola_map(self,
                        vals,
                        x=None,
                        y=None,
                        xi=None,
                        eta=None):
        vals_up = self.apply_jacobian_mat(vals['vals'][0],vals['vals'][1])
        vals0 = 1./abs(self.get_jacobian_det())*vals_up[0]
        vals1 = 1./abs(self.get_jacobian_det())*vals_up[1]
        return (vals0, vals1)

    def correct_div_space_vals(self,
                               vals,
                               i,
                               basis):
        """ 
        Directly applying the Piola map to elements on the
        reference triangle does not automatically preseve an element's
        continunity with respect to the normal component.  This
        function will update values from an Hdiv basis to ensure
        continutiy will be preseved on the physical domain.

        inputs
        ------
        vals : dic
           Dictionary of values
        i : idx
           Referenence element DOF index
        """
        new_vals = copy.deepcopy(vals)
        adjusted_vals = ['vals','div']
        dof_per_edge = basis.get_degree()+1   #ARB - note this is set up for bdm spaces
        # and may not work for other div_free spaces

        if i < basis.get_num_edge_dof():
            edge_num = i % 3
            physical_element_edge = self.element.get_edge(edge_num)            
            reference_element_edge_length = mt.ReferenceElement.get_edge_length(edge_num)
            physical_element_edge_length = physical_element_edge.get_edge_length()
            norm_scale = np.sign(physical_element_edge.get_outward_unit_normal_vec(self.element)[0]
                                 *physical_element_edge.get_unit_normal_vec()[0])
            if norm_scale == 0:
                norm_scale = np.sign(physical_element_edge.get_outward_unit_normal_vec(self.element)[1]
                                     *physical_element_edge.get_unit_normal_vec()[1])                
            scaling_factor = physical_element_edge_length*norm_scale/reference_element_edge_length

        # scale interior elements by one of the triangle edge lengths
        if i >= basis.get_num_edge_dof():
            scaling_factor = self.element.get_edge(1).get_edge_length()

        for val_type in adjusted_vals:
            if val_type=='vals':
                try:
                    update_vals = [0,0]
                    update_vals[0] = scaling_factor*vals['vals'][0]
                    update_vals[1] = scaling_factor*vals['vals'][1]
                    new_vals['vals'] = np.array(update_vals)
                    new_vals['Jt_vals'] = self.apply_jacobian_mat(update_vals[0],
                                                                  update_vals[1])
                except KeyError:
                    pass
            if val_type=='div':
                try:
                    update_div = scaling_factor*vals['div']
                    new_vals['div'] = update_div
                except KeyError:
                    pass
        return new_vals
