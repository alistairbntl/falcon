import numpy as np
import math

import quadrature as quad
import mesh_tools as mt

class Empty(object):

    def __init__(self):
        pass

class Basis(object):

    def __init__(self):
        self._set_function_dispatcher()

    def set_degree(self,k):
        self._k = k

    def get_degree(self):
        return self._k

    def _set_function_dispatcher(self):
        self._function_dispatcher = {'|Jt|': self.get_abs_det_jacobian,
                                     'quad_wght': self.get_quad_wght}

    def get_element_vals(self,
                         basis_func_idx,
                         quad_pt,
                         mapping,
                         value_types):
        """
        This function calculates basis values

        Arguments
        ---------
        basis_func_idx : int
            The current basis function
        quadrature : Quadrature
            Relvant quadrature information.
        mapping : Mapping
            The mapping object between the reference element and
            physical domain
        value_type : lst
            This list indicates the information needed for the finite
            element calculation

        Returns
        -------
        vals_dic : dic
            Dictionary with the function values
        """
        vals_dic = {}
        for val_type in value_types:
            val = self._function_dispatcher[val_type](basis_func_idx,
                                                      quad_pt,
                                                      mapping)
            vals_dic[val_type] = val
        return vals_dic

    def get_abs_det_jacobian(self,
                             basis_func_idx,
                             quad_pt,
                             mapping):
        return mapping.get_jacobian_det()

    def get_quad_wght(self,
                      basis_func_idx,
                      quad_pt,
                      mapping):
        return quad_pt.get_quad_weight()

class P0Basis_2D(Basis):

    def __init__(self):
        super(P0Basis_2D,self).__init__()
        self.set_degree(0)
        self._initialize_element_functions()
        self._initialize_edge_functions()
        self._set_local_function_dispatcher()

    def _set_local_function_dispatcher(self):
        local_functions = {'vals' : self.get_func_val}
        self._function_dispatcher.update(local_functions)

    def _set_basis_lst(self):
        if self.get_degree() == 0:
            basis_func = self._basis_funcs
            self._basis_lst = [basis_func[0]]

    def get_name(self):
        return "p0_basis"

    def get_func_val(self,
                     basis_func_idx,
                     quad_pt,
                     mapping):
        basis_func = self.get_basis_func(basis_func_idx)
        xi = quad_pt.vals[0] ; eta = quad_pt.vals[1]
        return basis_func(xi,eta)

    def get_basis_func(self,idx):
        try:
            basis_func = self._basis_lst[idx]
        except AttributeError:
            self._set_basis_lst()
            basis_func = self._basis_lst[idx]
        return basis_func

    def get_basis_func_lst(self):
        try:
            basis_func = self._basis_lst
        except AttributeError:
            self._set_basis_lst()
            basis_func = self._basis_lst
        return basis_func
    
        
    def get_num_dof(self):
        return 1

    def get_num_interior_dof(self):
        return 1

    def get_num_dof_per_edge(self):
        return 0

    def get_num_dof_per_node(self):
        return 0

    def _initialize_element_functions(self):
        self._basis_funcs = [lambda xi, eta : 1.0]

    def _initialize_edge_functions(self):
        self._funcs_on_bdy = [[],[],[]]
        for i in range(3):
            self._funcs_on_bdy[i].append(lambda xi, eta: 1.0)

    def get_func_on_bdy(self, edge_num, dof_num):
        return self._funcs_on_bdy[edge_num][dof_num]

    def get_func(self,dof_num):
        return self._basis_funcs[dof_num]

class P0SkewTensBasis_2D(Basis):

    def __init__(self):
        super(P0SkewTensBasis_2D,self).__init__()
        self.set_degree(0)
        self.P0_basis_2D = P0Basis_2D()
        self._set_local_function_dispatcher()

    def _set_local_function_dispatcher(self):
        local_functions = {'vals' : self.get_func_val}
        self._function_dispatcher.update(local_functions)

    def get_func_val(self,
                     basis_func_idx,
                     quad_pt,
                     mapping):
        basis_func = self.get_basis_func(basis_func_idx)
        xi = quad_pt[0] ; eta = quad_pt[1]
        f00 = basis_func[0][0](xi, eta) ; f01 = basis_func[0][1](xi, eta)
        f10 = basis_func[1][0](xi, eta) ; f11 = basis_func[1][1](xi, eta)
        return np.array([[f00, f01],[f10, f11]])

    def _set_basis_lst(self):
        if self.get_degree() == 0:
            basis_lst = self.P0_basis_2D.get_basis_func_lst()
            zero_func = lambda x,y : 0.
            self._p0_skewtens_lst = []
            for basis_ele in basis_lst:
                basis_ele_m1 = lambda x,y : -1*basis_ele(x,y)
                self._p0_skewtens_lst.append(np.array([[zero_func, basis_ele_m1],[basis_ele, zero_func]]))

    def get_basis_func(self,idx):
        try:
            basis_func = self._p0_vec_lst[idx]
        except AttributeError:
            self._set_basis_lst()
            basis_func = self._p0_skewtens_lst[idx]
        return basis_func

    def get_name(self):
        return "p0_skewtens_basis"
        
    def get_num_dof(self):
        return self.P0_basis_2D.get_num_dof()

    def get_num_interior_dof(self):
        return self.P0_basis_2D.get_num_interior_dof()

    def get_num_dof_per_edge(self):
        return self.P0_basis_2D.get_num_dof_per_edge()

    def get_num_dof_per_node(self):
        return self.P0_basis_2D.get_num_dof_per_node()

class P0VecBasis_2D(Basis):

    def __init__(self):
        super(P0VecBasis_2D,self).__init__()
        self.set_degree(0)
        self.P0_basis_2D = P0Basis_2D()
        self._set_local_function_dispatcher()

    def _set_local_function_dispatcher(self):
        local_functions = {'vals' : self.get_func_val}
        self._function_dispatcher.update(local_functions)

    def get_func_val(self,
                     basis_func_idx,
                     quad_pt,
                     mapping):
        basis_func = self.get_basis_func(basis_func_idx)
        xi = quad_pt[0] ; eta = quad_pt[1]
        f1 = basis_func[0](xi, eta) ; f2 = basis_func[1](xi, eta)
        return np.array([f1, f2])
        
    def _set_basis_lst(self):
        if self.get_degree() == 0:
            basis_lst = self.P0_basis_2D.get_basis_func_lst()
            zero_fun = lambda x,y : 0.
            self._p0_vec_lst = []
            for basis_ele in basis_lst:
                self._p0_vec_lst.append(np.array([basis_ele,zero_fun]))
                self._p0_vec_lst.append(np.array([zero_fun,basis_ele]))

    def get_basis_func(self,idx):
        try:
            basis_func = self._p0_vec_lst[idx]
        except AttributeError:
            self._set_basis_lst()
            basis_func = self._p0_vec_lst[idx]
        return basis_func

    def get_name(self):
        return "p0_vec_basis"
        
    def get_num_dof(self):
        return 2*self.P0_basis_2D.get_num_dof()

    def get_num_interior_dof(self):
        return 2*self.P0_basis_2D.get_num_interior_dof()

    def get_num_dof_per_edge(self):
        return 2*self.P0_basis_2D.get_num_dof_per_edge()

    def get_num_dof_per_node(self):
        return 2*self.P0_basis_2D.get_num_dof_per_node()
    
class P1Basis_2D(Basis):

    def __init__(self):
        super(P1Basis_2D,self).__init__()
        self.set_degree(1)
        self._initialize_element_functions()
        self._set_local_function_dispatcher()

    def _set_local_function_dispatcher(self):
        local_functions = {'vals' : self.get_func_val,
                           'dvals' : self.get_func_dval}
        self._function_dispatcher.update(local_functions)

    def get_func_val(self,
                     basis_func_idx,
                     quad_pt,
                     mapping):
        basis_func = self.get_basis_func(basis_func_idx)
        xi = quad_pt.vals[0] ; eta = quad_pt.vals[1]
        return basis_func(xi, eta)

    def get_func_dval(self,
                      basis_func_idx,
                      quad_pt,
                      mapping):
        basis_func = self.get_basis_dfunc(basis_func_idx)
        xi = quad_pt.vals[0] ; eta = quad_pt.vals[1]
        xi = basis_func[0](xi,eta) ; eta = basis_func[1](xi,eta)
        x, y = mapping.apply_inv_transpose_jacobian_mat(xi,eta)
        return x, y

    def get_basis_func(self, idx):
        return self._basis_funcs[idx]

    def get_basis_dfunc(self,idx):
        return self._basis_funcs_grad[idx]
        
    def get_name(self):
        return "p1_basis"            

    def get_num_dof(self):
        return 3

    def get_num_interior_dof(self):
        return 0

    def get_num_dof_per_edge(self):
        return 0

    def get_num_dof_per_node(self):
        return 1

    def _initialize_element_functions(self):
        self._basis_funcs = [lambda xi, eta: 1 - xi - eta,
                             lambda xi, eta: xi,
                             lambda xi, eta: eta]

        self._basis_funcs_grad = [[lambda xi, eta: -1., lambda xi, eta: -1.],
                                  [lambda xi, eta: 1.,  lambda xi, eta: 0.],
                                  [lambda xi, eta: 0.,  lambda xi, eta: 1.]]
                                  
    def get_func(self,dof_num):
        return self._basis_funcs[dof_num]

    def get_func_grad(self,dof_num):
        return self.basis_funcs_grad[dof_num]
    
class BDMTensBasis(Basis):

    def __init__(self,k):
        """ Initialize a tensor BDM basis with degree k

        input
        -----
        k : BDM degree
        """
        super(BDMTensBasis,self).__init__()
        self.set_degree(k)
        self.bdm_basis = BDMBasis(k)
        self._set_local_function_dispatcher()

    def _set_local_function_dispatcher(self):
        local_functions = {'vals' : self.get_func_vals,
                           'Jt_vals' : self.get_Jt_func_vals,
                           'div': self.get_div_vals}
        self._function_dispatcher.update(local_functions)        

    def get_func_vals(self,
                      basis_func_idx,
                      quad_pt,
                      mapping):
        basis_func = self.get_basis_func(basis_func_idx)
        xi = quad_pt[0] ; eta = quad_pt[1]
        v00 = basis_func[0][0](xi, eta) ; v01 = basis_func[0][1](xi, eta)
        v10 = basis_func[1][0](xi, eta) ; v11 = basis_func[1][1](xi, eta)
        return np.array([[v00, v01],[v10, v11]])

    def get_Jt_func_vals(self,
                         basis_func_idx,
                         quad_pt,
                         mapping):
        tens_vals = self.get_func_vals(basis_func_idx,
                                       quad_pt,
                                       mapping)
        row_1 = mapping.apply_jacobian_mat(tens_vals[0][0], tens_vals[0][1])
        row_2 = mapping.apply_jacobian_mat(tens_vals[1][0], tens_vals[1][1])
        return np.array([row_1, row_2])

    def get_div_vals(self,
                     basis_func_idx,
                     quad_pt,
                     mapping):
        basis_div_func = self.get_basis_div_func(basis_func_idx)
        xi = quad_pt.vals[0] ; eta = quad_pt.vals[1]
        f1 = basis_div_func[0](xi,eta) ; f2 = basis_div_func[1](xi,eta)
        return np.array([f1,f2])
        
    def get_name(self):
        return "bdm_tens_basis"

    def get_num_dof(self):
        return 2*self.bdm_basis.get_num_dof()

    def get_num_interior_dof(self):
        return 2*self.bdm_basis.get_num_interior_dof()

    def get_num_dof_per_edge(self):
        return 2*self.bdm_basis.get_num_dof_per_edge()

    def get_num_edge_dof(self):
        return 2*self.bdm_basis.get_num_edge_dof()

    def get_num_dof_per_node(self):
        return 2*self.bdm_basis.get_num_dof_per_node()

    def _set_basis_lst(self):
        if self.get_degree() == 1:
            basis_lst = self.bdm_basis.get_basis_func_lst()            
            self._bdm_tens_lst = [None]*2*len(basis_lst)
            zero_fun = lambda x,y : 0.
            for i,basis_ele in enumerate(basis_lst):
                j = i / 3
                k = i % 3
                a_tmp = [basis_ele[0], basis_ele[1]]
                b_tmp = [zero_fun, zero_fun]
                self._bdm_tens_lst[6*j + k] = np.array([a_tmp,b_tmp])
                self._bdm_tens_lst[6*j + k + 3] = (np.array([b_tmp,a_tmp]))

    def _set_basis_div_lst(self):
        if self.get_degree() == 1:
            basis_div_lst = self.bdm_basis.get_basis_div_func_lst()
            self._basis_div_lst = [None]*2*len(basis_div_lst)
            zero_func = lambda x,y : 0.
            for i,basis_div_func in enumerate(basis_div_lst):
                j = i / 3
                k = i % 3
                self._basis_div_lst[6*j + k] = np.array([basis_div_func , zero_func])
                self._basis_div_lst[6*j + k + 3] = np.array([zero_func , basis_div_func])

    def get_basis_func(self,idx):
        try:
            basis_func = self._bdm_tens_lst[idx]
        except AttributeError:
            self._set_basis_lst()
            basis_func = self._bdm_tens_lst[idx]
        return basis_func

    def get_basis_div_func(self,idx):
        try:
            basis_func = self._basis_div_lst[idx]
        except AttributeError:
            self._set_basis_div_lst()
            basis_func = self._basis_div_lst[idx]
        return basis_func

class BDMBasis(Basis):

    def __init__(self,k):
        """  Initialize a BDM basis of degree k.

        input
        -----
        k : BDM degree
        """
        super(BDMBasis,self).__init__()
        self.set_degree(k)
        self.edge_functions = BDMEdgeFuncs(self._k)
        self.interior_functions = BDMIntFuncs(self._k)
        self._set_local_function_dispatcher()

    def _set_local_function_dispatcher(self):
        local_functions = {'vals' : self.get_func_vals,
                           'Jt_vals' : self.get_Jt_func_vals,
                           'div': self.get_div_vals}
        self._function_dispatcher.update(local_functions)

    def _set_basis_lst(self):
        if self.get_degree() == 1:
            edge_func = self.edge_functions.edge_funcs
            self._basis_lst = [edge_func[0][0],
                               edge_func[1][0],
                               edge_func[2][0],
                               edge_func[0][1],
                               edge_func[1][1],
                               edge_func[2][1]]

        if self.get_degree() == 2:
            edge_func = self.edge_functions.edge_funcs
            int_func = self.interior_functions.int_funcs
            self._basis_lst = [edge_func[0][0],
                               edge_func[1][0],
                               edge_func[2][0],
                               edge_func[0][1],
                               edge_func[1][1],
                               edge_func[2][1],
                               edge_func[0][2],
                               edge_func[1][2],
                               edge_func[2][2],
                               int_func[0],
                               int_func[1],
                               int_func[2]]

    def _set_basis_div_lst(self):
        if self.get_degree() == 1:
            edge_div_func = self.edge_functions.edge_div_funcs
            self._basis_div_lst = [edge_div_func[0][0],
                                   edge_div_func[1][0],
                                   edge_div_func[2][0],
                                   edge_div_func[0][1],
                                   edge_div_func[1][1],
                                   edge_div_func[2][1]]

    def get_name(self):
        return "bdm_basis"

    def get_num_dof(self):
        if self.get_degree()==1:
            return 6
        elif self.get_degree()==2:
            return 12

    def get_num_interior_dof(self):
        if self.get_degree()==1:
            return 0
        elif self.get_degree()==2:
            return 3

    def get_num_dof_per_edge(self):
        if self.get_degree()==1:
            return 2
        elif self.get_degree()==2:
            return 3

    def get_num_edge_dof(self):
        if self.get_degree()==1:
            return 6
        if self.get_degree()==2:
            return 9

    def get_num_dof_per_node(self):
        if self.get_degree()==1:
            return 0
        elif self.get_degree()==2:
            return 0

    def get_basis_func(self,idx):
        try :
            basis_func = self._basis_lst[idx]
        except AttributeError:
            self._set_basis_lst()
            basis_func = self._basis_lst[idx]
        return basis_func

    def get_basis_func_lst(self):
        try :
            basis_func = self._basis_lst
        except AttributeError:
            self._set_basis_lst()
            basis_func = self._basis_lst
        return basis_func

    def get_basis_div_func(self,idx):
        try :
            basis_func = self._basis_div_lst[idx]
        except AttributeError:
            self._set_basis_div_lst()
            basis_func = self._basis_div_lst[idx]
        return basis_func

    def get_basis_div_func_lst(self):
        try :
            basis_func = self._basis_div_lst
        except AttributeError:
            self._set_basis_div_lst()
            basis_func = self._basis_div_lst
        return basis_func
    

    def get_func_vals(self,
                      basis_func_idx,
                      quad_pt,
                      mapping):
        basis_func = self.get_basis_func(basis_func_idx)
        xi = quad_pt[0] ; eta = quad_pt[1]
        v0 = basis_func[0](xi,eta)
        v1 = basis_func[1](xi,eta)
        return np.array([v0,v1])

    def get_Jt_func_vals(self,
                        basis_func_idx,
                        quad_pt,
                        mapping):
        val_pair = self.get_func_vals(basis_func_idx,
                                      quad_pt,
                                      mapping)
        xi_val = val_pair[0] ; eta_val = val_pair[1]
        return mapping.apply_jacobian_mat(xi_val,eta_val)

    def get_div_vals(self,
                     basis_func_idx,
                     quad_pt,
                     mapping):
        basis_div_func = self.get_basis_div_func(basis_func_idx)
        xi = quad_pt.vals[0] ; eta = quad_pt.vals[1]
        return basis_div_func(xi,eta)

    def get_edge_piola_normal_func(self,
                                   edge_num,
                                   basis_edge=None,
                                   basis_edge_idx=None,
                                   dof_num=None):
        pass


    def get_edge_normal_func(self,
                             edge_num,
                             basis_edge=None,
                             basis_edge_idx=None,
                             dof_num=None):
        """
        This function returns the normal function of a basis element
        along an edge on the reference element.

        Arguments
        ---------
        edge_num : int
            This is edge along which the functions normal trace is
            taken.

        basis_edge : int
            This is the edge along which the basis element is
            built.

        basis_edge_idx : int
            This is the basis function's index number within the set
            of basis functions built on basis_edge.

        dof_num : int
            Builds the normal component using the dof number.

        Notes
        -----
        This function and approach should be refactored, but a clearer
        approach to organizing this class is required first.
        """
        n = self.edge_functions.reference_element.get_normal_vec(edge_num)
        if type(basis_edge) is int and basis_edge_idx is not None:
            func = self.get_edge_func(basis_edge=basis_edge,
                                      basis_edge_idx=basis_edge_idx)
        if type(basis_edge) is str and basis_edge_idx is not None:
            func = self.get_int_func(basis_edge_idx)
        elif dof_num is not None:
            func = self.get_edge_func(dof_num=dof_num)
        return lambda xi, eta: n[0]*func[0](xi,eta) + n[1]*func[1](xi,eta)

    def get_int_func(self,
                     dof_num):
        return self.interior_functions.int_funcs[dof_num]

    def get_edge_func(self,
                      basis_edge=None,
                      basis_edge_idx=None,
                      dof_num=None):
        if basis_edge is not None and basis_edge_idx is not None:
            return self.edge_functions.edge_funcs[basis_edge][basis_edge_idx]
        elif dof_num is not None:
            basis_edge = dof_num % 3
            basis_edge_idx = dof_num / 3
            return self.edge_functions.edge_funcs[basis_edge][basis_edge_idx]

class BDMSubFuncs(object):

    def __init__(self, k):
        self.set_degree(k)
        self.quadrature = quad.Quadrature(self.get_degree() + 1)
        self.reference_element = mt.ReferenceElement()

    def set_degree(self,k):
        self.k = k

    def get_degree(self):
        return self.k

class BDMIntFuncs(BDMSubFuncs):

    def __init__(self, k):
        super(BDMIntFuncs, self).__init__(k)
        self._initialize_interior_functions()

    def _initialize_interior_functions(self):
        self.int_funcs = []
        if self.get_degree() == 1:
            # No interior functions for BDM1
            pass
        if self.get_degree() == 2:
            q0 = self.quadrature.edge_quad_pt[0]
            q1 = self.quadrature.edge_quad_pt[1]

            f11 = lambda xi, eta: (1-xi-eta) * BDMEdgeFuncs._edge_func_one(q0,q1)[0](xi, eta)
            f12 = lambda xi, eta: (1-xi-eta) * BDMEdgeFuncs._edge_func_one(q0,q1)[1](xi, eta)
            self.int_funcs.append( np.array([f11, f12]) )

            f21 = lambda xi, eta: xi * BDMEdgeFuncs._edge_func_two(q0,q1)[0](xi, eta)
            f22 = lambda xi, eta: xi * BDMEdgeFuncs._edge_func_two(q0,q1)[1](xi, eta)
            self.int_funcs.append( np.array([f21, f22]) )

            f31 = lambda xi, eta: eta * BDMEdgeFuncs._edge_func_three(q0,q1)[0](xi, eta)
            f32 = lambda xi, eta: eta * BDMEdgeFuncs._edge_func_three(q0,q1)[1](xi, eta)
            self.int_funcs.append( np.array([f31, f32]) )

class BDMEdgeFuncs(BDMSubFuncs):

    def __init__(self, k):
        super(BDMEdgeFuncs, self).__init__(k)
        self._initialize_edge_functions()

    @staticmethod
    def _edge_func_one(q1,q2):
        sqrt_2 = math.sqrt(2)
        const = sqrt_2 / (q2- q1)
        return np.array([lambda xi, eta: const * q2 * xi ,
                         lambda xi, eta: const * (q2-1)* eta])

    @staticmethod
    def _edge_func_two(q1,q2):
        const = 1.0 / (q2- q1)
        return np.array([lambda xi, eta: const * (q2*xi + eta - q2) ,
                         lambda xi, eta: const * (q2-1)*eta])

    @staticmethod
    def _edge_func_three(q1,q2):
        const = 1.0 / (q2- q1)
        return np.array([lambda xi, eta: const * (q2-1) * xi ,
                         lambda xi, eta: const * (xi + q2*eta - q2) ])

    def _initialize_edge_functions(self):
        self._get_edge_vec_functions()

    def _get_edge_vec_functions(self):
        """ Create BDM edge functions. """
        self.edge_funcs = [[],[],[]]
        self.edge_div_funcs = [[],[],[]]
        if self.k==1:
            q0 = self.quadrature.edge_quad_pt[0]
            q1 = self.quadrature.edge_quad_pt[1]

            self.edge_funcs[0].append(BDMEdgeFuncs._edge_func_one(q0,q1))
            self.edge_div_funcs[0].append(self._edge_func_div_one(q0,q1))
            self.edge_funcs[0].append(BDMEdgeFuncs._edge_func_one(q1,q0))
            self.edge_div_funcs[0].append(self._edge_func_div_one(q1,q0))

            self.edge_funcs[1].append(BDMEdgeFuncs._edge_func_two(q0,q1))
            self.edge_div_funcs[1].append(self._edge_func_div_two(q0,q1))
            self.edge_funcs[1].append(BDMEdgeFuncs._edge_func_two(q1,q0))
            self.edge_div_funcs[1].append(self._edge_func_div_two(q1,q0))

            self.edge_funcs[2].append(BDMEdgeFuncs._edge_func_three(q0,q1))
            self.edge_div_funcs[2].append(self._edge_func_div_three(q0,q1))
            self.edge_funcs[2].append(BDMEdgeFuncs._edge_func_three(q1,q0))
            self.edge_div_funcs[2].append(self._edge_func_div_three(q1,q0))

        if self.k==2:
            q0 = self.quadrature.edge_quad_pt[0]
            q1 = self.quadrature.edge_quad_pt[1]
            q2 = self.quadrature.edge_quad_pt[2]

            edge_one_funcs = self._get_edge_func_one(q0,q1,q2)
            for func in edge_one_funcs:
                self.edge_funcs[0].append(func)

            edge_two_funcs = self._get_edge_func_two(q0,q1,q2)
            for func in edge_two_funcs:
                self.edge_funcs[1].append(func)

            edge_three_funcs = self._get_edge_func_three(q0,q1,q2)
            for func in edge_three_funcs:
                self.edge_funcs[2].append(func)

    def _edge_func_div_one(self,q1,q2):
        sqrt_2 = math.sqrt(2)
        const = sqrt_2 / (q2-q1)
        div_f = lambda xi, eta: const*(q2 + (q2-1))
        return div_f

    def _edge_func_div_two(self,q1,q2):
        const = 1.0 / (q2 - q1)
        div_f = lambda xi, eta: const*(q2 + (q2-1))
        return div_f

    def _edge_func_div_three(self,q1,q2):
        const = 1.0 / (q2-q1)
        div_f = lambda xi, eta: const*((q2-1) + q2)
        return div_f

    def _lagrange_edge_one(self,p1,p2):
        return lambda xi, eta: (eta-p2) / (p1-p2)

    def _lagrange_edge_two(self,p1,p2):
        return lambda xi, eta: (eta-p2) / (p1-p2)

    def _lagrange_edge_three(self,p1,p2):
        return lambda xi, eta: (xi-p2) / (p1-p2)

    def _get_edge_func_one(self,q0,q1,q2):
        quad_pair_lst = [((q0,q1) , (q0,q2)),
                         ((q1,q2) , (q1,q0)),
                         ((q2,q0) , (q2,q1))]
        edge_one_func = []

        _edge_func_1 = BDMEdgeFuncs._edge_func_one(quad_pair_lst[0][0][0],
                                                   quad_pair_lst[0][0][1])
        _lagrange_1 = self._lagrange_edge_one(quad_pair_lst[0][1][0],
                                              quad_pair_lst[0][1][1])
        f1_1 = lambda xi, eta: _lagrange_1(xi,eta)*_edge_func_1[0](xi,eta)
        f2_1 = lambda xi, eta: _lagrange_1(xi,eta)*_edge_func_1[1](xi,eta)
        edge_one_func.append(np.array([f1_1,f2_1]))

        _edge_func_2 = BDMEdgeFuncs._edge_func_one(quad_pair_lst[1][0][0],
                                                   quad_pair_lst[1][0][1])
        _lagrange_2 = self._lagrange_edge_one(quad_pair_lst[1][1][0],
                                              quad_pair_lst[1][1][1])
        f1_2 = lambda xi, eta: _lagrange_2(xi,eta)*_edge_func_2[0](xi,eta)
        f2_2 = lambda xi, eta: _lagrange_2(xi,eta)*_edge_func_2[1](xi,eta)
        edge_one_func.append(np.array([f1_2,f2_2]))

        _edge_func_3 = BDMEdgeFuncs._edge_func_one(quad_pair_lst[2][0][0],
                                                   quad_pair_lst[2][0][1])
        _lagrange_3 = self._lagrange_edge_one(quad_pair_lst[2][1][0],
                                              quad_pair_lst[2][1][1])
        f1_3 = lambda xi, eta: _lagrange_3(xi,eta)*_edge_func_3[0](xi,eta)
        f2_3 = lambda xi, eta: _lagrange_3(xi,eta)*_edge_func_3[1](xi,eta)
        edge_one_func.append(np.array([f1_3,f2_3]))

        return edge_one_func

    def _get_edge_func_two(self,q0,q1,q2):
        quad_pair_lst = [((q0,q1) , (q0,q2)),
                         ((q1,q2) , (q1,q0)),
                         ((q2,q0) , (q2,q1))]
        edge_two_func = []

        _edge_func_1 = BDMEdgeFuncs._edge_func_two(quad_pair_lst[0][0][0],
                                                   quad_pair_lst[0][0][1])
        _lagrange_1 = self._lagrange_edge_two(quad_pair_lst[0][1][0],
                                              quad_pair_lst[0][1][1])
        f1_1 = lambda xi, eta: _lagrange_1(xi,eta)*_edge_func_1[0](xi,eta)
        f2_1 = lambda xi, eta: _lagrange_1(xi,eta)*_edge_func_1[1](xi,eta)
        edge_two_func.append(np.array([f1_1,f2_1]))

        _edge_func_2 = BDMEdgeFuncs._edge_func_two(quad_pair_lst[1][0][0],
                                                   quad_pair_lst[1][0][1])
        _lagrange_2 = self._lagrange_edge_two(quad_pair_lst[1][1][0],
                                              quad_pair_lst[1][1][1])
        f1_2 = lambda xi, eta: _lagrange_2(xi,eta)*_edge_func_2[0](xi,eta)
        f2_2 = lambda xi, eta: _lagrange_2(xi,eta)*_edge_func_2[1](xi,eta)
        edge_two_func.append(np.array([f1_2,f2_2]))

        _edge_func_3 = BDMEdgeFuncs._edge_func_two(quad_pair_lst[2][0][0],
                                                   quad_pair_lst[2][0][1])
        _lagrange_3 = self._lagrange_edge_two(quad_pair_lst[2][1][0],
                                              quad_pair_lst[2][1][1])
        f1_3 = lambda xi, eta: _lagrange_3(xi,eta)*_edge_func_3[0](xi,eta)
        f2_3 = lambda xi, eta: _lagrange_3(xi,eta)*_edge_func_3[1](xi,eta)
        edge_two_func.append(np.array([f1_3,f2_3]))

        return edge_two_func

    def _get_edge_func_three(self,q0,q1,q2):
        quad_pair_lst = [((q0,q1) , (q0,q2)),
                         ((q1,q2) , (q1,q0)),
                         ((q2,q0) , (q2,q1))]
        edge_three_func = []

        _edge_func_1 = BDMEdgeFuncs._edge_func_three(quad_pair_lst[0][0][0],
                                                     quad_pair_lst[0][0][1])
        _lagrange_1 = self._lagrange_edge_three(quad_pair_lst[0][1][0],
                                                quad_pair_lst[0][1][1])
        f1_1 = lambda xi, eta: _lagrange_1(xi,eta)*_edge_func_1[0](xi,eta)
        f2_1 = lambda xi, eta: _lagrange_1(xi,eta)*_edge_func_1[1](xi,eta)
        edge_three_func.append(np.array([f1_1,f2_1]))

        _edge_func_2 = BDMEdgeFuncs._edge_func_three(quad_pair_lst[1][0][0],
                                                     quad_pair_lst[1][0][1])
        _lagrange_2 = self._lagrange_edge_three(quad_pair_lst[1][1][0],
                                                quad_pair_lst[1][1][1])
        f1_2 = lambda xi, eta: _lagrange_2(xi,eta)*_edge_func_2[0](xi,eta)
        f2_2 = lambda xi, eta: _lagrange_2(xi,eta)*_edge_func_2[1](xi,eta)
        edge_three_func.append(np.array([f1_2,f2_2]))

        _edge_func_3 = BDMEdgeFuncs._edge_func_three(quad_pair_lst[2][0][0],
                                                     quad_pair_lst[2][0][1])
        _lagrange_3 = self._lagrange_edge_three(quad_pair_lst[2][1][0],
                                                quad_pair_lst[2][1][1])
        f1_3 = lambda xi, eta: _lagrange_3(xi,eta)*_edge_func_3[0](xi,eta)
        f2_3 = lambda xi, eta: _lagrange_3(xi,eta)*_edge_func_3[1](xi,eta)
        edge_three_func.append(np.array([f1_3,f2_3]))

        return edge_three_func
