import math
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse.linalg as sla

class Operators():

    @staticmethod
    def l2_norm(v1):
        return math.sqrt( sum(a**2 for a in v1) )

    @staticmethod
    def dot_product(v1,v2):
        return sum([n1*n2 for n1,n2 in zip(v1,v2)])

    @staticmethod
    def tensor_dot_product(tens1, tens2):
        a = [tens1[0][i]*tens2[0][i]+tens1[1][i]*tens1[1][i] for i in [0,1]]
        return sum(a)

    @staticmethod
    def scalar_product_grad(f1, f2, grad_f1, grad_f2):
        p1 = lambda x,y: grad_f1[0](x,y) * f2(x,y) + grad_f2[0](x,y) * f1(x,y)
        p2 = lambda x,y: grad_f1[1](x,y) * f2(x,y) + grad_f2[1](x,y) * f1(x,y)
        return [p1, p2]

    @staticmethod
    def lam_func_dot_product(f1,f2):
        f_new = lambda x,y : f1[0](x,y)*f2[0](x,y) + f1[1](x,y)*f2[1](x,y) 
        return f_new

    @staticmethod
    def lam_func_product(f1,f2):
        f_new = lambda x,y : f1(x,y)*f2(x,y)
        return f_new

    @staticmethod
    def deviatoric(sig1):
        """ Calculate the deviatoric of a tensor.

        Arguments
        ---------
        sig1 : numpy array

        Returns:
        --------
        deviatoric : numpy array
        """
        n = sig1.shape[0] ; I = np.identity(n)
        I *= 1./n * np.trace(sig1)
        return sig1 - I

    @staticmethod
    def cartesian_elasticity_tens_tens_1(sig_tens,
                                         tau_tens,
                                         mu = 1.0,
                                         lam = 1.0):
        """ Evaluate the elasticity inner product for a(.,.)

        Arguments
        ---------
        sig_tens : numpy array
            trial function vals
        tau_tens : numpy array
            test function vals
        mu : float
            Lame coefficient
        lam : float
            Lame coefficient

        Returns
        -------
        val : 
            Inner product value
        """
        n = sig_tens.shape[0]+1
        sig_tens_dev = Operators.deviatoric(sig_tens)
        tau_tens_dev = Operators.deviatoric(tau_tens)

        dev_product = np.tensordot(sig_tens_dev,tau_tens_dev)
        trace_product = sig_tens.trace() * tau_tens.trace()

        p1 = (1./(2*mu)) * dev_product
        p2 = (1./(n*(n*lam + 2*mu))) * trace_product

        return p1 + p2

    @staticmethod
    def cartesian_elasticity_tens_tens(sig_tens,
                                       tau_tens,
                                       mu = 1.0,
                                       lam = 1.0):
        
        tens_product = np.tensordot(sig_tens, tau_tens)
        trace_product = sig_tens.trace() * tau_tens.trace()

        p1 = (1./(2*mu)) * tens_product
        p2 = lam / (2*mu*(2*mu+3*lam)) * trace_product

        return p1 - p2
    
    @staticmethod
    def weak_symmetry_dot_product(tau_tens,
                                  sig_tens):
        """ Computes the off diagonal product of two tensors
        
        Arguments
        ---------
        tau_tens : numpy array
            test function vals
        sig_tens : numpy array
            trial function vals

        Returns
        -------
        val :

        Notes
        -----
        sig_tens should be axisymmetric
        """
        assert sig_tens[0][0]*tau_tens[0][0] == 0.
        assert sig_tens[1][1]*tau_tens[1][1] == 0.
        return np.tensordot(tau_tens,sig_tens)

    @staticmethod
    def first_row_axi_divergence_scalar_product(tau_tens,
                                                r_val,
                                                scalar):
        """ 
        Computes the divergence of the first row of a tensor
        and a scalar function.

        Arguments
        ---------
        tau_tens : dict
        """
        row_1_div = ( 1./r_val * tau_tens['Jt_vals'][0,0] +
                      tau_tens['div'][0] )
        return row_1_div * scalar
    
class GlobalMatrix():

    def __init__(self,
                 num_rows=None,
                 num_cols=None):
        self.set_size(num_rows,num_cols)
        self._row = []
        self._col = []
        self._val = []

    def add_new_entry(self,row,col,val):
        self._row.append(row)
        self._col.append(col)
        self._val.append(val)

    def set_size(self,num_rows,num_cols):
        self._num_rows = num_rows
        self._num_cols = num_cols

    def get_shape(self):
        return (self._num_rows, self._num_cols)

    def set_sparse_arrays(self):
        self._np_row = np.array(self._row)
        self._np_col = np.array(self._col)
        self._np_val = np.array(self._val)

    def set_coo(self):
        self._coo = coo_matrix((self._np_val,
                                (self._np_row,
                                 self._np_col)),
                                shape=self.get_shape())

    def initialize_sparse_matrix(self):
        self.set_sparse_arrays()
        self.set_coo()

    def set_csr_rep(self):
        self._csr = self._coo.tocsr()

    def get_csr_rep(self):
        return self._csr

    def get_array_rep(self):
        csr_rep = self.get_csr_rep()
        return csr_rep.toarray()
    
    def set_row_as_dirichlet_bdy(self,i):
        self._csr.data[self._csr.indptr[i]:self._csr.indptr[i+1]] = 0.
        d = self._csr.diagonal()
        d[i] = 1.0
        self._csr.setdiag(d)

    def set_average_pressure_val(self,num_pressure_dof):
        num_dof = max(self._row) + 1
        a = np.bincount(self._row)[-1]
        for term in range(-a,0):
            self._val[term] = 0.
        for t in range(num_dof-num_pressure_dof, num_dof):
            self.add_new_entry(num_dof-1, t, 1.0)

    def solve(self,rhs,sol_vec):
        rhs_vec = rhs.get_rhs_vec()
        mat = self.get_csr_rep()
        sol_vec.set_solution(sla.spsolve(mat,rhs_vec))

class LocalMatrixAssembler():

    def __init__(self,
                 dof_handler,
                 num_local_dof,
                 global_matrix_assembler):
        self._dof_handler = dof_handler
        self.set_num_local_dof(num_local_dof)
        self._local_mat = np.zeros(shape=(self.get_num_local_dof(),
                                          self.get_num_local_dof()))
        self._global_matrix_assembler = global_matrix_assembler

    def set_num_local_dof(self,num_dof):
        self._num_local_dof = num_dof

    def get_num_local_dof(self):
        return self._num_local_dof
        
    def add_val(self,i,j,val):
        self._local_mat[i][j] += val

    def reset_matrix(self):
        self._local_mat = np.zeros(shape=(self.get_num_local_dof(),
                                          self.get_num_local_dof() ))

    def distribute_local_2_global(self, eN):
        for i in range(self.get_num_local_dof()):
            row = self._dof_handler.get_local_2_global(eN,i)
            for j in range(self.get_num_local_dof()):
                col = self._dof_handler.get_local_2_global(eN,j)
                val = self._local_mat[i][j]
                self._global_matrix_assembler.add_new_entry(row,col,val)

class GlobalRHS():

    def __init__(self,
                 dof_handler):
        self.set_dof_handler(dof_handler)
        self.set_num_dof(self._dof_handler.get_num_dof())
        self._rhs_vec = np.zeros(shape=(self.get_num_dof(),
                                        1) )

    def set_num_dof(self,num_dof):
        self._num_dof = num_dof

    def set_dof_handler(self,dof_handler):
        self._dof_handler = dof_handler

    def get_dof_handler(self,dof_handler):
        return self._dof_handler

    def get_num_dof(self):
        return self._num_dof

    def get_rhs_vec(self):
        return self._rhs_vec

    def get_val(self,i):
        return self._rhs_vec[i]

    def add_val(self,eN,idx,val):
        row = self._dof_handler.get_local_2_global(eN,idx)
        self._rhs_vec[row] += val
    
    def set_value(self,idx,val):
        self._rhs_vec[idx] = val

    def view(self):
        print(self._rhs_vec)

class DiscreteSolutionVector():

    def __init__(self,num_dof):
        self.set_num_dof(num_dof)

    def set_num_dof(self,num_dof):
        self._num_dof = num_dof

    def set_solution(self,solu_vec):
        self._sol_vec = solu_vec

    def get_num_dof(self):
        return self._num_dof

    def get_solution(self):
        return self._sol_vec

    def get_solution_val(self,idx):
        return self._sol_vec[idx]

    def scale_basis_by_solution(self,
                                idx,
                                basis_vals):
        scaling_val = self.get_solution_val(idx)
        return scaling_val*basis_vals
