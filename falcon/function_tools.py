import numpy as np

class Operators():

    @staticmethod
    def create_dot_product_func(f1,f2):
        f_new = lambda x,y : f1[0](x,y)*f2[0](x,y) + f1[1](x,y)*f2[1](x,y) 
        return f_new

    @staticmethod
    def create_product_func(f1,f2):
        f_new = lambda x,y : f1(x,y)*f2(x,y)
        return f_new

    @staticmethod
    def create_sum_func(f1,f2):
        f_new = lambda x,y : f1(x,y) + f2(x,y)
        return f_new
        
    @staticmethod
    def create_scalar_vector_div_func(s1, grad_s1, v1, div_v1):
        div_p1 = Operators.create_dot_product_func(v1, grad_s1)
        div_p2 = Operators.create_product_func(s1, div_v1)
        return Operators.create_sum_func(div_p1, div_p2)

class Function(object):

    def __init__(self, f, div_f=None):
        self._f = f
        self._div_f = div_f

    def get_f_eval(self, quad_pt):
        vals = []
        for fun in self._f:
            vals.append(fun(quad_pt[0],quad_pt[1]))
        return vals

    def get_f_vel_eval(self, quad_pt):
        vals = np.zeros(2)
        vals[0] = self._f[0](quad_pt[0],quad_pt[1])
        vals[1] = self._f[1](quad_pt[0],quad_pt[1])
        return vals

    def get_p_eval(self, quad_pt):
        vals = self._f[2](quad_pt[0],quad_pt[1])
        return vals

    def get_normal_velocity_func(self, n):
        fx_n1 = lambda x, y : self._f[0](x,y)*n[0]
        fy_n2 = lambda x, y : self._f[1](x,y)*n[1]
        return lambda x, y : fx_n1(x,y) + fy_n2(x,y)

    def get_div_f_vel_eval(self, quad_pt):
        div_vals = []
        try:
            for fun in self._div_f:
                div_vals.append(fun(quad_pt[0],quad_pt[1]))
            return div_vals
        except AttributeError:
            assert 1 == 0

class TrueSolution(Function):

    def __init__(self, f, div_f = None):
        super(TrueSolution,self).__init__(f,
                                          div_f = div_f)
        self._f = f

class Iterator:
    def __init__(self, lst):
        self.lst = lst
    def __getitem__(self, idx):
        result = self.lst[idx]
        return result
