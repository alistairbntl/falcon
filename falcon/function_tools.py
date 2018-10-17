import numpy as np

class Function(object):

    def __init__(self,f):
        self._f = f

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

    def get_normal_velocity_func(self, n):
        fx_n1 = lambda x, y : self._f[0](x,y)*n[0]
        fy_n2 = lambda x, y : self._f[1](x,y)*n[1]
        return lambda x, y : fx_n1(x,y) + fy_n2(x,y)    

class TrueSolution(Function):

    def __init__(self, f):
        super(TrueSolution,self).__init__(f)
        self._f = f

class Iterator:
    def __init__(self, lst):
        self.lst = lst
    def __getitem__(self, idx):
        result = self.lst[idx]
        return result
