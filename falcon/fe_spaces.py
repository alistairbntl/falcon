class LinearBasisFunctions1D:
    """ Linear basis functions on the unit line. """
    def __init__(self):
        self._createFunctions()

    def _createFunctions(self):
        """ Initialize basis functions """
        self._phi0 = lambda x : 1-x
        self._phi1 = lambda x : x
        self.func_list = (self._phi0,
                          self._phi1)

class LinearBasisFunctions2D():
    """ Linear basis functions on the reference triangle. """
    def __init__(self):
        self._createFunctions()
        self._createFunctionGradients()
        self._createVectorFunctions()

    def getSize(self):
        return len(self.func_list)
        
    def _createFunctions(self):
        """ Initialize basis functions. """
        self._phi0 = lambda xi,eta : 1 - xi - eta
        self._phi1 = lambda xi,eta : xi
        self._phi2 = lambda xi,eta : eta
        self.func_list = (self._phi0,
                          self._phi1,
                          self._phi2)

    def _createFunctionGradients(self):
        """ Initialize the function gradients """        
        pass
        #The following code may not be appropriate for cylindrical coordinates
        # one_func  = lambda x,y : 1.
        # mone_func = lambda x,y : -1.
        # zero_func = lambda x,y : 0
        # self._dphi0 = (mone_func, mone_func)
        # self._dphi1 = (one_func , zero_func) 
        # self._dphi2 = (zero_func, one_func)
        # self.funcGrad_list = (self._dphi0,
        #                       self._dphi1,
        #                       self._dphi2)

    def _createVectorFunctions(self):
        """Create linear vector functions. """
        zero_func = lambda x,y : 0
        self.func_list_2D = ((self._phi0,zero_func),
                             (zero_func,self._phi0),
                             (self._phi1,zero_func),
                             (zero_func,self._phi1),
                             (self._phi2,zero_func),
                             (zero_func,self._phi2))

    def _createTensorFunctions(self):
        pass


class BDM1:
    """  BDM1 Function Space 

    Arguments
    ---------
    g1 : float
        The first of two Gaussian quadrature points used to build BDM1
    g2 : float
        The second of two Gaussian quadrature points for building BDM1
    """
    def __init__(self,g1,g2):
        self._createFunctions(g1,g2)

    def _createFunctions(self,g1,g2):
        def e_1(s1,s2):
            return lambda x,y : math.sqrt(2)/(s2-s1)*numpy.array([[s2*x],
                                                                  [(s2-1)*y]])
        def e_2(s1,s2):
            return lambda x,y : (1./(s2-s1))*numpy.array([[s2*x+y-s2],
                                                          [(s2-1)*y]])
        def e_3(s1,s2):
            return lambda x,y : (1./(s2-s1))*numpy.array([[(s2-1.)*x],
                                                          [x+s2*y-s2]])
        phi_11 = e_1(g1,g2)
        phi_12 = e_1(g2,g1)
        phi_21 = e_2(g2,g1)
        phi_22 = e_2(g1,g2)
        phi_31 = e_3(g1,g2)
        phi_32 = e_3(g2,g1)
        self.func_list = (phi_11,phi_12,
                          phi_21,phi_22,
                          phi_31,phi_32)

    def normal_component(self,c,pt,n):
        """ Return the normal component of a basis function at pt
        
        Arguments
        ---------
        c : int
            Index of the basis element.

        pt : tuple
            A point on the element boundary.

        n : tuple
            The unit normal vector.

        Return
        ------
        val : scalar
            The normal component of the vector.
        """
        f = self.func_list[c]
        fx = f(pt[0],pt[1])[0]
        fy = f(pt[0],pt[1])[1]
        return n[0]*fx + n[1]*fy
        

    def buildFunctionFromBasis(self,coeff):
        """ Build a function as a linear combination of basis functions
        
        Arguments
        ---------
        coeff : lst
            List of coefficients for building function.
        """
        def func(coeff):
            pass
        pass

class BDM2Functions:
    """ BDM2 Function Space
    
    Arguments
    ---------
    g1 : float
        The first of three Gaussian quadrature points used to build BDM2
    g2 : float
        The second Gaussian quadrature point used to build BDM2
    g3 : float
        The third Gaussian quadrature points used to build BDM2
    """
    def __init__(self,g1,g2,g3):
        self._createFunctions(g1,g2,g3)

    def _createFunctions(self,g1,g2,g3):
        pass
