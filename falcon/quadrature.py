import numpy
from numpy import linalg as LA
import scipy
import math

import falcon
from falcon import mesh_tools as mt

class Quadrature():

    def __init__(self,n):
        """
        input
        -----
        n : int
           Quadrature degree
        """
        self.n = n
        self._get_quadrature_pts()

    def _get_quadrature_pts(self):
        if self.n==1:
            
            self.edge_quad_pt = [0.5 - math.sqrt(3) / 6.,
                                 0.5 + math.sqrt(3) / 6.]
            self.edge_quad_wght = [0.5, 0.5]
            self._element_quad_pts = [mt.QuadraturePoint(1/3.,1/3.,0.5)]

        if self.n==2:

            self.edge_quad_pt = [0.5 - math.sqrt(3) / 6.,
                                 0.5 + math.sqrt(3) / 6.]

            self.edge_quad_wght = [0.5, 0.5]            

            self._element_quad_pts = [mt.QuadraturePoint(0.,1/2., 1/6.),
                                      mt.QuadraturePoint(1/2.,0., 1/6.),
                                      mt.QuadraturePoint(1/2.,1/2., 1/6.)]            

        if self.n==3:
            self.edge_quad_pt = [0.5 - math.sqrt(15)/10.,
                                 0.5,
                                 0.5 + math.sqrt(15)/10.]
            self.edge_quad_wght = [5./18., 8./18., 5./18.]
            self._element_quad_pts = [mt.QuadraturePoint(1./3., 1./3., -27/96.),
                                      mt.QuadraturePoint(1./5., 3./5., 25/96.),
                                      mt.QuadraturePoint(1./5., 1./5., 25/96.),
                                      mt.QuadraturePoint(3./5., 1./5., 25/96.)]

        if self.n==4:
            # ARB - this has not been tested at all!
            self.edge_quad_pt = [0.5 - math.sqrt(15)/10.,
                                 0.5,
                                 0.5 + math.sqrt(15)/10.]
            self.edge_quad_wght = [5./18., 8./18., 5./18.]
            self._element_quad_pts = [mt.QuadraturePoint(0.659027622374092, 0.231933368553031, 1./12.),
                                      mt.QuadraturePoint(0.659027622374092, 0.109039009072877, 1./12.),
                                      mt.QuadraturePoint(0.231933368553031, 0.659027622374092, 1./12.),
                                      mt.QuadraturePoint(0.231933368553031, 0.109039009072877, 1./12.),
                                      mt.QuadraturePoint(0.109039009072877, 0.659027622374092, 1./12.),
                                      mt.QuadraturePoint(0.109039009072877, 0.231933368553031, 1./12.)]
        if self.n==5:
            self.edge_quad_pt = [0.5 - math.sqrt(525 + 70*math.sqrt(30)) / 70.,
                                 0.5 - math.sqrt(525 - 70*math.sqrt(30)) / 70.,
                                 0.5 + math.sqrt(525 - 70*math.sqrt(30)) / 70.,
                                 0.5 + math.sqrt(525 + 70*math.sqrt(30)) / 70.]
            self.edge_quad_wght = [(18 - math.sqrt(30))/72. , (18 + math.sqrt(30))/72,
                                   (18 + math.sqrt(30))/72. , (18 - math.sqrt(30))/72.]
            # self._element_quad_pts = [mt.QuadraturePoint(0.33333333333333,0.33333333333333,0.22500000000000*.5),
            #                           mt.QuadraturePoint(0.47014206410511,0.47014206410511,0.13239415278851*.5),
            #                           mt.QuadraturePoint(0.47014206410511,0.05971587178977,0.13239415278851*.5),
            #                           mt.QuadraturePoint(0.05971587178977,0.47014206410511,0.13239415278851*.5),
            #                           mt.QuadraturePoint(0.10128650732346,0.10128650732346,0.12593918054483*.5),
            #                           mt.QuadraturePoint(0.10128650732346,0.79742698535309,0.12593918054483*.5),
            #                           mt.QuadraturePoint(0.79742698535309,0.10128650732346,0.12593918054483*.5)]

            self._element_quad_pts = [mt.QuadraturePoint(0.24928674517091,0.24928674517091,0.11678627572638*.5),
                                      mt.QuadraturePoint( 0.2492867451709, 0.5014265096581, 0.11678627572638*.5),
                                      mt.QuadraturePoint( 0.5014265096581, 0.2492867451709, 0.11678627572638*.5),
                                      mt.QuadraturePoint( 0.0630890144915, 0.0630890144915, 0.05084490637021*.5),
                                      mt.QuadraturePoint( 0.0630890144915, 0.8738219710170, 0.05084490637021*.5),
                                      mt.QuadraturePoint( 0.8738219710170, 0.0630890144915, 0.05084490637021*.5),
                                      mt.QuadraturePoint( 0.3103524510337, 0.6365024991214, 0.08285107561837*.5),
                                      mt.QuadraturePoint( 0.6365024991214, 0.0531450498448, 0.08285107561837*.5),
                                      mt.QuadraturePoint( 0.0531450498448, 0.3103524510337, 0.08285107561837*.5),
                                      mt.QuadraturePoint( 0.6365024991214, 0.3103524510337, 0.08285107561837*.5),
                                      mt.QuadraturePoint( 0.3103524510337, 0.0531450498448, 0.08285107561837*.5),
                                      mt.QuadraturePoint( 0.0531450498448, 0.6365024991214, 0.08285107561837*.5)]
        else:
            pass

    def _get_quadrature_pts_t(self):
        self.quad_pt_t = []
        for pt in self.quad_pt:
            t = 0.5*(1 - pt[0])
            self._quad_pt_t.append(t)

    def get_element_quad_pts(self):
        return self._element_quad_pts

    def find_quad_on_edge(self, edge):
        pt_lst = []
        for quad_num in range(self.edge_quad_pt):
                pt_lst.append(self.find_one_quad_on_edge(edge,quad_num))
        return pt_lst

    def find_one_quad_on_edge(self, edge, quad_num):
        def calculate_t(a,b,t):
            return a + (b-a)*t

        n0 = edge.get_node(0)
        n1 = edge.get_node(1)
        val = self.edge_quad_pt[quad_num]
        wght = self.edge_quad_wght[quad_num]

        if n0[0] != n1[0]:
            t_val = calculate_t(n0[0],n1[0],val)
            m = (n1[1]-n0[1]) / (n1[0] - n0[0])
            b = n0[1] - m*n0[0]
            y_val = m*t_val + b
            quad_wght = edge.get_edge_length() * wght
            return falcon.mesh_tools.QuadraturePoint(t_val,y_val,quad_wght)
        elif n0[0] == n1[0]:
            t_val = calculate_t(n0[1],n1[1],val)
            x_val = n0[0]
            quad_wght = edge.get_edge_length() * wght
            return falcon.mesh_tools.QuadraturePoint(x_val,t_val,quad_wght)

    def find_quad_on_edges(self,edges):
        """
        input
        -----
        edges : lst
        """
        def calculate_t(a,b,t):
            return 0.5*((a+b) + (b-a)*t)

        def calculate_edge_length(a,b):
            return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

        self.quad_pts_edges = []
        self.quad_wgts_edges = []
        
        for i,edge in enumerate(edges):
            quad_pts = []
            quad_wgts = []
            x0 = edge[0][0]
            x1 = edge[1][0]
            y0 = edge[0][1]
            y1 = edge[1][1]
            if x0 != x1:
                for j in range(self.n):
                    x_val = calculate_t(x0,x1,self.edge_quad_pt[j][0])
                    m = (y1-y0)/(x1-x0)
                    b = edge[0][1] - m*edge[0][0]
                    y_val = m*x_val + b
                    quad_pts.append((x_val,y_val))                    
                    quad_wgts.append(calculate_edge_length(edge[0],edge[1])/2.*self.edge_quad_pt[j][1])
            elif x0 == x1:
                for j in range(self.n):
                    y_val = calculate_t(y0,y1,self.edge_quad_pt[j][0])
                    x_val = x0
                    quad_pts.append((x_val,y_val))
                    quad_wgts.append(calculate_edge_length(edge[0],edge[1])/2.*self.quad_pt[j][1])
            self.quad_pts_edges.append(quad_pts)
            self.quad_wgts_edges.append(quad_wgts)

    def get_quad_pts_on_element(self,mapping):
        """ Finds the quadrature values on a non-reference element

        Parameters
        ----------
        mapping : ReferenceElementMap

        Returns
        -------
        quad_pts : lst
            List of quadrature points on non-reference element
        """
        self.quad_pts = []
        for pt in self._element_quad_pts:
            self.quad_pts.append(self.get_quad_on_element(mapping,pt))
        return self.quad_pts

    @staticmethod
    def get_quad_on_element(mapping,
                            quad_pt):
        """ Finds a quad value on a non-reference element

        Parameters
        ----------
        mapping : ReferenceElementMap
        quad_pt : QuadraturePoint
            Quadrature point on reference triangle

        Returns
        -------
        quad_pt : QuadraturePoint
            Quadrature point on mapping element
        """
        x_val, y_val = mapping.apply_affine_map(quad_pt[0], quad_pt[1])
        wght = mapping.get_jacobian_det()*quad_pt.get_quad_weight()
        return falcon.mesh_tools.QuadraturePoint(x_val,y_val,wght)
        
