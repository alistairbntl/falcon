import math
import os
import numpy as np

import quadrature as quad
import linalg_tools as la
import function_tools as ft

class Element(object):

    def __init__(self,points):
        self._points = points
        self._set_edges()

    def _set_edges(self):
        self._edges = (Edge(self._points[1],self._points[2]),
                       Edge(self._points[2],self._points[0]),
                       Edge(self._points[0],self._points[1]))

    def get_edge_length(self,i):
        return self._edges[i].get_edge_length()

    def get_edges(self):
        return self._edges

    def get_num_edges(self):
        return len(self.get_edges())

    def get_edge(self,i):
        return self._edges[i]

    def get_point(self,i):
        return self._points[i]

    def get_points(self):
        return self._points

    def get_area(self):
        p0 = self.get_point(0)
        p1 = self.get_point(1)
        p2 = self.get_point(2)
        det = (p0[0]-p2[0])*(p1[1]-p0[1])-(p0[0]-p1[0])*(p2[1]-p0[1])
        return 0.5*abs(det)

    def print_coordinates(self):
        for i, point in enumerate(self._points):
            print "point " + `i` + ": " + '('+ `point.vals[0]` + ' , ' + `point.vals[1]`+ ')'

    def get_center(self):
        x_bar = sum([self.get_point(0).vals[0],
                     self.get_point(1).vals[0],
                     self.get_point(2).vals[0]]) / 3.
        y_bar = sum([self.get_point(0).vals[1],
                    self.get_point(1).vals[1],
                     self.get_point(2).vals[1]]) / 3.
        return (x_bar, y_bar)
        

class MeshElement(Element):

    def __init__(self,
                 mesh_nodes,
                 mesh_edges,
                 bdy_flag,
                 global_idx):
        super(MeshElement,self).__init__(mesh_nodes)
        self._set_element_edges(mesh_edges)
        self._set_global_idx(global_idx)
        self._set_bdy_flag(bdy_flag)

    def _set_global_idx(self,global_idx):
        self._global_idx = global_idx

    def _set_element_edges(self, mesh_edges):
        self._edges = mesh_edges

    def _set_bdy_flag(self, bdy_flag):
        self._bdy_flag = bdy_flag

    def is_bdy_element(self):
        return self._bdy_flag
        
    def get_global_idx(self):
        return self._global_idx

    def get_node(self,i):
        return self._points[i]

    def get_nodes(self):
        return self._points

    def get_num_nodes(self):
        return len(self.get_nodes())

    def get_non_edge_node(self,local_edge_idx):
        """ Return the node not associated with an edge

        Arguments
        ---------
        local_edge_idx : int
            Index of the local edge

        Returns
        -------
        node : `node`
            The node that does not lie on current edge
        """
        edge = self.get_edge(local_edge_idx)
        nodes = edge.get_nodes()
        node_flag = [nodes[0] == node or nodes[1] == node for node in self.get_nodes()]
        idx = node_flag.index(False)
        return self.get_node(idx)

    def get_local_edge_idx(self,edge):
        for i, e in enumerate(self.get_edges()):
            if edge == e:
                return i
        assert 1==0, "Edge does not lie on element"

    def print_local_node_order(self):
        for node in self.get_nodes():
            print 'global node idx: ' + `node.get_global_idx()`

    def print_local_edge_order(self):
        for edge in self.get_edges():
            print 'global edge idx: ' + `edge.get_global_idx()`

    def print_local_edge_orientation(self):
        for i, edge in enumerate(self.get_edges()):
            print 'local edge : ' + `i` + ' global idx: ' + `edge.get_node(0).get_global_idx()` + ' to ' + `edge.get_node(1).get_global_idx()`

class ReferenceElement(Element):

    def __init__(self):
        p1 = Point(0.,0.) ; p2 = Point(1.,0.) ; p3 = Point(0., 1.)
        super(ReferenceElement,self).__init__([p1,p2,p3])
        self._init_edge_quad_pts()

    def _init_edge_quad_pts(self):
        """
        TODO - this is a silly function, it only allows a single 
        quadrature rule.  ultimately, this function needs to be
        removed and replace with 'set_edge_quad_pts' that takes
        the quadrature rule as an input.
        """
        self.reference_lagrange_pts = [[],[],[]]
        quadrature = quad.Quadrature(1)
        for i in range(3):
            edge_param = self.get_edge_parameterization(i)
            for pt in quadrature.edge_quad_pt:
                self.reference_lagrange_pts[i].append((edge_param[0](pt),
                                                       edge_param[1](pt)))

    def set_edge_quad_pts(self, quadrature):
        self.reference_lagrange_pts = [[],[],[]]
        for i in range(3):
            edge_param = self.get_edge_parameterization(i)
            for pt in quadrature.edge_quad_pt:
                self.reference_lagrange_pts[i].append((edge_param[0](pt),
                                                       edge_param[1](pt)))

    def get_lagrange_quad_point(self,i,n):
        '''
        Returns the lagrangian quad point on the boundaries of
        the reference triangle.

        Arguments
        ---------
        i : int 
            The reference triangle edge number
        n : int
            The quadrature point on the element
        '''
        return self.reference_lagrange_pts[i][n]

    @staticmethod
    def get_edge_parameterization(i):
        if i==0:
            return (lambda s: 1-s, lambda s: s)
        if i==1:
            return (lambda s: 0., lambda s: s)
        if i==2:
            return (lambda s: s, lambda s: 0.)

    @staticmethod
    def get_normal_vec(i):
        sqrt_2 = math.sqrt(2)
        if i==0:
            return (1./sqrt_2,1./sqrt_2)
        if i==1:
            return (-1., 0.)
        if i==2:
            return (0., -1.)

    @staticmethod
    def get_edge_length(i):
        if i==0:
            return math.sqrt(2)
        if i==1:
            return 1.
        if i==2:
            return 1.

class Point(object):

    def __init__(self,q1,q2):
        self.vals = (q1,q2)

    def __getitem__(self,i):
        """ Return the i-th point value

        input
        -----
        i : int
           The requested point value.
        """
        return self.vals[i]

    def get_distance(self,p):
        """ Find the distance with another point.

        input
        -----
        p : `Point`

        return
        ------
        val : float
        """
        p1 = p[0] ; q1 = self[0]
        p2 = p[1] ; q2 = self[1]
        return math.sqrt((p1-q1)**2 + (p2-q2)**2)

    def print_point(self):
        print 'point : ' + '('+ `self.vals[0]` + ' , ' + `self.vals[1]`+ ')'

    def get_vec_between_points(self,pt):
        """ Returns a vector between points.

        input
        -----
        pt : `Point`
            This is the terminal point

        return
        ------
        v1
        """
        p1 = pt[0] ; q1 = self[0]
        p2 = pt[1] ; q2 = self[1]
        return (p1-q1, p2-q2)

class Node(Point):

    def __init__(self,q1,q2,bdy_flag,global_idx):
        super(Node, self).__init__(q1,q2)
        self.set_bdy_flag(bdy_flag)
        self.set_global_idx(global_idx)

    def set_bdy_flag(self,bdy_flag):
        self.bdy_flag = bdy_flag
        
    def set_global_idx(self,global_idx):
        self.global_idx = global_idx

    def print_node(self):
        print 'global_idx : ' + `self.get_global_idx()`
        self.print_point()
    
    def get_global_idx(self):
        return self.global_idx

    def is_node_bdy(self):
        return self.bdy_flag

class QuadraturePoint(Point):

    def __init__(self,q1,q2,w=0):
        """ Initialize a quadrature point.

        input
        -----
        q1 : float
            The x-coordinate of the quadrature point
        q2 : float
            The y-coordinate of the quadrature point
        w : float
            The quadrature point's weight
        """
        super(QuadraturePoint,self).__init__(q1,q2)
        self.set_quad_weight(w)

    def set_quad_weight(self,w):
        self._weight = w

    def get_quad_weight(self):
        return self._weight

class Edge(object):

    def __init__(self,n1,n2):
        self._nodes = (n1,n2)

    def get_nodes(self):
        return self._nodes

    def get_node(self,i):
        return self._nodes[i]

    def print_node_coordinates(self,i):
        print "node (" + `i` + ") = " + `self._nodes[i][0]` + ', ' + `self._nodes[i][1]`

    def get_edge_length(self):
        return self.get_node(0).get_distance(self.get_node(1))

    def get_tangent_vec(self):
        vec = self.get_node(0).get_vec_between_points(self.get_node(1))
        return vec
    
    def get_unit_normal_vec(self):
        tan_vec = self.get_tangent_vec()
        if tan_vec[1] == 0:
            return (0.,1.)
        else:
            n2 = - tan_vec[0] / tan_vec[1]
            norm_fac = math.sqrt(1 + n2**2)
            return ( 1./norm_fac, n2/norm_fac)

class MeshEdge(Edge):

    def __init__(self,n1,n2,bdy_flag,global_idx):
        super(MeshEdge,self).__init__(n1,n2)
        self._set_bdy_flag(bdy_flag)
        self._set_global_idx(global_idx)

    def _set_bdy_flag(self, bdy_flag):
        self._bdy_flag = bdy_flag
        
    def _set_global_idx(self,global_idx):
        self._global_idx = global_idx

    def get_bdy_flag(self):
        return self._bdy_flag
        
    def get_global_idx(self):
        return self._global_idx

    def is_edge_bdy(self):
        return self._bdy_flag

    def print_node_indices(self):
        for node in self.get_nodes():
            print 'global node idx: ' + `node.get_global_idx()`

    def does_outward_normal_match_global_normal(self,element):
        local_edge_idx = element.get_local_edge_idx(self)
        n3 = element.get_non_edge_node(local_edge_idx)
        norm_vec = self.get_unit_normal_vec()
        v1 = n3.get_vec_between_points(self.get_node(0))
        v2 = n3.get_vec_between_points(self.get_node(1))
        dot = la.Operators().dot_product(norm_vec,v1)
        if dot >= 0:
            return True
        elif dot < 0:
            return False

    def get_outward_unit_normal_vec(self,element):
        norm_vec = self.get_unit_normal_vec()
        match = self.does_outward_normal_match_global_normal(element)
        if match:
            return norm_vec
        else:
            return (-norm_vec[0],-norm_vec[1])

class Mesh():

    def __init__(self, filename):
        self._set_file_paths(filename)
        self._set_nodes()
        self._set_edges()
        self._set_elements()
        
    def _set_file_paths(self,filename):
        cur_dir = os.path.abspath('.')
        cur_dir = os.path.join(cur_dir,'import_modules/')
        self._mesh_node_file = os.path.join(cur_dir,filename+'.node')
        self._mesh_ele_file = os.path.join(cur_dir,filename+'.ele')
        self._mesh_edge_file = os.path.join(cur_dir,filename+'.edge')

    def _set_nodes(self):
        with open(self._mesh_node_file,'r') as f:
            for i, line in enumerate(f):
                line_str = line.split()
                if i==0:
                    self.num_nodes = int(line_str[0])
                    node_lst = [None]*self.num_nodes
                if i > 0 and line[0]!='#':
                    coord = Node(float(line_str[1]),
                                 float(line_str[2]),
                                 bdy_flag =int(line_str[3]),
                                 global_idx=i-1)
                    node_lst[i-1] = coord
        self._node_array = tuple(node_lst)
        self.set_node_val_array()

    def _set_edges(self):
        self._edge_dic = {}
        _bdy_edge_array = []
        with open(self._mesh_edge_file,'r') as f:
            for i, line in enumerate(f):
                line_str = line.split()
                if i==0:
                    num_edges = int(line_str[0])
                    _edge_array = [None]*num_edges
                if i > 0 and line[0]!='#':
                    node_1 = self._node_array[int(line_str[1])-1]
                    node_2 = self._node_array[int(line_str[2])-1]
                    n1_idx = node_1.get_global_idx()
                    n2_idx = node_2.get_global_idx()
                    if n1_idx < n2_idx:
                        edge = MeshEdge(node_1,
                                        node_2,
                                        bdy_flag = int(line_str[3]),
                                        global_idx=i-1)
                    elif n2_idx < n1_idx:
                        edge = MeshEdge(node_2,
                                        node_1,
                                        bdy_flag = int(line_str[3]),
                                        global_idx=i-1)
                    _edge_array[i-1] = edge
                    self._edge_dic[(node_1.get_global_idx(),
                                    node_2.get_global_idx())] = i-1
                    self._edge_dic[(node_2.get_global_idx(),
                                    node_1.get_global_idx())] = i-1
                    if int(line_str[3]) == 1:                           
                        _bdy_edge_array.append(edge)                    
        self._edge_array = tuple(_edge_array)
        self._bdy_edge_array = tuple(_bdy_edge_array)

    def _set_elements(self):
        with open(self._mesh_ele_file,'r') as f:
            for i, line in enumerate(f):
                line_str = line.split()
                if i==0:
                    num_elements = int(line_str[0])
                    _element_array = [None]*num_elements
                if i > 0 and line_str[0]!='#':
                    node_idx = [self._node_array[int(a)-1].get_global_idx() for a in line_str[1:4]]
                    node_idx.sort()
                    nodes = [self._node_array[a] for a in node_idx]
                    bdy_element = bool(sum([node.is_node_bdy() for node in nodes]))
                    edge_idx = [self.get_global_edge_idx(nodes[1].get_global_idx(), nodes[2].get_global_idx()),
                                self.get_global_edge_idx(nodes[2].get_global_idx(), nodes[0].get_global_idx()),
                                self.get_global_edge_idx(nodes[0].get_global_idx(), nodes[1].get_global_idx())]
                    edges = tuple([self.get_edge(j) for j in edge_idx])
                    element = MeshElement(nodes,
                                          edges,
                                          bdy_flag = bdy_element,
                                          global_idx = i-1)
                    _element_array[i-1] = element
        self._element_array = tuple(_element_array)
        self.set_element_node_idx_array()

    def get_element(self,i):
        return self._element_array[i]

    def get_elements(self):
        return self._element_array

    def set_element_node_idx_array(self):
        num_elements = self.get_num_mesh_elements()
        self._element_node_idx_array = np.zeros(shape=(num_elements,
                                                       3))
        for i, element in enumerate(self._element_array):
            for j in range(self.get_num_nodes_per_element()):
                self._element_node_idx_array[i][j] = element.get_node(j).global_idx

    def get_element_node_idx_array(self):
        return self._element_node_idx_array

    def get_element_iterator(self):
        return ft.Iterator(self._element_array)

    def get_node(self,i):
        return self._node_array[i]

    def get_nodes(self):
        return self._node_array

    def set_node_val_array(self):
        num_nodes = self.get_num_mesh_nodes()
        self._node_val_array = np.zeros(shape=(num_nodes,2))
        for i, node in enumerate(self._node_array):
            self._node_val_array[i][0] = node[0]
            self._node_val_array[i][1] = node[1]

    def get_node_val_array(self):
        return self._node_val_array
            
    def get_mesh_plt_info(self):
        node_val_array = self.get_node_val_array()
        node_val_x = node_val_array[:,0]
        node_val_y = node_val_array[:,1]
        element_mesh_array = self.get_element_node_idx_array()
        return node_val_x, node_val_y, element_mesh_array

    def get_num_bdy_edges(self):
        return len(self._bdy_edge_array)

    def get_bdy_edges_iterator(self):
        return ft.Iterator(self._bdy_edge_array)

    def get_bdy_edge(self,i):
        return self._bdy_edge_array[i]

    def get_num_edges(self):
        return len(self._edge_array)

    def get_edge(self,i):
        return self._edge_array[i]

    def get_edges(self):
        return self._edge_array

    def get_num_nodes_per_element(self):
        return self.get_element(0).get_num_nodes()

    def get_num_mesh_elements(self):
        return len(self._element_array)

    def get_num_mesh_nodes(self):
        return len(self._node_array)

    def get_num_mesh_edges(self):
        return len(self._edge_array)

    def get_num_edges_per_element(self):
        return self.get_element(0).get_num_edges()

    def get_global_edge_idx(self,idx0,idx1):
        return self._edge_dic[(idx0,idx1)]

    def print_nodes(self):
        for i, node in enumerate(self._node_array):
            print "Node " +`i`+ ": " + '(' + `node[0]`+ ',' + `node[1]` + ')'

    def print_edge_node_indices(self):
        for edge in self.get_edges():
            print 'Edge ' + `edge.get_global_idx()`
            edge.print_node_indices()

    def print_element_node_indices(self):
        for element in self.get_elements():
            print 'Element ' + `element.get_global_idx()`
            element.print_node_indices()

class StructuredMesh(Mesh):

    def __init__(self, mesh_len, h):
        """ Initialization function for structured mesh.

        Arguments
        ---------
        mesh_len : lst
            List containing dimensions of the mesh

        h : float
            Length of triangle edge
        """
        self.set_mesh_len(mesh_len)        
        self.set_h(h)
        self._set_nodes()
        self._set_edges()
        self._set_elements()

    def set_mesh_len(self,mesh_len):
        self._xL = float(mesh_len[0])
        self._yL = float(mesh_len[1])

    def get_mesh_len(self):
        return [self._xL, self._yL]
        
    def set_h(self,h):
        self._h = h

    def get_h(self):
        return self._h

    def get_nx_ny(self):
        x_len = self.get_mesh_len()[0];  y_len = self.get_mesh_len()[1]
        h = self.get_h()
        nx = int(float(x_len) / h) ; ny = int(float(y_len) / h)
        return nx, ny
        
    def _set_nodes(self):
        x_len = self.get_mesh_len()[0];  y_len = self.get_mesh_len()[1]        
        nx, ny = self.get_nx_ny()

        xcoords = np.linspace(0. , x_len , nx+1)
        ycoords = np.linspace(0. , y_len , ny+1)
        x_grid, y_grid = np.meshgrid(xcoords, ycoords)
        node_vals = zip(x_grid.flatten() , y_grid.flatten())
        self.num_nodes = len(x_grid)*len(y_grid)
        node_lst = [None]*self.num_nodes
        
        for i, node_val in enumerate(node_vals):
            bdy_flag = int(0)
            if node_val[0] in [0, x_len] or node_val[1] in [0, y_len]:
                bdy_flag = int(1)
            coord = Node(float(node_val[0]),
                         float(node_val[1]),
                         bdy_flag = bdy_flag,
                         global_idx = i)
            node_lst[i] = coord

        self._node_array = tuple(node_lst)
        self.set_node_val_array()

    def _set_edges(self):
        self._edge_dic = {}
        bdy_edge_array = []

        nx, ny = self.get_nx_ny()
        num_edges = int(nx*(ny+1) + ny*(nx+1) + nx*ny)
        _edge_array = [None]*num_edges

        for j in range(ny+1):
            if j != ny:
                for i in range(nx+1):
                    if i != nx:
                        global_node_idx = int((nx+1)*j + i)
                        node_1 = self._node_array[global_node_idx]
                        node_2 = self._node_array[global_node_idx + 1]

                        bdy_flag = int(0)
                        if node_1.bdy_flag == 1 and node_2.bdy_flag == 1:
                            if node_1.vals[0] == node_2.vals[0]:
                                bdy_flag = int(1)
                            elif node_1.vals[1] == node_2.vals[1]:
                                bdy_flag = int(1)

                        global_edge_idx = int((3*nx + 1)*j + i)
                        edge = MeshEdge(node_1,
                                        node_2,
                                        bdy_flag = bdy_flag,
                                        global_idx = global_edge_idx)
#                        import pdb ; pdb.set_trace()
                        _edge_array[global_edge_idx] = edge
                        self._edge_dic[(node_1.get_global_idx(),
                                        node_2.get_global_idx())] = global_edge_idx
                        self._edge_dic[(node_2.get_global_idx(),
                                        node_1.get_global_idx())] = global_edge_idx            
                        if edge.get_bdy_flag() == 1:
                            bdy_edge_array.append(edge)

                        for k in range(2):
                            node_2 = self._node_array[global_node_idx + (nx+1)  + k]
                            bdy_flag = int(0)
                            if node_1.bdy_flag == 1 and node_2.bdy_flag == 1:
                                if node_1.vals[0] == node_2.vals[0]:
                                    bdy_flag = int(1)
                                elif node_1.vals[1] == node_2.vals[1]:
                                    bdy_flag = int(1)

                            global_edge_idx = int( (3*nx+1)*j + nx + 2*i + k)
                            edge = MeshEdge(node_1,
                                            node_2,
                                            bdy_flag = bdy_flag,
                                            global_idx = global_edge_idx)
                            _edge_array[global_edge_idx] = edge
                            self._edge_dic[(node_1.get_global_idx(),
                                            node_2.get_global_idx())] = global_edge_idx
                            self._edge_dic[(node_2.get_global_idx(),
                                            node_1.get_global_idx())] = global_edge_idx                            
                            if edge.get_bdy_flag() == 1:
                                bdy_edge_array.append(edge)
                            

                    elif i == nx:
                        global_node_idx = int((nx+1)*j + i)
                        node_1 = self._node_array[global_node_idx]
                        node_2 = self._node_array[global_node_idx + (nx+1)]

                        bdy_flag = int(0)
                        if node_1.bdy_flag == 1 and node_2.bdy_flag == 1:
                            if node_1.vals[0] == node_2.vals[0]:
                                bdy_flag = int(1)
                            elif node_1.vals[1] == node_2.vals[1]:
                                bdy_flag = int(1)
                            
                        global_edge_idx = int((3*nx + 1)*j + 3*nx)
                        edge = MeshEdge(node_1,
                                        node_2,
                                        bdy_flag = bdy_flag,
                                        global_idx = global_edge_idx)
                        _edge_array[global_edge_idx] = edge
                        self._edge_dic[(node_1.get_global_idx(),
                                        node_2.get_global_idx())] = global_edge_idx
                        self._edge_dic[(node_2.get_global_idx(),
                                        node_1.get_global_idx())] = global_edge_idx                        
                        if edge.get_bdy_flag() == 1:
                            bdy_edge_array.append(edge)

            elif j == ny:
                for i in range(nx):
                    if i != nx:
                        global_node_idx = int((nx+1)*j + i)
                        node_1 = self._node_array[global_node_idx]
                        node_2 = self._node_array[global_node_idx + 1]

                        bdy_flag = int(0)
                        if node_1.bdy_flag == 1 and node_2.bdy_flag == 1:
                            if node_1.vals[0] == node_2.vals[0]:
                                bdy_flag = int(1)
                            elif node_1.vals[1] == node_2.vals[1]:
                                bdy_flag = int(1)

                        global_edge_idx = int((3*nx + 1)*j + i)
                        edge = MeshEdge(node_1,
                                        node_2,
                                        bdy_flag = bdy_flag,
                                        global_idx = global_edge_idx)
                        _edge_array[global_edge_idx] = edge
                        self._edge_dic[(node_1.get_global_idx(),
                                        node_2.get_global_idx())] = global_edge_idx
                        self._edge_dic[(node_2.get_global_idx(),
                                        node_1.get_global_idx())] = global_edge_idx                                    
                        if edge.get_bdy_flag() == 1:
                            bdy_edge_array.append(edge)
                
        self._edge_array = tuple(_edge_array)
        self._bdy_edge_array = tuple(bdy_edge_array)

    def _set_elements(self):
        nx, ny = self.get_nx_ny()
        num_elements = 2*nx*ny
        _element_array = [None]*num_elements
        for j in range(ny):
            for i in range(nx):
                base_node_idx = int( (nx+1)*j + i )
                node_idx_comm = int( base_node_idx + (nx+2) )
                for k in range(2):
                    node_idx_diff = int( base_node_idx + (1-k) * (nx+1) + k )
                    if k==0:
                        idx = [base_node_idx, node_idx_diff, node_idx_comm]
                        idx.sort()
                        nodes = [self._node_array[ii] for ii in idx]
                    elif k==1:
                        idx = [base_node_idx, node_idx_comm, node_idx_diff]
                        idx.sort()
                        nodes = [self._node_array[ii] for ii in idx]
                    edge_idx = [self.get_global_edge_idx(nodes[1].get_global_idx(), nodes[2].get_global_idx()),
                                self.get_global_edge_idx(nodes[2].get_global_idx(), nodes[0].get_global_idx()),
                                self.get_global_edge_idx(nodes[0].get_global_idx(), nodes[1].get_global_idx())]
                    edges = tuple([self.get_edge(p) for p in edge_idx])
                    bdy_element = bool(sum(edge.is_edge_bdy() for edge in edges) >= 1)                    
                    element_idx = 2*(nx)*j + 2*i  + k                    
                    element = MeshElement(nodes,
                                          edges,
                                          bdy_flag = bdy_element,
                                          global_idx = element_idx)
                    _element_array[element_idx] = element
        self._element_array = tuple(_element_array)
        self.set_element_node_idx_array()

class OneElementMesh(Mesh):

    def __init__(self):
        """ Initialization function for structured mesh.

        Arguments
        ---------
        mesh_len : lst
            List containing dimensions of the mesh

        h : float
            Length of triangle edge
        """
        self._set_nodes()
        self._set_edges()
        self._set_elements()

        
    def _set_nodes(self):
        xcoords = [0.,1.]
        ycoords = [0.,1.]
        node_vals = [(0.,0.), (1.,0.), (0.,1.)]
        self.num_nodes = 3
        node_lst = [None]*self.num_nodes
        
        for i, node_val in enumerate(node_vals):
            coord = Node(float(node_val[0]),
                         float(node_val[1]),
                         bdy_flag = 1,
                         global_idx = i)
            node_lst[i] = coord

        self._node_array = tuple(node_lst)
        self.set_node_val_array()

    def _set_edges(self):
        self._edge_dic = {}
        bdy_edge_array = []

        num_edges = 3
        _edge_array = [None]*num_edges

        for i,nodes in enumerate([(1,2),(0,2),(0,1)]):
            node_1 = self._node_array[nodes[0]]
            node_2 = self._node_array[nodes[1]]

            bdy_flag = int(1)
            
            global_edge_idx = int(i)
            edge = MeshEdge(node_1,
                            node_2,
                            bdy_flag = bdy_flag,
                            global_idx = global_edge_idx)
            _edge_array[global_edge_idx] = edge
            self._edge_dic[(node_1.get_global_idx(),
                            node_2.get_global_idx())] = global_edge_idx
            self._edge_dic[(node_2.get_global_idx(),
                            node_1.get_global_idx())] = global_edge_idx            
            if edge.get_bdy_flag() == 1:
                bdy_edge_array.append(edge)
                        
        self._edge_array = tuple(_edge_array)
        self._bdy_edge_array = tuple(bdy_edge_array)

    def _set_elements(self):
        _element_array = [None]*1
        nodes = [self._node_array[ii] for ii in range(3)]
        edge_idx = [self.get_global_edge_idx(nodes[1].get_global_idx(), nodes[2].get_global_idx()),
                    self.get_global_edge_idx(nodes[2].get_global_idx(), nodes[0].get_global_idx()),
                    self.get_global_edge_idx(nodes[0].get_global_idx(), nodes[1].get_global_idx())]
        edges = tuple([self.get_edge(p) for p in edge_idx])
        bdy_element = True
        element_idx = 0 
        element = MeshElement(nodes,
                              edges,
                              bdy_flag = bdy_element,
                              global_idx = element_idx)
        _element_array[element_idx] = element
        self._element_array = tuple(_element_array)
        self.set_element_node_idx_array()

