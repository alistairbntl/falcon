import numpy as np
import mapping_tools as mpt

class DOFHandler():

    def __init__(self, mesh, basis):
        self.mesh = mesh
        self.basis_lst = basis
        self._set_dof_info()
        self._set_global_dof_idx()
        self._build_local_2_global()
        self._set_global_basis_dof_rng()

    def _build_local_2_global(self):
        """ Build the local 2 global map for FE calculations.
        """
        num_mesh_elements = self.mesh.get_num_mesh_elements()
        num_mesh_nodes = self.mesh.get_num_mesh_nodes()
        num_mesh_edges = self.mesh.get_num_mesh_edges()

        num_dof_per_element = self.get_num_dof_per_element()

        self._l2g_map = np.zeros(shape=(num_mesh_elements,
                                        num_dof_per_element))
        self._bdy_dof_dic = {}
        
        for element_idx in range(num_mesh_elements):

            dof_idx = 0

            cur_element = self.mesh.get_element(element_idx)
            cur_element_nodes = cur_element.get_nodes()
            cur_element_edges = cur_element.get_edges()

            for n, basis in enumerate(self.basis_lst):

                bdy_lst = set()
                
                translation_factor = self._update_translation_factor(n)

                num_dof_per_node = int(basis.get_num_dof_per_node())
                num_dof_per_edge = int(basis.get_num_dof_per_edge())
                num_interior_dof = int(basis.get_num_interior_dof())

                for i in range(num_dof_per_node):
                    for node in cur_element_nodes:
                        global_idx = int(i * num_mesh_nodes
                                         + node.get_global_idx())
                        self._set_l2g_map(element_idx,
                                          dof_idx,
                                          global_idx + translation_factor)
                        if node.is_node_bdy():
                            bdy_lst.add((global_idx+translation_factor,'n',node.get_global_idx()))
                        dof_idx += 1

                for i in range(num_dof_per_edge):
                    for edge in cur_element_edges:
                        global_idx = int(self.get_num_node_dof_basis(basis)
                                         + i*num_mesh_edges
                                         + edge.get_global_idx())
                        self._set_l2g_map(element_idx,
                                          dof_idx,
                                          global_idx + translation_factor)
                        if edge.is_edge_bdy():
                            bdy_lst.add((global_idx+translation_factor,'e',edge.get_global_idx()))
                        dof_idx += 1

                for i in range(num_interior_dof):
                    global_idx = int(self.get_num_node_dof_basis(basis)
                                     + self.get_num_edge_dof_basis(basis)
                                     + i*num_mesh_elements
                                     + cur_element.get_global_idx())

                    self._set_l2g_map(element_idx,
                                      dof_idx,
                                      global_idx + translation_factor)
                    if cur_element.is_bdy_element():
                        bdy_lst.add((global_idx+translation_factor,'E',cur_element.get_global_idx()))
                    dof_idx += 1

                self._set_bdy_dof_idx(bdy_lst,basis.get_name())

    def _set_global_basis_dof_rng(self):
        self._global_basis_dof_rng = []
        idx_counter = 0
        mesh = self.mesh
        for n, basis in enumerate(self.basis_lst):
            num_dof = (basis.get_num_dof_per_node() * mesh.get_num_mesh_nodes()
                       + basis.get_num_dof_per_edge() * mesh.get_num_mesh_edges()
                       + basis.get_num_interior_dof() * mesh.get_num_mesh_elements())
            self._global_basis_dof_rng.append((idx_counter,
                                               idx_counter + num_dof))
            idx_counter += num_dof

    def _update_translation_factor(self,n):
        translation_factor = int(0)
        if n==0:
            return translation_factor
        else:
            for i in range(n):
                translation_factor += self.get_num_dof_basis(self.basis_lst[i])
        return int(translation_factor)

    def _set_l2g_map(self,element_idx,dof_idx,global_idx):
        self._l2g_map[element_idx,dof_idx] = global_idx

    def _set_bdy_dof_idx(self,idx,basis):
        try:
            self._bdy_dof_dic[basis] = self._bdy_dof_dic[basis].union(idx)
        except KeyError:
            self._bdy_dof_dic[basis]=idx

    def _set_global_dof_idx(self):
        """ """
        self._set_num_dof()
        self._global_dof_idx = tuple(range(self.get_num_dof()))

    def _set_num_dof(self):
        self._num_dof = int(self.get_num_node_dof()
                            + self.get_num_edge_dof()
                            + self.get_num_interior_dof())

    def _set_dof_info(self):
        """
        This functions defines a tuple with the dof information
        for each basis as well as global dof information.
        """
        dof_info = []
        for basis in self.basis_lst:
            local_dof_info = []
            local_dof_info.append(basis.get_num_dof_per_node())
            local_dof_info.append(basis.get_num_dof_per_edge())
            local_dof_info.append(basis.get_num_interior_dof())
            dof_info.append(local_dof_info)
        self._individual_basis_dof_info = tuple(dof_info)
        self._mixed_dof_info = (self.get_num_node_dof(),
                                self.get_num_edge_dof(),
                                self.get_num_interior_dof())

    def get_global_basis_dof_rng(self,n):
        return self._global_basis_dof_rng[n]
    
    def get_mesh(self):
        return self.mesh

    def get_mixed_dof_info(self):
        return self._mixed_dof_info

    def get_individual_dof_info(self,basis_num=None):
        if basis_num is not None:
            return self._individual_basis_dof_info[basis_num]
        else:
            return self._individual_basis_dof_info

    def get_num_dof(self):
        return self._num_dof

    def get_global_basis_dof_rng(self):
        return self._global_basis_dof_rng

    def get_num_dof_basis(self,basis):
        return int(self.get_num_node_dof_basis(basis)
                   + self.get_num_edge_dof_basis(basis)
                   + self.get_num_interior_dof_basis(basis))

    def get_num_dof_per_element(self):
        return int(self.get_num_node_dof_per_element()
                   + self.get_num_edge_dof_per_element()
                   + self.get_num_interior_dof_per_element())

    def get_global_dof_idx(self):
        return self._global_dof_idx

    def get_local_2_global(self,element_idx,local_dof_idx):
        return int(self._l2g_map[element_idx,local_dof_idx])

    def get_local_edge_dof_idx(self,
                               global_dof_idx,
                               global_edge_idx):
        """ ARB - this function could cause problems on
        down the line ... I'm tired, just a heads up. """
        num_dof_per_edge = self.get_num_dof_per_edge()
        num_edges = self.mesh.get_num_edges()
        for i in range(num_dof_per_edge):
            if global_dof_idx - i*num_edges == global_edge_idx:
                return i
        return -1

    def get_num_node_dof(self):
        num_dof_per_node = 0
        for basis in self.basis_lst:
            num_dof_per_node += basis.get_num_dof_per_node()
        num_mesh_nodes = self.mesh.get_num_mesh_nodes()
        return int(num_dof_per_node*num_mesh_nodes)

    def get_num_node_dof_basis(self,basis):
        num_dof_per_node = basis.get_num_dof_per_node()
        num_mesh_nodes = self.mesh.get_num_mesh_nodes()
        return int(num_dof_per_node*num_mesh_nodes)

    def get_num_dof_per_edge(self):
        num_dof_per_edge = 0        
        for basis in self.basis_lst:
            num_dof_per_edge += basis.get_num_dof_per_edge()
        return num_dof_per_edge
    
    def get_num_edge_dof(self):
        num_dof_per_edge = self.get_num_dof_per_edge()
        num_mesh_edges = self.mesh.get_num_mesh_edges()
        return int(num_dof_per_edge*num_mesh_edges)

    def get_num_edge_dof_basis(self,basis):
        num_dof_per_edge = basis.get_num_dof_per_edge()
        num_mesh_nodes = self.mesh.get_num_mesh_edges()
        return int(num_dof_per_edge*num_mesh_nodes)

    def get_num_interior_dof(self):
        num_dof_per_element = 0
        for basis in self.basis_lst:
            num_dof_per_element += basis.get_num_interior_dof()
        num_mesh_elements = self.mesh.get_num_mesh_elements()
        return int(num_dof_per_element*num_mesh_elements)

    def get_num_interior_dof_basis(self,basis):
        num_dof_per_element = basis.get_num_interior_dof()
        num_mesh_elements = self.mesh.get_num_mesh_elements()
        return int(num_dof_per_element*num_mesh_elements)
    
    def get_num_node_dof_per_element(self):
        num_dof_per_node = 0
        for basis in self.basis_lst:
            num_dof_per_node += basis.get_num_dof_per_node()
        num_mesh_nodes = self.mesh.get_num_nodes_per_element()
        return int(num_dof_per_node*num_mesh_nodes)

    def get_num_edge_dof_per_element(self):
        num_dof_per_edge = 0
        for basis in self.basis_lst:
            num_dof_per_edge += basis.get_num_dof_per_edge()
        num_mesh_edges = self.mesh.get_num_edges_per_element()
        return int(num_dof_per_edge*num_mesh_edges)

    def get_num_interior_dof_per_element(self):
        num_dof_per_element = 0
        for basis in self.basis_lst:
            num_dof_per_element += basis.get_num_interior_dof()
        return int(num_dof_per_element)

    def get_bdy_dof_dic(self,basis_name=None):
        if basis_name:
            return self._bdy_dof_dic[basis_name]
        else:
            return self._bdy_dof_dic
