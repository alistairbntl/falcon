import math
import pytest
import numpy as np

from .context import falcon
from falcon import mesh_tools as mt

def test_point_1():
    p1 = mt.Point(1.,0.)
    q1 = mt.Point(0.,0.)
    a = p1.get_distance(q1)
    assert a == 1.0
    assert p1[0] == 1.0

def test_node_1():
    n1 = mt.Node(1.,0.,0,1)
    assert n1.get_global_idx() == 1
    assert n1.is_node_bdy() == 0

def test_element_1():
    p1 = mt.Point(2.,0.)
    p2 = mt.Point(1.,1.)
    p3 = mt.Point(1.,0.)
    coords = [p1,p2,p3]
    element = mt.Element(coords)

@pytest.mark.bdm
def test_reference_element_1():
    reference_element = mt.ReferenceElement()
    assert reference_element.get_edge_length(0) == math.sqrt(2)
    assert reference_element.get_edge_length(1) == 1.0
    assert reference_element.get_edge_length(2) == 1.0
    assert reference_element.get_center()[0] == 1./3.
    assert reference_element.get_center()[1] == 1./3.

def test_reference_element_2():
    reference_element = mt.ReferenceElement()
    edge_1_param = reference_element.get_edge_parameterization(0)
    assert edge_1_param[0](0.5) == 0.5
    assert edge_1_param[1](0.5) == 0.5
    edge_2_param = reference_element.get_edge_parameterization(1)
    assert edge_2_param[0](0.5) == 0.
    assert edge_2_param[1](0.5) == 0.5
    edge_3_param = reference_element.get_edge_parameterization(2)
    assert edge_3_param[0](0.5) == 0.5
    assert edge_3_param[1](0.5) == 0.

def test_mesh_edge_1():
    n1 = mt.Node(0.1, 0.2, 1, 0)
    n2 = mt.Node(0.1, 0.4, 1, 0)
    mesh_edge = mt.MeshEdge(n1,n2,1,0)
    node_0 = mesh_edge.get_node(0)
    node_1 = mesh_edge.get_node(1)
    assert node_0[0] == 0.1 ; assert node_0[1] == 0.2
    assert node_1[0] == 0.1 ; assert node_1[1] == 0.4
    edge_length = mesh_edge.get_edge_length()
    assert edge_length == 0.2
    assert mesh_edge.get_unit_normal_vec() == (1.0,0.0)

def test_edge_2():
    n1 = mt.Point(0.,1.)
    n2 = mt.Point(1.,0.)
    edge = mt.Edge(n1,n2)
    assert edge.get_unit_normal_vec() == (1./math.sqrt(2), 1./math.sqrt(2))
    
def test_basic_mesh_1(basic_mesh):
    mesh = basic_mesh
    assert mesh.get_node(0)[0] ==0
    assert mesh.get_node(0)[1] ==0
    assert mesh.get_node(1)[0] ==1
    assert mesh.get_node(1)[1] ==0
    assert mesh.get_node(2)[0] ==0
    assert mesh.get_node(2)[1] ==1
    assert mesh.get_node(3)[0] ==1
    assert mesh.get_node(3)[1] ==1
    assert mesh.get_element(0).get_point(0).vals[0]==0.
    assert mesh.get_element(0).get_point(0).vals[1]==0.
    assert mesh.get_element(0).get_point(1).vals[0]==1.
    assert mesh.get_element(0).get_point(1).vals[1]==0.
    assert mesh.get_element(0).get_point(2).vals[0]==0.
    assert mesh.get_element(0).get_point(2).vals[1]==1.

def test_basic_mesh_2(basic_mesh):
    mesh = basic_mesh
    for i in range(mesh.get_num_mesh_edges()):
        n0, n1 = mesh.get_edge(i).get_nodes()
        n0_idx = n0.get_global_idx()
        n1_idx = n1.get_global_idx()
        edge_idx_1 = mesh.get_global_edge_idx(n0_idx,n1_idx)
        edge_idx_2 = mesh.get_global_edge_idx(n1_idx,n0_idx)
        assert edge_idx_1 == mesh.get_edge(i).get_global_idx()
        assert edge_idx_2 == mesh.get_edge(i).get_global_idx()
        
def test_basic_mesh_3(basic_mesh):
    mesh = basic_mesh
    E0 = mesh.get_element(0)
    assert E0.get_edge(0).get_global_idx() == 2
    assert E0.get_edge(1).get_global_idx() == 0
    assert E0.get_edge(2).get_global_idx() == 1    
    E1 = mesh.get_element(1)
    assert E1.get_edge(0).get_global_idx() == 4
    assert E1.get_edge(1).get_global_idx() == 3
    assert E1.get_edge(2).get_global_idx() == 2

def test_basic_mesh_4(basic_mesh_2):
    mesh = basic_mesh_2
    assert mesh.get_node(0)[0] == 0.0 and mesh.get_node(0)[1] == 0.0
    assert mesh.get_node(1)[0] == 1.0 and mesh.get_node(1)[1] == 0.0
    assert mesh.get_node(2)[0] == 0.0 and mesh.get_node(2)[1] == 1.0
    assert mesh.get_node(3)[0] == 1.0 and mesh.get_node(3)[1] == 1.0
    assert abs(mesh.get_node(4)[0] - 0.65) < 0.005
    assert abs(mesh.get_node(4)[1] - 0.35) < 0.005
    area = 0.
    for eN in range(mesh.get_num_mesh_elements()):
        area += mesh.get_element(eN).get_area()
    assert area == 1.

def test_basic_mesh_5(basic_mesh):
    mesh = basic_mesh
    
    def get_test_info(eN,e):
        element = mesh.get_element(eN)
        edge = element.get_edge(e)
        return edge, element

    edge, n3 = get_test_info(0,0)
    assert (1./math.sqrt(2),1./math.sqrt(2)) == edge.get_unit_normal_vec()
    assert (1./math.sqrt(2),1./math.sqrt(2)) == edge.get_outward_unit_normal_vec(n3)
    assert True == edge.does_outward_normal_match_global_normal(n3)
    edge, n3 = get_test_info(0,1)
    assert (1.0,0.0) == edge.get_unit_normal_vec()
    assert (-1.0,0.0) == edge.get_outward_unit_normal_vec(n3)
    assert False == edge.does_outward_normal_match_global_normal(n3)    
    edge, n3 = get_test_info(0,2)
    assert (0.0,1.0) == edge.get_unit_normal_vec()
    assert (0.0,-1.0) == edge.get_outward_unit_normal_vec(n3)
    assert False == edge.does_outward_normal_match_global_normal(n3)
    edge, n3 = get_test_info(1,0)
    assert (0.0,1.0) == edge.get_unit_normal_vec()
    assert (0.0,1.0) == edge.get_outward_unit_normal_vec(n3)
    assert True == edge.does_outward_normal_match_global_normal(n3)
    edge, n3 = get_test_info(1,1)
    assert (1.0,0.0) == edge.get_unit_normal_vec()
    assert (1.0,0.0) == edge.get_outward_unit_normal_vec(n3)
    assert True == edge.does_outward_normal_match_global_normal(n3)    
    edge, n3 = get_test_info(1,2)
    assert (1./math.sqrt(2),1./math.sqrt(2)) == edge.get_unit_normal_vec()
    assert (-1./math.sqrt(2),-1./math.sqrt(2)) == edge.get_outward_unit_normal_vec(n3)
    assert False == edge.does_outward_normal_match_global_normal(n3)    

def test_struc_mesh_1():
    mesh = mt.StructuredMesh([1,1],0.25)
    assert mesh.get_node(0)[0] == 0.0 and mesh.get_node(0)[1] == 0.0
    assert mesh.get_node(0).bdy_flag == 1
    assert mesh.get_node(1)[0] == 0.25 and mesh.get_node(1)[1] == 0.0
    assert mesh.get_node(1).bdy_flag == 1
    assert mesh.get_node(2)[0] == 0.5 and mesh.get_node(2)[1] == 0.0
    assert mesh.get_node(2).bdy_flag == 1
    assert mesh.get_node(3)[0] == 0.75 and mesh.get_node(3)[1] == 0.0
    assert mesh.get_node(3).bdy_flag == 1
    assert mesh.get_node(24)[0] == 1.0 and mesh.get_node(24)[1] == 1.0
    assert mesh.get_node(24).bdy_flag == 1

@pytest.mark.struc_mesh
def test_struc_mesh_2():
    mesh = mt.StructuredMesh([1,1],1.0)
    e0 = mesh.get_edge(0)
    assert e0.is_edge_bdy() == 1 ; assert e0.get_edge_length() == 1.0
    assert np.allclose(e0.get_nodes()[0].vals, (0.,0.))
    assert np.allclose(e0.get_nodes()[1].vals, (1.,0.))
    e0 = mesh.get_edge(1)
    assert e0.is_edge_bdy() == 1 ; assert e0.get_edge_length() == 1.0
    assert np.allclose(e0.get_nodes()[0].vals, (0.,0.))
    assert np.allclose(e0.get_nodes()[1].vals, (0.,1.))
    e0 = mesh.get_edge(2)
    # Note - this is a degenerate case for which e0 is a boundary node ***
    assert e0.is_edge_bdy() == 1 ; assert e0.get_edge_length() == math.sqrt(2)
    assert np.allclose(e0.get_nodes()[0].vals, (0.,0.))
    assert np.allclose(e0.get_nodes()[1].vals, (1.,1.))
    e0 = mesh.get_edge(3)
    assert e0.is_edge_bdy() == 1 ; assert e0.get_edge_length() == 1.0
    assert np.allclose(e0.get_nodes()[0].vals, (1.,0.))
    assert np.allclose(e0.get_nodes()[1].vals, (1.,1.))
    e0 = mesh.get_edge(4)
    assert e0.is_edge_bdy() == 1 ; assert e0.get_edge_length() == 1.0
    assert np.allclose(e0.get_nodes()[0].vals, (0.,1.))
    assert np.allclose(e0.get_nodes()[1].vals, (1.,1.))
    E0 = mesh.get_element(0) ; E1 = mesh.get_element(1)
    assert E0.get_area() == 0.5 ; assert E1.get_area() == 0.5

@pytest.mark.struc_mesh
def test_struc_mesh_3():
    mesh = mt.StructuredMesh([1,1],0.5)
    ele = mesh.get_element(0)
    assert np.allclose(ele.get_nodes()[0].vals,(0.,0.))
    assert np.allclose(ele.get_nodes()[1].vals,(0.,0.5))
    assert np.allclose(ele.get_nodes()[2].vals,(0.5,0.5))
    assert ele.is_bdy_element() == True ; assert ele.get_area() == 0.125
    ele = mesh.get_element(1)
    assert np.allclose(ele.get_nodes()[0].vals,(0.,0.))
    assert np.allclose(ele.get_nodes()[1].vals,(0.5,0.5))
    assert np.allclose(ele.get_nodes()[2].vals,(0.5,0.))
    assert ele.is_bdy_element() == True ; assert ele.get_area() == 0.125
    ele = mesh.get_element(2)
    assert np.allclose(ele.get_nodes()[0].vals,(0.5,0.))
    assert np.allclose(ele.get_nodes()[1].vals,(0.5,0.5))
    assert np.allclose(ele.get_nodes()[2].vals,(1.0,0.5))
    assert ele.is_bdy_element() == True ; assert ele.get_area() == 0.125 
    ele = mesh.get_element(3)
    assert np.allclose(ele.get_nodes()[0].vals,(0.5,0.))
    assert np.allclose(ele.get_nodes()[1].vals,(1.0,0.5))
    assert np.allclose(ele.get_nodes()[2].vals,(1.0,0.))
    assert ele.is_bdy_element() == True ; assert ele.get_area() == 0.125
