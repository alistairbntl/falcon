import math
import pytest
import numpy as np

from .context import falcon
from falcon import mesh_tools as mt
from falcon import mapping_tools as mp
from falcon import quadrature as quad
from falcon import bdm_basis as bdm
from falcon import dof_handler as dof
from falcon import linalg_tools as la
from falcon import function_tools as ft

def test_quad_on_element_1(basic_mesh):
    mesh = basic_mesh
    element_itr = mesh.get_element_iterator()
    quadrature = quad.Quadrature(1)
    for i,mesh_element in enumerate(element_itr):
        mapping = mp.ReferenceElementMap(mesh_element)
        ele_quad = quadrature.get_quad_pts_on_element(mapping)
        if i==0:
            assert abs(ele_quad[0][0]-1./3.) < 1.0E-12
            assert abs(ele_quad[0][1]-1./3.) < 1.0E-12
            assert abs(ele_quad[0].get_quad_weight() - 0.5) < 1.0E-12
        if i==1:
            assert abs(ele_quad[0][0]-2./3.) < 1.0E-12
            assert abs(ele_quad[0][1]-2./3.) < 1.0E-12
            assert abs(ele_quad[0].get_quad_weight() - 0.5) < 1.0E-12    

def test_quad_on_element_2(basic_mesh):
    mesh = basic_mesh
    element_itr = mesh.get_element_iterator()
    quadrature = quad.Quadrature(2)
    for i,mesh_element in enumerate(element_itr):
        mapping = mp.ReferenceElementMap(mesh_element)
        ele_quad = quadrature.get_quad_pts_on_element(mapping)
        if i==0:
            assert abs(ele_quad[0][0]-1./3.) < 1.0E-12
            assert abs(ele_quad[0][1]-1./3.) < 1.0E-12
            assert abs(ele_quad[0].get_quad_weight() + 27./96.) < 1.0E-12
        if i==1:
            assert abs(ele_quad[0][0]-2./3.) < 1.0E-12
            assert abs(ele_quad[0][1]-2./3.) < 1.0E-12
            assert abs(ele_quad[0].get_quad_weight() + 27./96.) < 1.0E-1

def test_bdy_quad_1():
    quadrature = quad.Quadrature(3)
    assert abs(quadrature.edge_quad_pt[0] - (0.5 - math.sqrt(15) / 10.)) < 1.0E-12
    assert abs(quadrature.edge_quad_pt[1] - 0.5) < 1.0E-12
    assert abs(quadrature.edge_quad_pt[2] - (0.5 + math.sqrt(15) / 10.)) < 1.0E-12    

def test_quad_weight():
    quadrature = quad.Quadrature(2)
    quad_pts = quadrature.get_element_quad_pts()
    vol = 0.
    for pt in quad_pts:
        vol += pt.get_quad_weight()
    assert vol == 0.5
