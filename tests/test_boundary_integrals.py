import math
import pytest
import numpy as np
import scipy.integrate as integrate

from .context import falcon
from falcon import mesh_tools as mt
from falcon import mapping_tools as mp
from falcon import quadrature as quad
from falcon import bdm_basis as bdm
from falcon import fe_spaces as fe
from falcon import dof_handler as dof
    
def calculate_quadrature(quad,i,func):
    I = 0.
    for pt in zip(quad.edge_quad_pt,quad.edge_quad_wght):
        I += (func
              (mt.ReferenceElement.get_edge_parameterization(i)[0](pt[0]),
               mt.ReferenceElement.get_edge_parameterization(i)[1](pt[0]))*
              pt[1])
    return I

def test_int_1(simple_element):
    reference_element, element, quadrature =simple_element
    f_reference = lambda xi, eta: eta
    ints = []
    for i,edge in enumerate(reference_element.get_edges()):
        I = 0
        I+= calculate_quadrature(quadrature,
                                 i,
                                 f_reference)
        ints.append(I*element.get_edge_length(i))
    assert ints == [math.sqrt(5), math.sqrt(5), 0.]

def test_int_bdm_1(simple_element):
    reference_element, element, quadrature = simple_element
    BDM1_basis = bdm.BDMBasis(1)
    ints_0 = np.zeros(3)
    ints_1 = np.zeros(3)
    for i,edge in enumerate(reference_element.get_edges()):
        f_reference_0 = BDM1_basis.get_edge_normal_func(i,2,0)
        f_reference_1 = BDM1_basis.get_edge_normal_func(i,2,1)
        I0 = 0
        I1 = 0
        I0 += calculate_quadrature(quadrature,
                                   i,
                                   f_reference_0)
        I1 += calculate_quadrature(quadrature,
                                   i,
                                   f_reference_1)
        ints_0[i] = I0*element.get_edge_length(i)
        ints_1[i] = I1*element.get_edge_length(i)
    assert np.allclose(ints_0, np.array([0,0,2]))
    assert np.allclose(ints_1, np.array([0,0,2]))

def test_int_bdm_2(simple_element_orientation1):
    reference_element, element, quadrature = simple_element_orientation1
    BDM1_basis = bdm.BDMBasis(1)
    ints_0 = np.zeros(3)
    ints_1 = np.zeros(3)
    for i,edge in enumerate(reference_element.get_edges()):
        f_reference_0 = BDM1_basis.get_edge_normal_func(i,0,0)
        f_reference_1 = BDM1_basis.get_edge_normal_func(i,0,1)
        I0 = 0
        I1 = 0
        I0 += calculate_quadrature(quadrature,
                                   i,
                                   f_reference_0)
        I1 += calculate_quadrature(quadrature,
                                   i,
                                   f_reference_1)
        ints_0[i] = I0*element.get_edge_length(i)
        ints_1[i] = I1*element.get_edge_length(i)
    assert np.allclose(ints_0, np.array([2,0,0]))
    assert np.allclose(ints_1, np.array([2,0,0]))

def test_bdm_on_non_reference_element(basic_mesh_bdm_v2):
    """
    This is a silly test I wrote to help understand something.
    """
    bdm_basis, mesh, dof_handler, reference_element,quadrature = basic_mesh_bdm_v2

    g1 = 0.5 - math.sqrt(3) / 6.
    g2 = 0.5 + math.sqrt(3) / 6.
    phi_1 = lambda x, y : 1./(g2-g1)*((y-g1) + (1-g1)*(x-1))
    phi_2 = lambda x, y : 1./(g2-g1)*(-g1*(y-1))
    n1 = lambda x,y : 1.
    n2 = lambda x,y : 0.
    normal_vec = lambda x,y : phi_1(x,y)*n1(x,y) + phi_2(x,y)*n2(x,y)
    normal_vec_edge_1 = lambda y : normal_vec(1, y)
    I = integrate.quad(normal_vec_edge_1, 0, 1)
    
    e1 = mesh.get_element(1)
    basis_func_1 = bdm_basis.get_edge_normal_func(0,0,1)
    I0 = 0.
    I0 = calculate_quadrature(quadrature,
                              0,
                              basis_func_1)
