import math

from .context import falcon
from falcon import mapping_tools as mp
from falcon import mesh_tools as mt

def test_reference_element_1():
    p1 = mt.Point(1.,0.); p2 = mt.Point(2.,0.); p3 = mt.Point(1.,1.)
    coords = [p1,p2,p3]
    element = mt.Element(coords)
    reference_map = mp.ReferenceElementMap(element)
    assert reference_map.get_jacobian_det() == 1.0

def test_reference_element_2():
    p1 = mt.Point(1.,0.); p2 = mt.Point(4.,0.); p3 = mt.Point(1.,1.)
    coords = [p1,p2,p3]
    element = mt.Element(coords)
    reference_map = mp.ReferenceElementMap(element)
    assert reference_map.get_jacobian_det() == 3.

def test_reference_element_mesh_r1(basic_mesh_2):
    mesh = basic_mesh_2
    element = mesh.get_element(0)
    reference_map = mp.ReferenceElementMap(element)
    assert reference_map.apply_affine_map(0,0).vals == (1,0)
    assert reference_map.apply_affine_map(1,0).vals == (1,1)
    assert reference_map.apply_inverse_affine_map(1,0) == (0,0)
    assert reference_map.apply_inverse_affine_map(1,1) == (1,0)
    assert abs(reference_map.get_jacobian_det()-0.3535533) < 1.0e-6
