import math
import pytest
import numpy as np

from .context import falcon
from falcon import bdm_basis as bdm
from falcon import mesh_tools as mt
from falcon import mapping_tools as mpt
from falcon import quadrature as quad
from falcon import linalg_tools as la

@pytest.mark.bdm
def test_bdm_normal_edge_1(bdm_edge_tests):
    BDM1_basis, quadrature, reference_element = bdm_edge_tests
    for edge_0 in range(3):
        for edge_1 in range(3):
            func_0 = BDM1_basis.get_edge_normal_func(edge_1,edge_0,0)
            func_1 = BDM1_basis.get_edge_normal_func(edge_1,edge_0,1)
            for idx in range(2):
                pt = reference_element.get_lagrange_quad_point(edge_1,idx)
                if edge_0 == edge_1:
                    if idx==0:
                        assert func_0(pt[0],pt[1]) == 1.0
                        assert func_1(pt[0],pt[1]) == 0.0
                    if idx==1:
                        assert func_0(pt[0],pt[1]) == 0.0
                        assert func_1(pt[0],pt[1]) == 1.0
                else:
                    assert abs(func_0(pt[0],pt[1])) <= 1.0e-12
                    assert abs(func_1(pt[0],pt[1])) <= 1.0e-12

@pytest.mark.bdm
def test_bdm2_normal_edge_1(bdm2_edge_tests):
    BDM2_basis, quadrature, reference_element = bdm2_edge_tests
    reference_element.set_edge_quad_pts(quadrature)
    int_funcs = BDM2_basis.interior_functions.int_funcs
    for edge_0 in range(3):
        for edge_1 in range(3):
            func_0 = BDM2_basis.get_edge_normal_func(edge_1,edge_0,0)
            func_1 = BDM2_basis.get_edge_normal_func(edge_1,edge_0,1)
            func_2 = BDM2_basis.get_edge_normal_func(edge_1,edge_0,2)
            int_func_0 = BDM2_basis.get_edge_normal_func(edge_1,'interior',0)
            int_func_1 = BDM2_basis.get_edge_normal_func(edge_1,'interior',1)
            int_func_2 = BDM2_basis.get_edge_normal_func(edge_1,'interior',2)
            for idx in range(3):
                pt = reference_element.get_lagrange_quad_point(edge_1,idx)
                assert abs(int_func_0(pt[0],pt[1])-0.) <= 1.0e-12
                assert abs(int_func_1(pt[0],pt[1])-0.) <= 1.0e-12
                assert abs(int_func_2(pt[0],pt[1])-0.) <= 1.0e-12                
                if edge_0 == edge_1 and edge_0==0:
                    if idx==0:
                        assert abs(func_0(pt[0],pt[1])-1.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-0.0) <= 1.0e-12
                    if idx==1:
                        assert abs(func_0(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-1.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-0.) <= 1.0e-12
                    if idx==2:
                        assert abs(func_0(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-1.) <= 1.0e-12
                elif edge_0 == edge_1 and edge_0==1:
                    if idx==0:
                        assert abs(func_0(pt[0],pt[1])-1.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-0.) <= 1.0e-12
                    if idx==1:
                        assert abs(func_0(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-1.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-0.) <= 1.0e-12                        
                    if idx==2:
                        assert abs(func_0(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-1.) <= 1.0e-12
                elif edge_0 == edge_1 and edge_0 == 2:
                    if idx==0:
                        assert abs(func_0(pt[0],pt[1])-1.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-0.) <= 1.0e-12
                    if idx==1:
                        assert abs(func_0(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-1.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-0.) <= 1.0e-12                        
                    if idx==2:
                        assert abs(func_0(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-1.) <= 1.0e-12
                else:
                    assert abs(func_0(pt[0],pt[1])) <= 1.0e-12
                    assert abs(func_1(pt[0],pt[1])) <= 1.0e-12

@pytest.mark.bdm
def test_bdm_val_cal_1():
    reference_element = mt.ReferenceElement()
    bdm_basis = bdm.BDMBasis(1)
    mapping = mpt.PiolaMap(reference_element)
    quadrature = quad.Quadrature(1)
    val_types = ['vals','Jt_vals','|Jt|','div']
    for i in range(bdm_basis.get_num_dof()):
        for pt in quadrature.get_element_quad_pts():
            test = bdm_basis.get_element_vals(i,
                                              pt,
                                              mapping,
                                              val_types)
            if i == 0 or 3:
                test['div'] == math.sqrt(2)
            else:
                test['div'] == 1.0
            assert np.array_equal(test['vals'], test['Jt_vals'])
            assert test['|Jt|'] == 1.0

@pytest.mark.bdm
def test_bdm_edge_length_1(basic_mesh_2_BDM1):
    BDM1_basis, mesh, dof_handler = basic_mesh_2_BDM1
    reference_element = mt.ReferenceElement()
    quadrature = quad.Quadrature(1)                                      

    E0 = mesh.get_element(0)
    piola_map = mpt.PiolaMap(E0)
    mesh_edge_1 = E0.get_edge(0)

    val_types = ['vals']
    pt1 = reference_element.get_lagrange_quad_point(0,0)
    vals = BDM1_basis.get_element_vals(0,
                                       pt1,
                                       piola_map,
                                       val_types)

    vals = piola_map.correct_div_space_vals(vals,
                                            0,
                                            BDM1_basis)
    
    func_vals = piola_map.apply_piola_map(vals)
    norm_val = la.Operators.dot_product(func_vals,
                                        mesh_edge_1.get_unit_normal_vec())

    
    E2 = mesh.get_element(2)
    piola_map_2 = mpt.PiolaMap(E2)
    mesh_edge_2 = E2.get_edge(2)

    pt2 = reference_element.get_lagrange_quad_point(2,0)
    vals = BDM1_basis.get_element_vals(2,
                                       pt2,
                                       piola_map_2,
                                       val_types)

    vals = piola_map_2.correct_div_space_vals(vals,
                                              2,
                                              BDM1_basis)

    func_vals = piola_map_2.apply_piola_map(vals,xi=pt2[0],eta=pt2[1])
    norm_val_1 = la.Operators.dot_product(func_vals,
                                          mesh_edge_2.get_unit_normal_vec())

    assert abs(norm_val-norm_val_1) < 1.e-8

@pytest.mark.bdm
def test_p1basis_2d():
    # build test on reference element
    reference_element = mt.ReferenceElement()
    P1_basis = bdm.P1Basis_2D()
    mapping = mpt.ReferenceElementMap(reference_element)
    quadrature = quad.Quadrature(1)
    val_types = ['vals','dvals']
    for i in range(P1_basis.get_num_dof()):
        for pt in quadrature.get_element_quad_pts():
            test = P1_basis.get_element_vals(i,
                                             pt,
                                             mapping,
                                             val_types)
            assert (test['vals']-1./3.) < 1e-8
            if i==0:
                np.array_equal(test['dvals'],(-1.0,-1.0))
            elif i==1:
                np.array_equal(test['dvals'],(0.,1.))
            elif i==2:
                np.array_equal(test['dvals'],(1.,0.))
    # build test on non-reference element
    p1 = mt.Point(1.,0.) ; p2 = mt.Point(1.,1.) ; p3 = mt.Point(0.5,0.5)
    element = mt.Element([p1,p2,p3])
    mapping = mpt.ReferenceElementMap(element)
    for i in range(P1_basis.get_num_dof()):
        for pt in quadrature.get_element_quad_pts():
            test = P1_basis.get_element_vals(i,
                                             pt,
                                             mapping,
                                             val_types)
            assert (test['vals']-1./3.) < 1e-8
            if i==0:
                np.array_equal(test['dvals'],(-1.,-1.))
            elif i==1:
                np.array_equal(test['dvals'],(0.,-2.))
            elif i==2:
                np.array_equal(test['dvals'],(1.,1.))

@pytest.mark.p2basis
def test_p2basis_2d():
    # build test on reference element
    reference_element = mt.ReferenceElement()
    P2_basis = bdm.P2Basis_2D()
    mapping = mpt.ReferenceElementMap(reference_element)
    p1 = mt.QuadraturePoint(0.5, 0.5, 1.) ; p2 = mt.QuadraturePoint(0., 0.5, 1.)
    p3 = mt.QuadraturePoint(0.5, 0.0, 1.) ; p4 = mt.QuadraturePoint(0., 0.0, 1.)
    p5 = mt.QuadraturePoint(1.0, 0.0, 1.) ; p6 = mt.QuadraturePoint(0., 1.0, 1.)
    quad_pts = [p1, p2, p3, p4, p5, p6]
    val_types = ['vals']
    for i in range(P2_basis.get_num_dof()):
        for j, pt in enumerate(quad_pts):
            test = P2_basis.get_element_vals(i,
                                             pt,
                                             mapping,
                                             val_types)
            if i==j:
                assert (test['vals']-1.) < 1.e-8
            else:
                assert (test['vals']-0.) < 1.e-8
    # TODO - add gradients e.g. :

    #         if i==0:
    #             np.array_equal(test['dvals'],(-1.,-1.))
    #         elif i==1:
    #             np.array_equal(test['dvals'],(0.,-2.))
    #         elif i==2:
    #             np.array_equal(test['dvals'],(1.,1.))


@pytest.mark.bdm
def test_bdm_tens_basis_1():
    bdm_tens_basis = bdm.BDMTensBasis(1)
    assert bdm_tens_basis.get_num_dof() == 12
    assert bdm_tens_basis.get_num_interior_dof() == 0
    assert bdm_tens_basis.get_num_dof_per_edge() == 4
    assert bdm_tens_basis.get_num_edge_dof() == 12
    assert bdm_tens_basis.get_num_dof_per_node() == 0

@pytest.mark.bdm
def test_bdm_tens_basis_2():
    bdm_tens_basis = bdm.BDMTensBasis(1) ; bdm_basis = bdm.BDMBasis(1)

    p1 = mt.Point(1.,0.) ; p2 = mt.Point(1.,1.) ; p3 = mt.Point(0.5,0.5)
    element = mt.Element([p1,p2,p3])
    mapping = mpt.ReferenceElementMap(element)
    quadrature = quad.Quadrature(1)    
    val_types = ['vals','Jt_vals','div']
    
    for i in range(bdm_basis.get_num_dof()):
        j = i / 3
        k = i % 3
        for pt in quadrature.get_element_quad_pts():
            vals = bdm_basis.get_element_vals(i,
                                              pt,
                                              mapping,
                                              val_types)
            vals_tens_1 = bdm_tens_basis.get_element_vals(6*j + k,
                                                          pt,
                                                          mapping,
                                                          val_types)
            vals_tens_2 = bdm_tens_basis.get_element_vals(6*j + k + 3,
                                                          pt,
                                                          mapping,
                                                          val_types)
            assert np.array_equal(vals['vals'],vals_tens_1['vals'][0])
            assert np.array_equal(vals['Jt_vals'],vals_tens_1['Jt_vals'][0])
            assert np.array_equal(vals['vals'],vals_tens_2['vals'][1])
            assert np.array_equal(vals['Jt_vals'],vals_tens_2['Jt_vals'][1])
            assert np.array_equal(vals['div'], vals_tens_1['div'][0])
            assert np.array_equal(vals['div'], vals_tens_2['div'][1])            

@pytest.mark.bdm
def test_bdm2_div_1(bdm2_edge_tests):
    BDM2_basis, quadrature, reference_element = bdm2_edge_tests
    edge_div_funcs = BDM2_basis.edge_functions.edge_div_funcs
    assert abs(edge_div_funcs[0][0](1./3,1./3) - 0.7856742) < 1.0E-8
    assert edge_div_funcs[0][0](1.,0,) < 1.0E-8
    assert abs(edge_div_funcs[0][1](1./3,1./3) - 1.2570787221) < 1.0E-8
    assert abs(edge_div_funcs[0][2](1./3,1./3) - 0.7856742) < 1.0E-8
    assert abs(edge_div_funcs[0][2](.2,.6) - 2.874807049) < 1.0E-8
    assert abs(edge_div_funcs[1][0](1./3,1./3) - 5./9) < 1.0E-8
    assert abs(edge_div_funcs[1][0](0.2, 0.6) - 1.0) < 1.0E-8
    assert abs(edge_div_funcs[1][1](1./3, 1./3) - 8./9) < 1.0E-8
    assert abs(edge_div_funcs[1][1](0.2, 0.6) - 2.065591117978) < 1.0E-8
    assert abs(edge_div_funcs[1][2](1./3,1./3) - 5./9) < 1.0E-8
    assert abs(edge_div_funcs[1][2](0.2, 0.6) - 2.032795558988) < 1.0E-8
    assert abs(edge_div_funcs[2][0](1./3,1./3) - 5./9) < 1.0E-8
    assert abs(edge_div_funcs[2][0](.2, .6) - 1./3) < 1.0E-8
    assert abs(edge_div_funcs[2][1](1./3,1./3) - 8./9) < 1.0E-8
    assert abs(edge_div_funcs[2][1](.2,.6) - 0.3005377743) < 1.0E-8
    assert abs(edge_div_funcs[2][2](1./3,1./3) - 5./9) < 1.0E-8
    assert abs(edge_div_funcs[2][2](.2,.6) + 0.18306444616) < 1.0E-8    
            
@pytest.mark.bdm
def test_P0_vec_basis_1():
    p0_vec_test_basis = bdm.P0VecBasis_2D()
    assert p0_vec_test_basis.get_num_dof() == 2
    assert p0_vec_test_basis.get_num_interior_dof() == 2
    assert p0_vec_test_basis.get_num_dof_per_edge() == 0
    assert p0_vec_test_basis.get_num_dof_per_node() == 0

@pytest.mark.bdm
def test_P0_vec_basis_2():
    p0_vec_test_basis = bdm.P0VecBasis_2D() ; p0_basis = bdm.P0Basis_2D()

    p1 = mt.Point(1.,0.) ; p2 = mt.Point(1.,1.) ; p3 = mt.Point(0.5,0.5)
    element = mt.Element([p1,p2,p3])
    mapping = mpt.ReferenceElementMap(element)
    quadrature = quad.Quadrature(1)    
    val_types = ['vals']

    for i in range(p0_basis.get_num_dof()):
        for pt in quadrature.get_element_quad_pts():
            vals = p0_basis.get_element_vals(i,
                                             pt,
                                             mapping,
                                             val_types)
            vals_vec_1 = p0_vec_test_basis.get_element_vals(2*i,
                                                            pt,
                                                            mapping,
                                                            val_types)
            vals_vec_2 = p0_vec_test_basis.get_element_vals(2*i + 1,
                                                            pt,
                                                            mapping,
                                                            val_types)
            assert np.array_equal(vals['vals'], vals_vec_1['vals'][0])
            assert np.array_equal(vals['vals'], vals_vec_2['vals'][1])            

@pytest.mark.bdm
def test_P0_skewtens_basis_1():
    p0_skewtens_basis = bdm.P0SkewTensBasis_2D()
    assert p0_skewtens_basis.get_num_dof() == 1
    assert p0_skewtens_basis.get_num_interior_dof() == 1
    assert p0_skewtens_basis.get_num_dof_per_edge() == 0
    assert p0_skewtens_basis.get_num_dof_per_node() == 0

@pytest.mark.bdm
def test_P0_vec_basis_2():
    p0_vec_test_basis = bdm.P0SkewTensBasis_2D() ; p0_basis = bdm.P0Basis_2D()

    p1 = mt.Point(1.,0.) ; p2 = mt.Point(1.,1.) ; p3 = mt.Point(0.5,0.5)
    element = mt.Element([p1,p2,p3])
    mapping = mpt.ReferenceElementMap(element)
    quadrature = quad.Quadrature(1)    
    val_types = ['vals']

    for i in range(p0_basis.get_num_dof()):
        for pt in quadrature.get_element_quad_pts():
            vals = p0_basis.get_element_vals(i,
                                             pt,
                                             mapping,
                                             val_types)
            vals_vec_1 = p0_vec_test_basis.get_element_vals(i,
                                                            pt,
                                                            mapping,
                                                            val_types)
            assert np.array_equal(vals['vals'], -1*vals_vec_1['vals'][0][1])
            assert np.array_equal(vals['vals'], vals_vec_1['vals'][1][0])

def test_P0_vec_basis_2():
    p1_vec_test_basis = bdm.P1SkewTensBasis_2D() ; p1_basis = bdm.P1Basis_2D()

    p1 = mt.Point(1.,0.) ; p2 = mt.Point(1.,1.) ; p3 = mt.Point(0.5,0.5)
    element = mt.Element([p1,p2,p3])
    mapping = mpt.ReferenceElementMap(element)
    quadrature = quad.Quadrature(1)    
    val_types = ['vals']

    for i in range(p1_basis.get_num_dof()):
        for pt in quadrature.get_element_quad_pts():
            vals = p1_basis.get_element_vals(i,
                                             pt,
                                             mapping,
                                             val_types)
            vals_vec_1 = p1_vec_test_basis.get_element_vals(i,
                                                            pt,
                                                            mapping,
                                                            val_types)
            assert abs(vals['vals'] + vals_vec_1['vals'][0][1]) < 1e-12
            assert abs(vals['vals'] - vals_vec_1['vals'][1][0]) < 1e-12

def test_BDM2_tens_basis_1():
    bdm2_tens_basis = bdm.BDMTensBasis(2) ; bdm2_basis = bdm.BDMBasis(2)
    
    p1 = mt.Point(1.,0.) ; p2 = mt.Point(1.,1.) ; p3 = mt.Point(0.5,0.5)
    element = mt.Element([p1,p2,p3])
    mapping = mpt.ReferenceElementMap(element)
    quadrature = quad.Quadrature(1)    
    val_types = ['vals']

    assert bdm2_tens_basis.get_num_dof() == 24

    for i in range(bdm2_basis.get_num_dof()):
        j = i / 3 ; k = i % 3
        for pt in quadrature.get_element_quad_pts():
            vals = bdm2_basis.get_element_vals(i,
                                               pt,
                                               mapping,
                                               val_types)
            vals_tens_1 = bdm2_tens_basis.get_element_vals(6*j+k,
                                                           pt,
                                                           mapping,
                                                           val_types)
            vals_tens_2 = bdm2_tens_basis.get_element_vals(6*j+k + 3,
                                                           pt,
                                                           mapping,
                                                           val_types)
            assert np.array_equal(vals['vals'], vals_tens_1['vals'][0])
            assert np.array_equal(vals['vals'], vals_tens_2['vals'][1])            

@pytest.mark.bdm3
def test_BDM3_basis_1():
    bdm3_basis = bdm.BDMBasis(3)
    
    p1 = mt.Point(1.,0.) ; p2 = mt.Point(1.,1.) ; p3 = mt.Point(0.5,0.5)
    element = mt.Element([p1,p2,p3])
    mapping = mpt.ReferenceElementMap(element)
    quadrature = quad.Quadrature(5)
    val_types = ['vals']

    assert bdm3_basis.get_num_dof() == 20

    pt = quadrature._element_quad_pts[0]
    edge_pts = quadrature.edge_quad_pt

    def t_quad_pt(quad_pt):
        vals = bdm3_basis.get_element_vals(18, quad_pt, mapping, val_types)
        assert la.Operators.l2_norm(vals['vals']) <= 1.0e-12
        vals = bdm3_basis.get_element_vals(19, quad_pt, mapping, val_types)
        assert la.Operators.l2_norm(vals['vals']) <= 1.0e-12

    # test bubble functions vanish on edges
    for pt in edge_pts:
        x = 1- pt ; y = pt
        quad_pt = mt.QuadraturePoint(x,y)
        t_quad_pt(quad_pt)
        x = 0. ; y = pt
        quad_pt = mt.QuadraturePoint(x,y)
        t_quad_pt(quad_pt)
        x = pt ; y = 0.
        quad_pt = mt.QuadraturePoint(x,y)
        t_quad_pt(quad_pt)
#    for i in range(bdm3_basis.get_num_dof()):
#        j = i / 3 ; k = i % 3
#        for pt in quadrature.get_element_quad_pts():
#            vals = bdm2_basis.get_element_vals(i,
#                                               pt,
#                                               mapping,
#                                               val_types)

@pytest.mark.bdm3
def test_BDM3_basis_2(bdm3_edge_tests):
    bdm3_basis, quadrature, reference_element = bdm3_edge_tests
    reference_element.set_edge_quad_pts(quadrature)
    for edge_0 in range(3):
        for edge_1 in range(3):
            func_0 = bdm3_basis.get_edge_normal_func(edge_1,edge_0,0)
            func_1 = bdm3_basis.get_edge_normal_func(edge_1,edge_0,1)
            func_2 = bdm3_basis.get_edge_normal_func(edge_1,edge_0,2)
            func_3 = bdm3_basis.get_edge_normal_func(edge_1,edge_0,3)
            for idx in range(4):
                pt = reference_element.get_lagrange_quad_point(edge_1,idx)
                for int_func_idx in range(8):
                    int_func_normal = bdm3_basis.get_edge_normal_func(edge_1,'interior',int_func_idx)
                    assert abs(int_func_normal(pt[0],pt[1])) <= 1.0e-12
                if edge_0 == edge_1 and edge_0 == 0:
                    if idx==0:
                        assert abs(func_0(pt[0],pt[1])-1.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-0.0) <= 1.0e-12
                        assert abs(func_3(pt[0],pt[1])-0.0) <= 1.0e-12                        
                    if idx==1:
                        assert abs(func_0(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-1.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_3(pt[0],pt[1])-0.) <= 1.0e-12                        
                    if idx==2:
                        assert abs(func_0(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-1.) <= 1.0e-12
                        assert abs(func_3(pt[0],pt[1])-0.) <= 1.0e-12                        
                    if idx==3:
                        assert abs(func_0(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_3(pt[0],pt[1])-1.) <= 1.0e-12                        
                elif edge_0 == edge_1 and edge_0==1:
                    if idx==0:
                        assert abs(func_0(pt[0],pt[1])-1.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_3(pt[0],pt[1])-0.) <= 1.0e-12                        
                    if idx==1:
                        assert abs(func_0(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-1.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_3(pt[0],pt[1])-0.) <= 1.0e-12                        
                    if idx==2:
                        assert abs(func_0(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-1.) <= 1.0e-12
                        assert abs(func_3(pt[0],pt[1])-0.) <= 1.0e-12
                    if idx==3:
                        assert abs(func_0(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_3(pt[0],pt[1])-1.) <= 1.0e-12                        
                elif edge_0 == edge_1 and edge_0 == 2:
                    if idx==0:
                        assert abs(func_0(pt[0],pt[1])-1.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_3(pt[0],pt[1])-0.) <= 1.0e-12                        
                    if idx==1:
                        assert abs(func_0(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-1.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_3(pt[0],pt[1])-0.) <= 1.0e-12                        
                    if idx==2:
                        assert abs(func_0(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-1.) <= 1.0e-12
                        assert abs(func_3(pt[0],pt[1])-0.) <= 1.0e-12
                    if idx==3:
                        assert abs(func_0(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_1(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_2(pt[0],pt[1])-0.) <= 1.0e-12
                        assert abs(func_3(pt[0],pt[1])-1.) <= 1.0e-12
                else:
                    assert abs(func_0(pt[0],pt[1])) <= 1.0e-12
                    assert abs(func_1(pt[0],pt[1])) <= 1.0e-12
