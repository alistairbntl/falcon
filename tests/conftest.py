import pytest

from .context import falcon
from falcon import mesh_tools as mt
from falcon import mapping_tools as mp
from falcon import quadrature as quad
from falcon import bdm_basis as bdm
from falcon import dof_handler as dof
from falcon import linalg_tools as la

@pytest.fixture(scope='module')
def basic_mesh():
    test_mesh = 'test_mesh.1'
    mesh = mt.Mesh(test_mesh)
    yield mesh

@pytest.fixture(scope='module')
def basic_mesh_2():
    test_mesh = 'test_mesh.3'
    mesh = mt.Mesh(test_mesh)
    yield mesh

@pytest.fixture(scope='module')
def basic_mesh_bdm():
    test_mesh = 'test_mesh.1'
    mesh = mt.Mesh(test_mesh)
    bdm_basis = bdm.BDMBasis(1)
    dof_handler = dof.DOFHandler(mesh,
                                 [bdm_basis])
    yield bdm_basis, mesh, dof_handler

@pytest.fixture(scope='module')
def basic_mesh_bdm1():
    test_mesh = 'test_mesh.1'
    mesh = mt.Mesh(test_mesh)
    bdm_basis = bdm.BDMBasis(1)
    dof_handler = dof.DOFHandler(mesh,
                                 [bdm_basis])
    yield dof_handler

@pytest.fixture(scope='module')
def basic_mesh_bdm2():
    test_mesh = 'test_mesh.1'
    mesh = mt.Mesh(test_mesh)
    bdm_basis = bdm.BDMBasis(2)
    dof_handler = dof.DOFHandler(mesh,
                                 [bdm_basis])
    yield dof_handler
    
@pytest.fixture(scope='module')
def basic_mesh_p1():
    test_mesh = 'test_mesh.1'
    mesh = mt.Mesh(test_mesh)
    p1_basis = bdm.P1Basis_2D()
    dof_handler = dof.DOFHandler(mesh,
                                 [p1_basis])
    yield dof_handler

@pytest.fixture(scope='module')
def basis_mesh_bdm1_p0():
    test_mesh = 'test_mesh.1'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(1), bdm.P0Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def basis_mesh_bdm1tens_p0vec():
    test_mesh = 'test_mesh.1'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMTensBasis(1), bdm.P0VecBasis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield dof_handler
    
@pytest.fixture(scope='module')
def basis_mesh_bdm2_p1():
    test_mesh = 'test_mesh.1'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(2), bdm.P1Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield dof_handler
    
@pytest.fixture(scope='module')
def basic_mesh_bdm_v2():
    test_mesh = 'test_mesh.1'
    mesh = mt.Mesh(test_mesh)
    bdm_basis = bdm.BDMBasis(1)
    dof_handler = dof.DOFHandler(mesh,[bdm_basis])
    reference_element = mt.ReferenceElement()
    quadrature = quad.Quadrature(1)    
    yield bdm_basis, mesh, dof_handler, reference_element, quadrature

@pytest.fixture(scope='module')
def basic_mesh_P0():
    test_mesh = 'test_mesh.1'
    mesh = mt.Mesh(test_mesh)
    P0_basis = [bdm.P0Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 P0_basis)
    yield P0_basis, mesh, dof_handler

@pytest.fixture(scope='module')
def basic_mesh_2_P0():
    test_mesh = 'test_mesh.3'
    mesh = mt.Mesh(test_mesh)
    P0_basis = bdm.P0Basis_2D()
    dof_handler = dof.DOFHandler(mesh,
                                 [P0_basis])
    yield P0_basis, mesh, dof_handler

@pytest.fixture(scope='module')
def basic_mesh_2_BDM1():
    test_mesh = 'test_mesh.3'
    mesh = mt.Mesh(test_mesh)
    BDM1_basis = bdm.BDMBasis(1)
    dof_handler = dof.DOFHandler(mesh,
                                 [BDM1_basis])
    yield BDM1_basis, mesh, dof_handler
    
@pytest.fixture(scope='module')
def basic_mesh_bdm1_p0():
    test_mesh = 'test_mesh.1'
#    test_mesh = 'refine_mesh.2'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(1), bdm.P0Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler
    
@pytest.fixture(scope='module')
def basic_mesh_s1_p1():
    test_mesh = 'test_struc_mesh.2'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.P1Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def basic_mesh_s1_bdm1_partial_r1():
    test_mesh = 'test_mesh.1'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(1)]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def basic_mesh_s1_bdm2_partial_r1():
    test_mesh = 'test_mesh.1'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(2)]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def basic_mesh_s1_bdm1_partial_r2():
    test_mesh = 'refine_mesh.2'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(1)]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def basic_mesh_s1_bdm2_partial_r2():
    test_mesh = 'refine_mesh.2'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(2)]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def basic_mesh_s1_bdm1_p0():
    test_mesh = 'test_struc_mesh.2'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(1), bdm.P0Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def basic_mesh_s1_bdm2_p1():
    test_mesh = 'test_struc_mesh.2'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(2), bdm.P1Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def basic_mesh_s1_bdm1_p0_tmp1():
    test_mesh = 'refine_mesh.3'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(1), bdm.P0Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def basic_mesh_s1_bdm1_p0_tmp():
    test_mesh = 'refine_mesh.1'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(1), bdm.P0Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler
    
@pytest.fixture(scope="module")
def simple_element():
    reference_element = mt.ReferenceElement()
    # create a triangle that is not the reference triangle
    p0 = mt.Point(0.,0.); p1 = mt.Point(4.,0.) ; p2 = mt.Point(2.,4.)
    coords = [p0,p1,p2]
    element = mt.Element(coords)
    quadrature = quad.Quadrature(1)
    quad_pt = quadrature.edge_quad_pt
    quad_wght = quadrature.edge_quad_wght
    yield (reference_element, element, quadrature)

@pytest.fixture(scope="module")
def simple_element_orientation1():
    reference_element = mt.ReferenceElement()
    # create a triangle that is not the reference triangle
    p1 = mt.Point(0.,0.); p2 = mt.Point(4.,0.) ; p0 = mt.Point(2.,4.)
    coords = [p0,p1,p2]
    element = mt.Element(coords)
    quadrature = quad.Quadrature(1)
    quad_pt = quadrature.edge_quad_pt
    quad_wght = quadrature.edge_quad_wght
    yield (reference_element, element, quadrature)

@pytest.fixture(scope="module")
def bdm_edge_tests():
    BDM1_basis = bdm.BDMBasis(1)
    quadrature = quad.Quadrature(1)
    reference_element = mt.ReferenceElement()
    yield (BDM1_basis,quadrature,reference_element)

@pytest.fixture(scope="module")
def bdm2_edge_tests():
    BDM2_basis = bdm.BDMBasis(2)
    quadrature = quad.Quadrature(3)
    reference_element = mt.ReferenceElement()
    yield (BDM2_basis,quadrature,reference_element)
    
@pytest.fixture(scope='module')
def basic_aximesh_bdm1_p0():
    test_mesh = 'axidarcy_meshes/test_struc_mesh.2'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(1), bdm.P0Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def basic_mesh_p0vec():
    test_mesh = 'test_mesh.1'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.P0VecBasis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler
    
@pytest.fixture(scope='module')
def basic_mesh_bdm1_p0_elasticity():
    test_mesh = 'test_mesh.1'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMTensBasis(1),
             bdm.P0VecBasis_2D(),
             bdm.P0SkewTensBasis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def elasticity_bdm1_p0_structured():
    mesh = mt.StructuredMesh([1.,1.],1./10)
    basis = [bdm.BDMTensBasis(1),
             bdm.P0VecBasis_2D(),
             bdm.P0SkewTensBasis_2D()]         
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def elasticity_bdm1_partial():
    mesh = mt.StructuredMesh([1.,1.],1./8)
    basis = [bdm.BDMTensBasis(1),
             bdm.P0SkewTensBasis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def elasticity_bdm2_partial():
    mesh = mt.StructuredMesh([1.,1.],1.)
    basis = [bdm.BDMTensBasis(2),
             bdm.P1SkewTensBasis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def elasticity_bdm2_partial_one_element():
    mesh = mt.OneElementMesh()
    basis = [bdm.BDMTensBasis(2),
             bdm.P1SkewTensBasis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def elasticity_bdm2():
    mesh = mt.StructuredMesh([1.,1.],1./6)
    basis = [bdm.BDMTensBasis(2),
             bdm.P1VecBasis_2D(),
             bdm.P1SkewTensBasis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler
    
@pytest.fixture(scope='module')
def elasticity_bdm2_one_element():
    mesh = mt.OneElementMesh()
    basis = [bdm.BDMTensBasis(2),
             bdm.P1VecBasis_2D(),
             bdm.P1SkewTensBasis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler
    
@pytest.fixture(scope='module')
def darcy_bdm2_partial_one_element():
    mesh = mt.OneElementMesh()
    basis = [bdm.BDMBasis(2),
             bdm.P1Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler
    
@pytest.fixture(scope='module')
def basic_mesh2_bdm1_p0_elasticity():
    test_mesh = 'convergence_test_meshes/convergence_mesh_2'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMTensBasis(1),
             bdm.P0VecBasis_2D(),
             bdm.P0SkewTensBasis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def basic_mesh2_bdm1_partial_elasticity():
    test_mesh = 'convergence_test_meshes/convergence_mesh_5'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMTensBasis(1),
             bdm.P0SkewTensBasis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def darcy_bdm2_p1_converge_2():
    test_mesh = 'convergence_test_meshes/new_convergence_mesh.1'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(2), bdm.P1Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def darcy_bdm2_p1_structured_converge_1():
    mesh = mt.StructuredMesh([1.,1.],1./12)
    basis = [bdm.BDMBasis(2), bdm.P1Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def darcy_bdm1_p0_structured_converge_1():
    mesh = mt.StructuredMesh([1.,1.],1./12)
    basis = [bdm.BDMBasis(1), bdm.P0Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def darcy_bdm1_p0_converge_1():
    test_mesh = 'convergence_test_meshes/new_convergence_mesh.1'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(1), bdm.P0Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def darcy_bdm1_p0_converge_2():
    test_mesh = 'convergence_test_meshes/new_convergence_mesh.2'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(1), bdm.P0Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def darcy_bdm1_p0_converge_3():
    test_mesh = 'convergence_test_meshes/new_convergence_mesh.3'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(1), bdm.P0Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def darcy_bdm1_p0_converge_4():
    test_mesh = 'convergence_test_meshes/new_convergence_mesh.4'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(1), bdm.P0Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler

@pytest.fixture(scope='module')
def darcy_bdm1_p0_converge_5():
    test_mesh = 'convergence_test_meshes/new_convergence_mesh.5'
    mesh = mt.Mesh(test_mesh)
    basis = [bdm.BDMBasis(1), bdm.P0Basis_2D()]
    dof_handler = dof.DOFHandler(mesh,
                                 basis)
    yield basis, mesh, dof_handler
