import pytest
import numpy as np

from falcon import function_tools as ft

def test_create_product_func_1():
    f1 = lambda x,y : 0.
    f2 = lambda x,y : x**2 + y**2
    operator = ft.Operators
    f_new = operator.create_product_func(f1,f2)
    assert f_new(0.,1.) == 0. ; assert f_new(1.,0.) == 0.
    assert f_new(1.,1.) == 0. ; assert f_new(0.,0.) == 0.
    assert f_new(0,-1.) == 0. ; assert f_new(1.,-1.) == 0.

def test_create_sum_func_1():
    f1 = lambda x,y : 0.
    f2 = lambda x,y : x**2 + y**2
    operator = ft.Operators
    f_new = operator.create_sum_func(f1,f2)
    assert f_new(0.,1.) == 1. ; assert f_new(1.,0.) == 1.
    assert f_new(1.,1.) == 2. ; assert f_new(0.,0.) == 0.
    assert f_new(0,-1.) == 1. ; assert f_new(1.,-1.) == 2.

def test_create_dot_product_func_1():
    f1 = [lambda x, y: 1.,
          lambda x,y:  0.]
    f2 = [lambda x, y : x**2 + y**2,
          lambda x, y : x**2 + y**2]
    operator = ft.Operators
    f_new = operator.create_dot_product_func(f1,f2)
    assert f_new(0.,1.) == 1. ; assert f_new(1.,0.) == 1.
    assert f_new(1.,1.) == 2. ; assert f_new(0.,0.) == 0.
    assert f_new(0,-1.) == 1. ; assert f_new(1.,-1.) == 2.

def test_create_scalar_vector_div_func_1():
    s1 = lambda x,y : 1. ; grad_s1 = [lambda x,y : 0., lambda x,y : 0.]
    v1 = [lambda x,y : x, lambda x,y : y] ; div_v1 = lambda x,y : 2.
    operator = ft.Operators
    f_new = operator.create_scalar_vector_div_func(s1,
                                                   grad_s1,
                                                   v1,
                                                   div_v1)
    assert f_new(1.,1.) == 2

def test_create_scalar_vector_div_func_2():
    s1 = lambda x,y : x*y**2
    grad_s1 = [lambda x,y : y**2, lambda x,y : 2*x*y]
    v1 = [lambda x,y : x**2 + y**2, lambda x,y : -y**2+2]
    div_v1 = lambda x,y : 2*x - 2.*y
    operator = ft.Operators
    f_new = operator.create_scalar_vector_div_func(s1,
                                                   grad_s1,
                                                   v1,
                                                   div_v1)
    assert f_new(1.,1.) == 4
