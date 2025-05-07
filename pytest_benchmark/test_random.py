# cython: language_level=3
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2024, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

import pytest

import arrayfire as af
import numpy as np
import dpnp
import cupy

ROUNDS = 30
ITERATIONS = 4

NNUMBERS = 2**16
PKGS = [dpnp, np, cupy, af]
IDS = [pkg.__name__ for pkg in PKGS]

@pytest.mark.parametrize(
    "pkg", PKGS, ids=IDS
)

def setup():
        np.random.seed(1)
        dpnp.random.seed(1)
        cupy.random.seed(1)

def randn_np():
    arr = np.random.normal(size=(NNUMBERS))

def randn_dpnp():
    arr = dpnp.random.normal(size=(NNUMBERS))

def randn_cupy():
    arr = cupy.random.normal(size=(NNUMBERS))
    cupy.cuda.runtime.deviceSynchronize()

def randn_af():
    arr = af.randn((NNUMBERS))
    af.eval(arr)
    af.sync()

def randu_np():
    arr = np.random.uniform(size=(NNUMBERS))

def randu_dpnp():
    arr = dpnp.random.uniform(size=(NNUMBERS))

def randu_cupy():
    arr = cupy.random.uniform(size=(NNUMBERS))
    cupy.cuda.runtime.deviceSynchronize()

def randu_af():
    arr = af.randu((NNUMBERS))
    af.eval(arr)
    af.sync()

class TestRandom:
    def test_normal(self, benchmark, pkg):
        FUNCS = { "dpnp" : randn_dpnp , "numpy" : randn_np, \
         "cupy" : randn_cupy , "arrayfire" : randn_af }
        
        result = benchmark.pedantic(
            target=FUNCS[pkg.__name__],
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    
    def test_uniform(self, benchmark, pkg):
        FUNCS = { "dpnp" : randu_dpnp , "numpy" : randu_np, \
         "cupy" : randu_cupy , "arrayfire" : randu_af }
        
        result = benchmark.pedantic(
            target=FUNCS[pkg.__name__],
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS
        )
