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

from common import *

ITERATIONS = 20

def randn_np():
    arr = np.random.normal(size=(NNSIZE))

def randn_dpnp():
    arr = dpnp.random.normal(size=(NNSIZE))

def randn_cupy():
    arr = cupy.random.normal(size=(NNSIZE))
    cupy.cuda.runtime.deviceSynchronize()

def randn_af():
    arr = af.randn((NNSIZE))
    af.eval(arr)
    af.sync()

def randu_np():
    arr = np.random.uniform(size=(NNSIZE))

def randu_dpnp():
    arr = dpnp.random.uniform(size=(NNSIZE))

def randu_cupy():
    arr = cupy.random.uniform(size=(NNSIZE))
    cupy.cuda.runtime.deviceSynchronize()

def randu_af():
    arr = af.randu((NNSIZE))
    af.eval(arr)
    af.sync()

@pytest.mark.parametrize(
    "pkgid", IDS, ids=IDS
)
class TestRandom:
    def test_normal(self, benchmark, pkgid):
        initialize_package(pkgid)

        pkg = PKGDICT[pkgid]
        FUNCS = { "dpnp" : randn_dpnp , "numpy" : randn_np, \
         "cupy" : randn_cupy , "arrayfire" : randn_af }
        
        benchmark.extra_info["description"] = f"{NNSIZE:.2e} Samples"
        result = benchmark.pedantic(
            target=FUNCS[pkg.__name__],
            rounds=ROUNDS,
            iterations=ITERATIONS
        )

    
    def test_uniform(self, benchmark, pkgid):
        initialize_package(pkgid)

        pkg = PKGDICT[pkgid]
        FUNCS = { "dpnp" : randu_dpnp , "numpy" : randu_np, \
         "cupy" : randu_cupy , "arrayfire" : randu_af }

        result = benchmark.pedantic(
            target=FUNCS[pkg.__name__],
            rounds=ROUNDS,
            iterations=ITERATIONS
        )
