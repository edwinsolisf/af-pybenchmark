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

ITERATIONS = 1

def generate_arrays(pkgid, count):
    arr_list = []
    pkg = PKGDICT[pkgid]
    pkg = pkg.__name__
    if "cupy" == pkg:
        cupy.random.seed(1)
        for i in range(count):
            arr_list.append(cupy.random.rand(NSIZE, NSIZE, dtype=DTYPE))
        cupy.cuda.runtime.deviceSynchronize()
    elif "arrayfire" == pkg:
        af.device_gc()
        for i in range(count):  
            arr_list.append(af.randu((NSIZE, NSIZE), dtype=getattr(af, DTYPE)))
    elif "dpnp" == pkg:
        dpnp.random.seed(1)
        for i in range(count):
            arr_list.append(dpnp.random.rand(NSIZE, NSIZE).astype(DTYPE))
    elif "numpy" == pkg:
        np.random.rand(1)
        for i in range(count):
            arr_list.append(np.random.rand(NSIZE, NSIZE).astype(DTYPE))

    return arr_list

@pytest.mark.parametrize(
    "pkgid", IDS, ids=IDS
)
class Eindot:
    def test_dot_a_b(self, benchmark, pkgid):
        setup = lambda: (generate_arrays(pkgid, 2), {})
        pkg = PKGDICT[pkgid]
        result = benchmark.pedantic(
            target=pkg.dot,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS,
        )

    def test_matmul_a_b(self, benchmark, pkgid):
        setup = lambda: (generate_arrays(pkgid, 2), {})
        pkg = PKGDICT[pkgid]
        result = benchmark.pedantic(
            target=pkg.matmul,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS,
        )

    def test_matmul_a_bt(self, benchmark, pkgid):
        a, b = generate_arrays(pkgid, 2)
        setup = lambda: ([a, b.T], {})
        pkg = PKGDICT[pkgid]
        result = benchmark.pedantic(
            target=pkg.matmul,
            setup=setup,
            rounds=ROUNDS,
            iterations=ITERATIONS,
        )

        
@pytest.mark.parametrize(
    "pkgid", IDS, ids=IDS
)
class TestLinalg:
    # def test_lstsq(self, benchmark, pkg):
    #     a, b = generate_arrays(pkg)
    #     setup = lambda: (generate_arrays(pkg), {"rcond":-1})

    #     result = benchmark.pedantic(
    #         target=pkg.lstsq,
    #         setup=setup,
    #         rounds=ROUNDS,
    #         iterations=ITERATIONS,
    #     )
    def test_cholesky(self, benchmark, pkgid):
        arr = generate_arrays(pkgid, 1)[0]
        pkg = PKGDICT[pkgid]
        setup = lambda: ([pkg.matmul(arr.T, arr) + pkg.matmul(arr.T, arr).T], {})

        benchmark.extra_info["description"] = f"{NSIZE}x{NSIZE} Matrix"
        if pkg.__name__ == 'arrayfire':
            result = benchmark.pedantic(
                target=pkg.cholesky,
                setup=setup,
                rounds=ROUNDS,
                iterations=ITERATIONS
            )
        else:
            result = benchmark.pedantic(
                target=pkg.linalg.cholesky,
                setup=setup,
                rounds=ROUNDS,
                iterations=ITERATIONS
            )

    def test_svd(self, benchmark, pkgid):
        setup = lambda: (generate_arrays(pkgid, 1), {})
        benchmark.extra_info["description"] = f"{NSIZE}x{NSIZE} Matrix"
        pkg = PKGDICT[pkgid]
        if pkg.__name__ == 'arrayfire':
            result = benchmark.pedantic(
                target=pkg.svd,
                setup=setup,
                rounds=ROUNDS,
                iterations=ITERATIONS
            )
        else:
            result = benchmark.pedantic(
                target=pkg.linalg.svd,
                setup=setup,
                rounds=ROUNDS,
                iterations=ITERATIONS
            )
    
    def test_inv(self, benchmark, pkgid):
        arr = generate_arrays(pkgid, 1)[0]
        pkg = PKGDICT[pkgid]
        setup = lambda: ([pkg.matmul(arr,arr)], {})

        benchmark.extra_info["description"] = f"{NSIZE}x{NSIZE} Matrix"
        if pkg.__name__ == 'arrayfire':
            result = benchmark.pedantic(
                target=pkg.inverse,
                setup=setup,
                rounds=ROUNDS,
                iterations=ITERATIONS
            )
        else:
            result = benchmark.pedantic(
                target=pkg.linalg.inv,
                setup=setup,
                rounds=ROUNDS,
                iterations=ITERATIONS
            )

    def test_pinv(self, benchmark, pkgid):
        setup = lambda: (generate_arrays(pkgid, 1), {})

        pkg = PKGDICT[pkgid]
        benchmark.extra_info["description"] = f"{NSIZE}x{NSIZE} Matrix"
        if pkg.__name__ == 'arrayfire':
            result = benchmark.pedantic(
                target=pkg.pinverse,
                setup=setup,
                rounds=ROUNDS,
                iterations=ITERATIONS
            ) 
        else:
            result = benchmark.pedantic(
                target=pkg.linalg.pinv,
                setup=setup,
                rounds=ROUNDS,
                iterations=ITERATIONS
            )

    def test_det(self, benchmark, pkgid):
        arr = generate_arrays(pkgid, 1)[0]
        pkg = PKGDICT[pkgid]
        setup = lambda: ([pkg.matmul(arr, arr.T)], {})
        benchmark.extra_info["description"] = f"{NSIZE}x{NSIZE} Matrix"

        if pkg.__name__ == 'arrayfire':
            result = benchmark.pedantic(
                target=pkg.det,
                setup=setup,
                rounds=ROUNDS,
                iterations=ITERATIONS
            )
        else:
            result = benchmark.pedantic(
                target=pkg.linalg.det,
                setup=setup,
                rounds=ROUNDS,
                iterations=ITERATIONS
            )
    
    def test_norm(self, benchmark, pkgid):
        setup = lambda: (generate_arrays(pkgid, 1), {})
        pkg = PKGDICT[pkgid]
        benchmark.extra_info["description"] = f"{NSIZE}x{NSIZE} Matrix"

        if pkg.__name__ == 'arrayfire':
            result = benchmark.pedantic(
                target=pkg.norm,
                setup=setup,
                rounds=ROUNDS,
                iterations=ITERATIONS
            )   
        else:
            result = benchmark.pedantic(
                target=pkg.linalg.norm,
                setup=setup,
                rounds=ROUNDS,
                iterations=ITERATIONS
            )   