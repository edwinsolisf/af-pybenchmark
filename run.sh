#! /bin/bash

# install arrayfire wheels
python -m pip install gdown
python -m gdown --fuzzy https://drive.google.com/file/d/1GicXDBrQpwR1x13W3Ug5EioNylcEikf1/view?usp=sharing
python -m pip install arrayfire_binary_python_wrapper-0.8.0+af3.10.0-py3-none-linux_x86_64.whl
git clone https://github.com/edwinsolisf/arrayfire-py.git --branch afwheel310
python3 -m pip install -r arrayfire-py/dev-requirements.txt
python -m pip install build scikit-build-core
python -m build arrayfire-py --wheel
python -m pip install arrayfire-py/dist/arrayfire-0.1.0-py3-none-any.whl

# Run benchmarks
python -m pip install -r requirements.txt
python -m pytest pytest_benchmark/ --benchmark-json=results.json
python graphs.py