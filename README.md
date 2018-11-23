## qgate_sandbox
quantum gate simulator prototyping

### Build / Install

- To build, CUDA toolkit is required.  Please install CUDA 9 or later.
- setup.py assumes CUDA toolkit is under /usr/local/cuda

~~~
# move to your source directory
cd qgate_sandbox
# build
python setup.py bdist_wheel --plat-name manylinux1_x86_64
# install
pip install pip install dist/qgate-*-*-linux_x86_64.whl
~~~

### Run

Please go to the example directory, and run scripts.
~~~
# move to examples
cd examples
# run example scripts
python gate_tests.py
~~~
