## qgate

Quantum gate simulator

The current version is 0.1 (alpha, 3/31/2019).

Please visit [Qgate documentation](<https://shinmorino.github.io/qgate/docs/0.1/>) ([Quick start guide](<https://shinmorino.github.io/qgate/docs/0.1/quick_start_guide.html>)) for usages.


### Build / Install

- To build, CUDA toolkit is required.  Please install CUDA 9 or later.
- setup.py assumes CUDA toolkit is under /usr/local/cuda

~~~
# move to your source directory
cd qgate
# build
python setup.py bdist_wheel --plat-name manylinux1_x86_64
# install
pip install pip install dist/qgate-*-*-linux_x86_64.whl
~~~

Wheels for Python 2.7 and 3.5 are available from links below.  These packages have been built with CUDA 10 on Ubuntu 16.04.

- [qgate-0.1.0-cp27-cp27mu-linux_x86_64.whl](<https://github.com/shinmorino/qgate/raw/gh-pages/packages/0.1/qgate-0.1.0-cp27-cp27mu-linux_x86_64.whl>)

- [qgate-0.1.0-cp35-cp35m-linux_x86_64.whl](<https://github.com/shinmorino/qgate/raw/gh-pages/packages/0.1/qgate-0.1.0-cp35-cp35m-linux_x86_64.whl>)



### Run

Please go to the example directory, and run scripts.
~~~
# move to examples
cd examples
# run example scripts
python gate_tests.py
~~~


### Development plan

**v0.1**

- Current version. The first release.

**v0.2**

- Assert operator

- Cohere/decohere (dynamically adding/removing qubits to/from state vector)

**v0.3**

- Optimization
