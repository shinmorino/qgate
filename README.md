## Qgate

Quantum gate simulator

Version 0.2 has beein released on 7/15/2019.

Please visit [Qgate documentation](<https://shinmorino.github.io/qgate/docs/0.1/>) ([Quick start guide](<https://shinmorino.github.io/qgate/docs/0.1/quick_start_guide.html>)) for usages.

Note: The document is for version 0.1.  Version 0.2 docs will be uploaded later.


### Build / Install

- To build, CUDA toolkit is required.  Please install CUDA 9 or later.
- setup.py assumes CUDA toolkit is under /usr/local/cuda

~~~
# move to your source directory
cd qgate
# build
python setup.py bdist_wheel --plat-name manylinux1_x86_64
# install
pip install dist/qgate-*-*-linux_x86_64.whl
~~~

Wheels for Python 2.7, 3.5 and 3.6 are available from links below.  These packages have been built with CUDA 10 on Ubuntu 16.04 (2.7, 3.5) and on Ubuntu 18.04 (3.6), and are expected to work on other linux distros.

- 2.7: [qgate-0.2.0.post1-cp27-cp27mu-linux_x86_64.whl](<https://github.com/shinmorino/qgate/raw/gh-pages/packages/0.2/qgate-0.2.0.post1-cp27-cp27mu-manylinux1_x86_64.whl>)

- 3.5: [qgate-0.2.0.post1-cp35-cp35m-linux_x86_64.whl](<https://github.com/shinmorino/qgate/raw/gh-pages/packages/0.2/qgate-0.2.0.post1-cp35-cp35m-manylinux1_x86_64.whl>)

- 3.6: [qgate-0.2.0.post1-cp36-cp36m-linux_x86_64.whl](<https://github.com/shinmorino/qgate/raw/gh-pages/packages/0.2/qgate-0.2.0.post1-cp36-cp36m-manylinux1_x86_64.whl>)


### NVIDIA driver requirement

NVIDIA Driver 410.48 or later is required to use prebuilt wheels, 

If CUDA 9 (R384) driver is installed and you want to stay at R384, please visit [4. CUDA Compatilbility platform](<https://docs.nvidia.com/deploy/cuda-compatibility/#cuda-compatibility-platform>).  Drivers of 384.111 or later are able to run CUDA 10 applications.

### Run

Please go to the example directory, and run scripts.
~~~
# move to examples
cd examples
# run example scripts
python gate_tests.py
~~~


### Current version

**0.2**

New

- Dynamic Qubit Grouping

  dynamically adding/removing qubits to/from state vectors

- Multi-GPU

  Using device memories in multiple GPUs to run bigger circuits.

  Prerquisite is GPUDirectP2P.  To get performance, NVLink is required.

- SamplingPool

  Efficent and fast sampling by using pre-calculated probability vector.

- Blueqat plugin (preliminary)

- OpenQASM parser (preliminary)

  Parsing OpenQASM to dynamically create circuits or generate python source code to define circuits.

  (Currently macro and opaque are not implemented.)

Fixes/Changes

- Global phases of U3 and U2 gates are adjusted for consistency with Qgate gate set.

- The name of global phase gate is changed from Expia to Expii.

- The matrix for U2 gate was numerically incrrect.  It was fixed and tested.


### Development plan

**0.2.1**

- Gate reordering

- Assert operator

- Refactoring

**0.3**

- Gate fusion

- Other optimizations

### Version history

**0.1**

- The first release.
