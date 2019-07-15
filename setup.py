from setuptools import setup, find_packages, Extension, dist
import numpy as np
import sys
import os

from distutils.command.install import install as DistutilsInstall

name = 'qgate'
version = '0.2.0r1'

pyver= [
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]

email = 'shin.morino@gmail.com'    
author='Shinya Morino'


classifiers=[
    'Operating System :: POSIX :: Linux',
    'Natural Language :: English',
    'License :: OSI Approved :: Apache Software License',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
]

url = 'https://github.com/shinmorino/qgate/'

import os
os.system('cd qgate/simulator/src; python incpathgen.py > incpath')
os.system('make -C qgate/simulator/src clean cuda_obj')


npinclude = np.get_include()

ext_modules = []

# cpu_runtime
ext = Extension('qgate/simulator/glue',
                include_dirs = [npinclude],
                sources = ['qgate/simulator/src/glue.cpp',
                           'qgate/simulator/src/Parallel_linux.cpp',
                           'qgate/simulator/src/GateMatrix.cpp',
                           'qgate/simulator/src/Misc.cpp',
                           'qgate/simulator/src/Types.cpp'],
                extra_compile_args = ['-std=c++11', '-fopenmp', '-Wno-format-security'],
                extra_link_args = ['-fopenmp'])
ext_modules.append(ext)
ext = Extension('qgate/simulator/cpuext',
                include_dirs = [npinclude],
                sources = ['qgate/simulator/src/cpuext.cpp',
                           'qgate/simulator/src/CPUQubitStates.cpp',
                           'qgate/simulator/src/CPUQubitProcessor.cpp',
                           'qgate/simulator/src/CPUSamplingPool.cpp',
                           'qgate/simulator/src/CPUQubitsStatesGetter.cpp',
                           'qgate/simulator/src/BitPermTable.cpp',
                           'qgate/simulator/src/Parallel_linux.cpp',
                           'qgate/simulator/src/Types.cpp'],
                extra_compile_args = ['-std=c++11', '-fopenmp', '-Wno-format-security'],
                extra_link_args = ['-fopenmp'])
ext_modules.append(ext)
ext = Extension('qgate/simulator/cudaext',
                include_dirs = [npinclude, '/usr/local/cuda/include'],
                sources = ['qgate/simulator/src/cudaext.cpp',
                           'qgate/simulator/src/CUDAGlobals.cpp',
                           'qgate/simulator/src/CUDAQubitStates.cpp',
                           'qgate/simulator/src/CUDAQubitProcessor.cpp',
                           'qgate/simulator/src/CUDAQubitsStatesGetter.cpp',
                           'qgate/simulator/src/CUDADevice.cpp',
                           'qgate/simulator/src/CPUSamplingPool.cpp',
                           'qgate/simulator/src/TransferringRunner.cpp',
                           'qgate/simulator/src/MultiDeviceMemoryStore.cpp',
                           'qgate/simulator/src/ProcessorRelocator.cpp',
                           'qgate/simulator/src/BitPermTable.cpp',
                           'qgate/simulator/src/DeviceTypes.cpp',
                           'qgate/simulator/src/Parallel_linux.cpp',
                           'qgate/simulator/src/Types.cpp'],
                extra_objects = ['qgate/simulator/src/DeviceGetStates.o',
                                 'qgate/simulator/src/DeviceProcPrimitives.o',
                                 'qgate/simulator/src/DeviceProbArrayCalculator.o'],
                libraries = ['cudart_static', 'rt'],
                library_dirs = ['/usr/lib', '/usr/local/cuda/lib64'],
                extra_compile_args = ['-std=c++11', '-fopenmp', '-Wno-format-security'],
                extra_link_args = ['-fopenmp'])
ext_modules.append(ext)
                    
setup(
    name=name,
    version=version,
    url=url,
    author=author,
    author_email=email,
    maintainer=author,
    maintainer_email=email,
    description='Quantum gate simulator.',
#    long_description=long_description,
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy>=1.14', 'ply>=3.10'],
    keywords='Quantum gate simulator, OpenMP, GPU, CUDA',
    classifiers=classifiers,
    ext_modules=ext_modules
)
