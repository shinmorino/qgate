Welcome to Qgate documentation!
===============================

Qgate is a quantum gate simulator.  Current version is v0.2.

Python API is provided to develop quantum circuits and to run circuits.  For simulation, versions ofruntimes based on Python, CPU and GPU(CUDA) are provided.

note:

  Currently this document is not fully updated for Qgate 0.2.

Features
--------

#. Implemented in python with extensions to accelerate simulations

#. High-level quantum gate programming, including multi-control-bit control gate and multi-qubit measurement.

#. interoperable to other quantum gate computing frameworks with plugins.

   Blueqat plugin is currently planned.
   
#. Various simulator runtimes
   
   Python version    : As a reference imlpementation to show algorithm
   
   CPU version       : Parallelized by using OpenMP for performance.
   
   GPU(CUDA) version : Accelerated by using CUDA, NVIDIA GPUs.


Table of contents
-----------------
.. toctree::
   :maxdepth: 1

   quick_start_guide
   gate
   measurement
   simulator

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
