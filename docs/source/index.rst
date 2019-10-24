Welcome to Qgate documentation!
===============================

Qgate is a quantum circuit simulator.  Current version is 0.2.1.

Python API is provided to develop quantum circuits and to run circuits.  For simulation, versions ofruntimes based on Python, CPU and GPU(CUDA) are provided.

Concepts
--------

#. Easy development of quantum circuits with fast simulations for experiments

   | Simple built-in gate set.  Controlled gates can have multiple-control bits.
   | Simulations are accelerated by modern computing devices.

#. Big single node

   | Utilizing a big server with a huge amount of memory.
   | Focusing on performance.  No intra-node communication.
   
#. Works as a backend of other quantum circuit simulators.

   | Other quantum circuit simulators can be accelerated.
   
   
Tutorials
---------

#. :ref:`quick_start_guide:Quick start guide`

   | Explains how to create a quantum circuits, run simulations and measure qubit states. 

#. :ref:`sampling_example:Sampling example (Phase Estimation)`

   | A practical example of phase estimation algorithm, and sampling from resulting quantum states.
   
   
Table of contents
-----------------
.. toctree::
   :maxdepth: 1

   quick_start_guide
   sampling_example
   qreg_and_reference
   operators
   gate
   measurement
   if_clause
   simulator
   sampling_pool
   plugin
   changes_in_0_3

Support for other  quantum simulators
-------------------------------------

* :ref:`plugin:Plugins`

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
