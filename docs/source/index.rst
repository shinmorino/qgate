Welcome to Qgate documentation!
===============================

Qgate is a quantum circuit simulator.  Current version is 0.2.2.

Concepts
--------

#. Easy development of quantum circuits with fast simulations for experiments

   | Simple built-in gate set.  Controlled gates can have multiple-control bits.
   | Simulations are accelerated by modern computing devices.

#. Big single node

   | Utilizing a big server with a huge amount of memory.
   | Focusing on performance.  No intra-node communication.
   
#. Designed to work as a backend of existing quantum computing SDKs.

   | Quantum circuit simulations can be acclerated without changing SDKs you are using.
   
   
Tutorials
---------

#. :ref:`quick_start_guide:Quick start guide`

   | Explains how to prepare a quantum circuits, run simulations and measure qubit states. 

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
   observation
   sampling
   plugin
   changes_in_0_3

Support for other quantum circuit simulators
--------------------------------------------

* :ref:`plugin:Plugins`

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
