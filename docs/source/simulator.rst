Simulator
=========

Simulator executes simulations for quantum circuits.

Qgate currently implements 3 versions of simulators, and instantiated using **qgate.simulator.<runtime>()** methods, where runtime is py, cpu and cuda.


#. Python

   | Instantiated by using **qgate.simulator.py()**.

   | Python verion is provided as a reference, and simply implements algorithms.  No optimization is applied.


#. CPU(multicore)

   | Instantiated by using **qgate.simulator.cpu()**.
   
   | CPU verion utilizes multi-cores to accelerate simulation.

#. GPU(CUDA)

   | Instantiated by using **qgate.simulator.cuda()**.
   
   | GPU(CUDA) version utilizes NVIDIA GPU and CUDA to accelerate simulation.  It's enabled if a workstation/server has NVIDIA GPU in it.

Use cases
---------

There're 2 use cases of Simulator

#. Running a circuit sucessively to see how the circuit works,

#. Sampling multiple measurement results.

In this page, the use case 1. is covered.  For sampling, please see :ref:`sampling:Sampling`.


Running simulations
-------------------

Simulator instance has **Simulator.run(circuit)** method. In Qgate, quantum circuits are built by using Python's list.  By passing this list, simulation runs.  **Simulator.run(circuit)** returns after all operatos in a given circuit is applied.

.. code-block:: python

   circuit = [ op0, op1, ... ]   # A quantum circuit built on a python's list.
   
   sim = qgate.simulator.cpu()   # creating a simulator instance.
   
   sim.run(circuit)              # running a simulation.

You can successively run circuits by calling Simulator.run() multiple times.  After running circuits, you can reset simulator instance to the initial state by calling Simulator.reset(). 

.. code-block:: python
   
   sim = qgate.simulator.cpu()           # creating a simulator instance.

   circuit0 = [ op0_0, op0_1, ... ]      # creating the 1st circuit.
   
   sim.run(circuit0)                     # running the 1st circuit.

   p = sim.qubits.calc_probability(qreg) # calucating probablity Pr(0|qreg>)
                                         # to see qreg states.
   ... do something with prob. ...
   
   circuit1 = [ op1_0, op1_1, ... ]      # creating the 2nd circuit.
   
   sim.run(circuit1)                     # running a simulation.

   states = sim.qubits.states[:]         # Getting state vector.
   ... do something to check state vector. ...
   
   states = sim.qubits.prob[:]           # Getting probability vector.
   ... do something to check state vector. ...

   sim.reset()                           # resetting simulator


Accessing measurement results
-----------------------------

**Simulator.values** is a property to access to simulation results, which stores values obtained by measure and probability operators.

On creating measure and probability operators, references are used to refer simulation results.  By using **Simulator.values.get(refs)** method, simulation restults are obtained.

If measurement has not been done or probablity has not been calculated, None is returned.

When one reference is passed to this method, one value is returned, and when a reference list is passed, a list of values returned.

.. code-block:: python

   qreg0, qreg1 = new_qregs(2)
   ref0, ref1 = new_references(2)

   circuit += [
       ...
       prob(ref0, qreg0),
       ...
       measure(ref1, qreg1)
       ...
   ]
		
   sim.run(circuit)              # simulation executed

   p = sim.values.get(ref0)      # retrieving a probability on qreg0.
   
   r = sim.values.get(ref1)      # retrieving a measurement result on qreg1.

   # using list of references to get multiple values by one call.
   p, r = sim.values.get([ref0, ref1))


Calculating probability after simulation
----------------------------------------

**Simulator.qubits.calc_probability(qreg)** returns probability for a specified qreg.

.. code-block:: python
		
   sim.run(circuit)                            # simulation executed

   states = sim.qubits.calc_probability(qreg)  # calculating probability, Pr(0|qreg>).

Setting qubit ordering
----------------------
   
By using **simulator.qubits.set_ordering(qreglist)**, qubit ordering is defined.  It is used to access state vector and probability vector.

Accessing state and probability vectors
---------------------------------------
   
**Simulator.qubits** property is available to directly access to state vector.

**Simulator.qubits.states** property is for accessing states directly, and **Simulator.qubits.prob** property is for getting probability of states.

Both properties works like numpy arrays, accepting slices to specify an index range.

.. code-block:: python
		
   sim.run(circuit)                     # simulation executed

   states = sim.qubits.states[:]        # getting a copy of whole state vector.

   states = sim.qubits.states[1::2]     # using slice.

   probs = sim.qubits.prob[:]           # caluclate probability for whole state vector.

   probs = sim.qubits.prob[1::2]        # using slice.


Each index bit is correspoinding to a qreg(qubit).  To specify qubit ordering (bit position of a qreg in state vector index), **simulator.qubits.set_ordering(qreglist)** is available.
   
.. note::

   Simulator.qubits.states internally calculates and copies values. For performance reasons, please make a copy of values.

.. code-block:: python

   sim.qubis.set_ordering(qreglist) # set qreg ordering

   sim.run(...)                     # run a circuit.
   
   # Supposed usage
   states = sim.states[:]      # copy states to array
   for i in range(N) :
       v = states[i]
       ... use v to calculate something ...

   # Unsupposed usage(slow).
   for i in range(N) :
       states = sim.states[i]  # accessing sates one by one.
       ... use v to calculate something ...
