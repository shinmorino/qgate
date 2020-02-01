Simulator
=========

Simulator executes simulations for quantum circuits.

Qgate currently implements 3 runtimes.  Corresponding to these runtimes, simulator instances are created by using  **qgate.simulator.<runtime>()** methods where **<runtime>** is py, cpu or cuda.


#. Python

   | Instantiated by using **qgate.simulator.py()**.

   | Python verion is provided as a simple reference implementation.  No optimization is applied.


#. CPU(multicore)

   | Instantiated by using **qgate.simulator.cpu()**.
   
   | CPU verion utilizes multi-cores to accelerate simulation.

#. GPU(CUDA)

   | Instantiated by using **qgate.simulator.cuda()**.
   
   | GPU(CUDA) version utilizes NVIDIA GPU and CUDA to accelerate simulation.  It's enabled if a workstation/server has NVIDIA GPU in it.

Use cases
---------

There're 2 use cases of Simulator

#. Run circuits sucessively to see how these circuits work,

#. Samping.  Runinng a circuit one time, and getting multiple measurement results by using probabilities calculated from state vectors.

In this page, the use case 1. is covered.  For sampling, please see :ref:`sampling:Sampling`.


Running simulations
-------------------

Simulator instance has **simulator.run(circuit)** method. In Qgate, quantum circuits are built by using Python's list.  By passing this list, simulation runs.  **simulator.run(circuit)** returns when all operators in a given circuit are applied.

.. code-block:: python

   circuit = [ op0, op1, ... ]   # A quantum circuit built on a python's list.
   
   sim = qgate.simulator.cpu()   # creating a simulator instance.
   
   sim.run(circuit)              # running a simulation.

You can successively run circuits by calling simulator.run() multiple times.  After running circuits, you can reset to the initial state by calling **Simulator.reset()**. 

.. code-block:: python
   
   sim = qgate.simulator.cpu()           # creating a simulator instance.

   circuit0 = [ op0_0, op0_1, ... ]      # creating the 1st circuit.
   
   sim.run(circuit0)                     # running the 1st circuit.

   p = sim.qubits.calc_probability(qreg) # calucating probablity Pr(<0|qreg>)
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

**Simulator.values** is a property to access simulation results, which stores results of Measure and Prob operators.

Measure and Prob operators are asynchnously executed during simulation.  After simulation completed, simulation results are retrieved by calling **Simulator.values.get(refs)** method.  The parameter, refs, is reference(s) given to Measure and Prob operators as their parameters.

If measurement has not been done or probablity has not been calculated, None is returned.

In cases that one reference is passed to this method, one value is returned, and in cases that a list of references is passed, a list of values is returned.


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

**Simulator.qubits.calc_probability(qreg)** returns a probability for a specified qreg.  This call is synchronous while Prob operator is applied asynchronously.

.. code-block:: python
		
   sim.run(circuit)                            # simulation executed

   states = sim.qubits.calc_probability(qreg)  # calculating probability, Pr(<0|qreg>).

Setting qubit ordering
----------------------
   
By using **simulator.qubits.set_ordering(qreglist)**, qubit ordering is defined.  By specifying qreg order, quantum states and probabilities in vectors are reordered.

Accessing state and probability vectors
---------------------------------------
   
**Simulator.qubits** property is for direct access to state vectors.

Qubit states and probabilities are retrieved by using **Simulator.qubits.states** and **Simulator.qubits.prob** properties respectively.

Both properties work like numpy arrays, accepting slices to specify index ranges.

.. code-block:: python
		
   sim.run(circuit)                     # simulation executed

   states = sim.qubits.states[:]        # getting a copy of the whole state vector.

   states = sim.qubits.states[1::2]     # using slice.

   probs = sim.qubits.prob[:]           # caluclate probability for the whole state vector.

   probs = sim.qubits.prob[1::2]        # using slice.


Each bit in index of retrived arrays is correspoinding to a qreg(qubit).  To specify qubit ordering (bit position of a qreg in state vector index), **simulator.qubits.set_ordering(qreglist)** is used.
   
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

   # Unsupposed usage (slow).
   for i in range(N) :
       states = sim.states[i]  # accessing sates one by one.
       ... use v to calculate something ...
