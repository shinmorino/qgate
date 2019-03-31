Simulator
=========

Simulator is to run simulation of quantum circuits.

Simulator instance is created by using **qgate.simulator.<runtime>()**, where runtime is py, cpu and cuda.

Qgate currently implements 3 versions of simulators,

#. Python

   Instantiated by using **qgate.simulator.py()**.

   Python verion is provided as a reference, and simply implements algorithms.  No optimization is applied.


#. CPU(multicore)

   Instantiated by using **qgate.simulator.cpu()**.
   
   CPU verion utilizes multi-cores to accelerate simulation.

#. GPU(CUDA)

   Instantiated by using **qgate.simulator.cuda()**.
   
   GPU(CUDA) version utilizes NVIDIA GPU and CUDA to accelerate simulation.  It's enabled if a workstation/server has NVIDIA GPU in it.


Running a simulation
--------------------

Simulator instance has **Simulator.run(circuit)** method. In qgate, quantum circuits are built by using Python's list.  By passing this list, simulation runs.  **Simulator.run(circuit)** returns after all operatos in a given circuit is applied.


.. code-block:: python

   circuit = [ op0, op1, ... ]   # A quantum circuit built on a python's list.
   
   sim = qgate.simulator.py()    # creating simulator instance.
   
   sim.run(circuit)              # running a simulation.


Accessing measurement results
-----------------------------

**Simulator.values** is a property to access to simulation results, which stores values obtained by Measure and Probability operators.

On creating measure and probability operators, references are used as a placeholder to receive simulation results.  By using **Simulator.values.get(refs)** method, simulation restults are obtained.

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

   states = sim.qubits.calc_probability(qreg)  # calculating probability, Pr(0|Qreg>).
   

Accessing state vector
----------------------
   
**Simulator.qubits** property is available to directly access to state vector.

**Simulator.qubits.states** property is for accessing states directly, and **Simulator.qubits.prob** property is for getting probability of states.

Both properties works like numpy arrays, accepting slices to specify an index range.

.. code-block:: python
		
   sim.run(circuit)                     # simulation executed

   states = sim.qubits.states[:]        # getting a copy of whole state vector.

   states = sim.qubits.states[1::2]     # using slice.

   probs = sim.qubits.prob[:]           # caluclate probability for whole state vector.

   probs = sim.qubits.prob[1::2]        # using slice.


Each index bit correspoinding to a qreg(qubit).  To find out a lane (bit position of a qreg in state vector index), **simulator.qubits.lanes.get_state_idx(qreg)** is available.  (May be revised later version.)
   
.. note::

   Simulator.qubits.states internally calculates and copies values. For performance reasons, please make a copy of values.

.. code-block:: python

   sim.run(...)        # run a circuit.
   
   # Expected usage
   states = sim.states[:]     # copy all states
   for i in range(N) :
       v = states[i]
       ... use v to calculate something ...

   # Unexpected usage
   for i in range(N) :
       states = sim.states[i]  # accessing one by one, it's slow.
       ... use v to calculate something ...


