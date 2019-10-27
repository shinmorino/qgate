Sampling
========

There're 2 ways to do sampling after simulation.

.. note::

  Sampling implementation is preliminary in 0.2.x.  APIs are subject to change.


Sampling by iterated simulations
--------------------------------

**Simulator.sample(circuit, ref_array, n_samples)** method runs circuit simulations for n_samples times, and returns measurement results as :ref:`observation:ObservationList`.

circuit   : a circuit to execute sampling
ref_array : a list of references which is used in measure operators in a circuit.
n_samples : number of sampling

Bit ordering of returned ObservationList is defined by ref_array.

.. code-block:: python

   qregs = new_qregs(n_qregs)
   refs = new_references(2)
   
   ops = [ ... gate sequence ...,
           measure(refs[0], qregs[n],
	   ... gate sequence ...,
           measure(refs[1], qregs[m],
	   ... gate sequence ...
	 ]
   
   sim = qgate.simulator.cpu()
   obslist = sim.sample(ops, refs, 100) # sampling results are returned as observation list.
   hist = obslist.histgram()   # getting histgram from observations.


Sampling pool
-------------
   
Sampling pool is for efficient and fast sampling by using pre-calculated probability vector.

In Qgate 0.2.x, sampling implementation has limitations shown below:

(1) Quantum circuits is not allowed to have measurement,

    | Measurement oprations destory quantum coherence, so should not be included. 

(2) Quantum circuits is not allowed to have if-clause.

    | if-clause uses measurement results to branch execution of quantum circuits.

Sampling pool is created by calling **Qubits.create_sampling_pool(qregs)**.  The parameter, qregs, is a list of qregs to be sampled.

By calling **SamplingPool.sample(n_samples)**, sampling results are returned as ObservationList.

Bit ordering of returned ObservationList is defined by the qregs parameter in **Qubits.create_sampling_pool(qregs)**.
    
.. code-block:: python

   ops = ... preparing circuit ...
   
   sim = qgate.simulator.cpu()
   sim.run(ops)

   qregs = ... list of quantum registers which will be sampled ...
   sampling_pool = sim.qubits.create_sampling_pool(qregs)

   obslist = sampling_pool.sample(n_samples) # sampling results are returned as observation list.
   hist = obslist.histgram()                 # getting histgram from observations.

