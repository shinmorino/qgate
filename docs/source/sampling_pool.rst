Sampling
========

There're 2 ways to do sampling after simulation.

.. note::

  Sampling implementation is preliminary in 0.2.x.  APIs are subject to change.


Sampling by iterated simulations
--------------------------------

Simulator.sample() method runs circuit simulations for n_samples times where m_samples is the number of sampling.  Measurement results are accumlated and returned.


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
   obslist = sim.sample(ops)   # sampling results are returned as observation list.
   hist = obslist.histgram()   # getting histgram from observations.


Sampling pool
-------------
   
Sampling pool is for efficient and fast sampling by using pre-calculated probability.

In Qgate 0.2.x, sampling implementation has limitations shown below:

(1) Quantum circuits is not allowed to have measurement,

    | Measurement oprations destory quantum coherence, so should not be included. 

(2) Quantum circuits is not allowed to have if-clause.

    | if-clause uses measurement results to branch execution of quantum circuits.

.. code-block:: python

   ops = ... preparing circuit ...
   
   sim = qgate.simulator.cpu()
   sim.run(ops)

   qregs = ... list of quantum registers which will be sampled ...
   sampling_pool = sim.qubits.create_sampling_pool(qregs)

   obslist = sampling_pool.sample(n_samples) # sampling results are returned as observation list.
   hist = obslist.histgram()                 # getting histgram from observations.
