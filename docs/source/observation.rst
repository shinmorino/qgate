Handling observation results
============================

Observation
-----------

Observation is a class to represent one sampling result.

Sampled value is internally held as integer.  On printing, it shows a bit string.  Observation.int property returns a sampled value as integer.

Methods that create Observation instances have a parameter to define bit ordering, and bits of measurement results are permuted according to the specified bit ordering.

.. code-block:: console

    >>> print(obs)
    00011001100110011001

    >>> print(obs.int)
    104857

When a qubit is not measured, a bit value in bit string is '*', and 0 in integer.

.. code-block:: console

    >>> print(obs)
    000110011001*0011**1

    >>> print(obs.int)
    104729

obs(qreg) extracts corresponding bit for the passed qreg.  If qreg is not measured, it returns None.

.. code-block:: console

    >>> print(obs(qreg))  # assuming qreg points to MSB
    0

   
ObservationList
---------------

ObservationList is a collection that contains sampling results.  This class is iterable and can be accessed by index.  For bit ordering and unmeasured qreg, ObservationList has the same rule as that of Observation.

On iteration, elements are given as Observation instances. 

.. code-block:: python

   obslist = sampling_pool.sample(n_samples) # sampling results are returned as observation list.
   # as iterable
   for obs in obslist: # type of obs is Observation
      print(obs)     # bit representation
      print(obs.int) # as integer

   # accessed by index.
   obs = obslist[0]   # obs is Observation instance.

Similar to Observation, ObservationList.intarray property returns sampling results as integer array.  To extract sampled value for one qreg, () operator is used.

   When a qreg is not measured, the corresponding bit is treated in the same way of Observation, '*' for print(), 0 for conversion to integer, None for extraction by qreg.

.. code-block:: console

   # printing obslist returns list of bit string.
   >>> print(obslist)
   [00011001100110011010, 00011001100110011011, 00011001100110011010]
		
   # intarray property returns integer array.
   >>> print(obslist.intarray)
   array([104858, 104859, 104858])

   # extracting sampled bits corresponding to qreg which corresponds to LSB.
   >>> print(obslist[qreg])
   [0, 1, 0]
   

ObservationHistgram
-------------------

ObservationHistgram is created by calling Observation.histgram().  It acts like a collection.Counter.  The key is integer and value is observation count.

When creating ObservationHistgram, a unmeasured bits are converted to 0, which is different from Observation and ObservationList, but other rules are the same.

On printing, keys are represented as bit string.

As a associative container in python, ObservationHistgram has methods of ObservationHistgram.keys(), ObservationHistgram.items().  It's also accessed by key, but only integer is acceptable as key.

.. code-block:: python

   # creating histgram.
   hist = obslist.histgram()

   >>> print(hist)
   {00011001100110010101: 1, 00011001100110011000: 1, 00011001100110011001: 2, 00011001100110011010: 9, 00011001100110011100: 1, 00011001100110011110: 2}
   
   >>> for v, count in hist.items():
   ...     print(v, count)
   ... 
   104853 1
   104856 1
   104857 2
   104858 9
   104860 1
   104862 2

   >>> print(hist[0])
   0
   >>> print(hist[104858])
   9
