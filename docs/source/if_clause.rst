If clause
=========

By using if clause, operators are conditionally applied in a quantum circuit.

If clause has the following syntax, and takes integer or function as conditional.

.. code-block:: python

   if_(refs, cond, ops)

- refs : reference or reference list
- cond : condition.
- ops  : operator or operator list.  When cond is satisfied, ops are executed.

When a reference is given to refs, a list containing this reference is internally created and used.

cond can take an integer value or a function.  Please see following sections.

When a operator is given as ops, a list containing this operator is internally created and used.


Using integer value as condition
--------------------------------

Values referred by given references are encoded as binary representation.  The reference at index 0 is least significant.  When a given reference is not measured yet, referred bit value is set to 0.

If the specified integer equals the encoded value, ops are executed.

The following is a simplified implementation to explain how if_ operator works.

.. code-block:: python
		
   v = 0
   for idx, ref in enumerate(refs):
       if get_value_from_ref(ref) == 1 :
           v |= 1 << idx

   if v == cond :
       ... run operators in ops ...

The following is an example.

.. code-block:: python
		
    qregs = new_qreg(3)

    # ... adding gates to a circuit
    
    refs = new_references(2)
    m0 = measure(refs[0], qregs[0])
    m1 = measure(refs[1], qregs[1])
    circuit += [m0, m1] # adding measure operation
      
    # X and Z gates are applied when measurement results for m0 and m1 are
    # 0 and 1 respectively.
    circuit += [if_(refs, 0b10, [X(qregs[2]), Z(qregs[1]), ...]),
		(other operatos) ...]
    

Using a function as condition
-----------------------------

.. note::
  
  will be removed in 0.3.  (see :ref:`changes_in_0_3:if_ clause`)

A function can be passed to cond.

This function receives a list of values referred by specified references, and should return True or False.
The order of values in the value list is the same as the order of references in refs.  If this function returns True, ops are executed.  When a given reference is not measured yet, None is passed as referred value.

.. code-block:: python
		
    qregs = new_qreg(3)
    
    # ... adding gates to a circuit
    
    refs = new_references(2)
    m0 = measure(refs[0], qregs[0])
    m1 = measure(refs[1], qregs[1])
    circuit += [m0, m1] # adding measure operations
      
    # compare function
    # returns True when measurement results for m0 and m1 are 0 and 1 respectively.
    def pred(values) :
        return values[0] == 0 and values[1] == 1
	
    # X and Z gates are applied when pred returns True
    circuit += [if_(refs, pred, [X(qregs[2]), Z(qregs[1]), ...]),
		(other operatos) ...]
