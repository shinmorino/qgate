Changes planned in Qgate 0.3
============================

Script API changes
------------------

Some functions in qgate.script API will be renamed.

- The first letter of all functions are capitalized.
- An operator, release_qreg() is renamed to RelaseQreg().

============ =============
0.1, 0.2.x   0.3 and above
============ =============
ctrl         Ctrl
measure      Measure
prob         Prob
if\_         If
barrier      Barrier
reset        Reset
============ =============

One can use updated API with Qgate 0.2.1 by importing qgate.script.script2.*

.. code-block:: python

    # to use script API from 0.3.
    import qgate.script.script2.*


Source code updater is included in 0.2.1 as a module of qgate.tools.update_script.  Below is example usages.

.. code-block:: bash

   # update.  Original source files are saved as *.py.org
   $ python -m qgate.tools.update_script file1.py file2.py ... 

   # update and overwrite.
   # Original files are overwritten by updated script files.
   $ python -m qgate.tools.update_script -o file1.py file2.py ... 
		

if\_ clause
-----------

if\_ clause will be changed not to accept function as cond, but accept a list of values of 0, 1 and None.  If a reference is not measured yet, corresponding value is set to None.


.. code-block:: python
		
    qregs = new_qreg(3)
    # ... applying gates to qreg
    
    refs = new_references(2)
    m0 = measure(refs[0], qregs[0])
    m1 = measure(refs[1], qregs[1])
    circuit += [m0, m1] # adding measure operation
      
    # The following will not be accepted from 0.3
    # def pred(values) :
    #    return values[0] == 0 and values[1] == 1
    # circuit += [if_(refs, pred, [X(qregs[2]]]), ...]

    # From 0.3, please use the following.
    cond = [0, 1]
    # X gate is applied when cond matches values pointed by refs.
    circuit += [if_(refs, cond, [X(qregs[2]]]), ...]

Gate set consolidation
----------------------

Built-in gate set will be conslidated for clearer internal design.

U1, U2 and U3 gates will be removed from built-in gates, and reimplemented as extensions.  These gates will be redefined as macro gates (or sequences of gates).

This does not mean unavailability of these gates from 0.3.  Script APIs of U1(), U2() and U3() will be provided in 0.3 and later. 

Sampling pool
-------------

Sampler class will be added, though details are not decided.
