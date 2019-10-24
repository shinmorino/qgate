Quantum register and referecne
==============================

Quantum register
^^^^^^^^^^^^^^^^

Quantum register (qreg) is a logical representation of qubit.  Operators has qreg(s) as target and controlled qubits.

Qreg are created by using **create_qreg()** or **create_qregs(n_qregs)**.

.. code-block:: python

    import qgate.script.*
    
    # creating one qreg.
    qreg = new_qreg()

    # creating 10 qregs.
    # new_qregs(n_qreg) returns a list of qregs.
    qregs = new_qregs(10)
    

Reference
^^^^^^^^^

Reference refers variables that are set during simulations

Use-cases are to refer measurement results and calculated probabilities for measure and pbo operations.  Reference is also used in if\_ clause to refer measurement results.

References are created by using **create_reference()** or **create_referecnes(n_refs)**.

.. code-block:: python

    import qgate.script.*
    
    # creating one qreg.
    ref = new_reference()

    # creating 10 qregs.
    # new_refs(n_qreg) returns a list of references.
    refs = new_references(10)
    
