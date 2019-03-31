Gate
====

Qgate implements various gates as shown Table 1.
Multi-control-bit and adjoint are also supported.

================ ============================================================================
 Type            Gate
================ ============================================================================
 1 qubit gate     
   No parameter    ID, H, X, Y, Z, S, T
   1 parameter     Rx(theta), Ry(theta), Rz(theta), U1(_lambda), Expii(theta), Expia(theta)
   2 parameters    U2(phi, _lambda)
   3 parameters    U3(theta, phi, lambda)
 Composed gate   Expi(theta)(gatelist)
 2 qubit gate    Swap(qreg0, qreg1)
================ ============================================================================


1 qubit gate
------------

To create a 1 qubit gate, the following syntax is used.

Tokens surrounded by ``<>`` may appear 0- or 1-time according to gates to be declared.

.. code-block:: python

  <cntr(qregs).>GateType<(paramters)><.Adj>(qreg)

- Control bits

  ``cntr(qregs).`` specify control bits.  It appears only when controlled gates are decalared.
  A comma-seperated list of qregs, a list of qregs, or their mixture is accepted.

- GateType<(parameters)>

  GateType is the gate name, such as H, Rx and Expii.
  If a specified gate type does not have any parameter, ``(paramters)`` is omitted.

- <.Adj>

  Specifying a gate is adjoint of GateType.
  All gates except for Swap gate support adjoint.
  Gates such as H and X are hermite, so their adjoint is identical.  In these cases, .Adj is simply ignored.

- (qreg)

  Qreg instance as a target qubit(qreg).


I, X, Y, Z, H, S, T gate, Single qubit gate without parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These gates are 1 qreg gates without parameters.

I, X, Y, Z : Identity and Pauli gates

.. math::
   
   I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix},
   X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix},
   Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix},
   Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}

H: Hadamard gate
S, T: Phase shfit gates
   
.. math::
   
   H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix},
   S = \begin{pmatrix} 1 & 0 \\ 0 & {e}^{\frac{i\pi}4} \end{pmatrix},
   T = \begin{pmatrix} 1 & 0 \\ 0 & {e}^{\frac{i\pi}4} \end{pmatrix}

   
.. code-block:: python

   # examples
   
   igate = I(qreg)  # I gate
   xgate = X(qreg)  # X gate
   ygate = Y(qreg)  # Y gate
   zgate = Z(qreg)  # Z gate
   
   hgate = H(qreg)  # H gate
   sgate = S(qreg)  # S gate
   tgate = T(qreg)  # T gate

   cx = ctrl(qreg0).X(qreg)         # CX gate
   ccx = ctrl(qreg0, qreg1).X(qreg) # Toffoli gate

   S.Adj(qreg)                      # Adjoint of S gate
   ctrl(qreg0).S.Adj(qreg1)         # Adjoint of controlled S gate



Rx, Ry, Rz, U1, Expii, Expiz gate, single qubit gate with one parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These gates are 1 qreg gates without parameters.

Rx(theta), Ry(theta), Rz(theta) : Rotation around X, Y, Z axes

.. math::
   
   Rx(\theta) = e^{-i{\theta}X / 2} = \begin{pmatrix} cos(\frac{\theta}2) & - i sin(\frac{\theta}2) \\ - i sin(\frac{\theta}2) & cos(\frac{\theta}2) \end{pmatrix}
   
   Ry(\theta) = e^{-i{\theta}Y / 2} = \begin{pmatrix} cos(\frac{\theta}2) & - sin(\frac{\theta}2) \\ sin(\frac{\theta}2) & cos(\frac{\theta}2) \end{pmatrix}
   
   Rz(\theta) = e^{-i{\theta}Z / 2} = \begin{pmatrix} e^{-i{\theta}/2} & 0 \\ 0 & e^{i{\theta}/2} \end{pmatrix}


U1(theta) : Phase shift gate with a given angle. This gate comes from OpenQASM specification.
   
.. math::
   
   U_1(\lambda) = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 \\ 0 & e^{i\lambda} \end{pmatrix}


Expii, Expiz : Exponents of I and Z matrices.

.. math::
   
   Expii(\theta) = e^{i{\theta}I} = \begin{pmatrix} e^{i\theta} & 0 \\ 0 & e^{i\theta} \end{pmatrix}
   
   Expiz(\theta) = e^{i{\theta}Z} = \begin{pmatrix} e^{i\theta} & 0 \\ 0 & e^{-i\theta} \end{pmatrix}

   
.. code-block:: python

   # examples
   
   rxgate = Rx(theta)(qreg)  # Rx gate
   rygate = Ry(theta)(qreg)  # Ry gate
   rzgate = Rz(theta)(qreg)  # Rz gate
   
   u1gate = U1(theta)(qreg)  # U1 gate
   expiigate = Expii(theta)(qreg)  # exp(i * theta * I) gate
   expizgate = Expiz(theta)(qreg)  # exp(i * theta * Z) gate

   crz = ctrl(qreg0).Rz(theta)(qreg)  # controlled Rz gate
   eizdg = Expiz(theta).Adj(qreg)     # Adjoint of Expiz gate


.. note::
   
   Rz gate definition is different from that defined in OpenQASM.
   Please us U1 gate as Rz gate if you need quantum circuits compatible with OpenQASM.
   

U2 gate, single qubit gate with 2 parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

U2(phi, lambda) : u2 gate defined in OpenQASM.
   
.. math::
   
   U_2(\phi, \lambda) = U_1(\phi + \frac{\pi}2)R_x(\frac{\pi}2)U_1(\lambda - \frac{\pi}2)
   
.. code-block:: python

   # examples
   
   u2gate = U2(phi, _lambda)  # U2 gate

   cu2 = ctrl(qreg0).u2(pha, _lambda)  # CX gate
   cu2dg = .Adj(qreg)                 # Adjoint of S gate


U3 gate, single qubit gate with 3 parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

U3(theta, phi, lambda) : u3 gate defined in OpenQASM.
   
.. math::
   
   U_3(\theta, \phi, \lambda) = U_1(\phi + 3\pi)R_x(\frac{\pi}2)U_1(\theta + \pi)R_x(\frac{\pi}2)U_1(\lambda - \frac{\pi}2)
   
.. code-block:: python

   # examples
   
   u3gate = U3(theta, phi, _lambda)  # U3 gate

   cu3 = ctrl(qreg0).u3(theta, pha, _lambda)  # Controlled U3 gate
   cu2dg = .Adj(qreg)                         # Adjoint of U3 gate


Composed gate
-------------

Expi is the only composed gate currently qgate implements.

Expi(theta)(gatelist) : Exponent of tensor product of gates in gatelist.

Expi gate is allowed to have (multiple-)controll bits.



Gates in gatelist should be Pauli and identity operators.  This gate applies exponent of tensor product of gates in gatelist to multiple qregs.

If there are sets of gates which have the same target qreg, these gates are fused to one gate before calculating tensor product.

.. math::
   
   Expi(\theta)(gatelist) = e^{i \theta [P_0 \otimes P_1 \otimes P_2 \otimes ... \otimes P_N]}

where :math:`P_i` is a matrix product of operators that shares a target qreg.


.. code-block:: python

   # examples
   gatelist = [Z(qreg0), Z(qreg1), X(qreg2), ... ]
   expigate = Expii(theta)(gatelist)          # Expii gate with a given gatelist

   cexpi = ctrl(qreg).expii(theta)(gatelist)  # Controlled U3 gate
   cu2dg = .Adj(qreg)                         # Adjoint of U3 gate


2 qubit gate
------------

Expi is the only composed gate currently qgate implements.

Swap(qreg0, qreg1) : Swapping qreg0 and qreg1.

Swap does not have neigher any control-bits nor adjoint.

.. code-block:: python

   # examples
   swap = Swap(qreg0, qreg1)
   
