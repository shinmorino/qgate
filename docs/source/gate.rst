Gate
====

The below table shows built-in gates in Qgate.

================ ============================================================================
 Type            Gate
================ ============================================================================
 1 qubit gate     
   No parameter    I, H, X, Y, Z, S, T
   1 parameter     Rx(theta), Ry(theta), Rz(theta), U1(_lambda), Expii(theta), Expiz(theta)
   2 parameters    U2(phi, _lambda)
   3 parameters    U3(theta, phi, lambda)
 Composed gate   Expi(theta)(gatelist)
 2 qubit gate    Swap(qreg0, qreg1)
================ ============================================================================

Qgate also suports controlled gate and adjoint.

- Controlled gate

  | All gates except for Swap gate works as controlled gate.  Every contolled gate is able to have arbitary number of control bits.

- Adjoint

  | All gates except for Swap gate have their adjoint.



1 qubit gate
------------

To create a 1 qubit gate, the following syntax is used.

Tokens surrounded by ``<>`` is optional, may appear 0- or 1-time according to a gate to be created.

.. code-block:: python

  <ctrl(qregs).>GateType<(paramters)><.Adj>(qreg)

- Control bits

  ``ctrl(qregs).`` specify control bits.  It's required to create a controlled gate.
  A comma-separated list of qregs, a list of qregs, or their mixture is accepted.

- GateType<(parameters)>

  GateType is a gate name, such as H, Rx and Expii.
  If a specified gate type does not have any parameter, ``(paramters)`` is omitted.

- <.Adj>

  Specifying a gate is adjoint of GateType.
  All gates except for Swap gate support adjoint.
  Gates such as H and X are hermite, so their adjoint is identical.  In these cases, .Adj is simply ignored.

- (qreg)

  Qreg instance as a target.


Examples:

.. code-block:: python

   # Hadamard gate
   H(qreg0)

   # Controlled X gate (CX gate)
   ctrl(qreg0).X(qreg1)

   # 2-control-bit X gate (Toffoli gate)
   ctrl(qreg0, qreg1).X(qreg2)

   # Rx gate (1 parameter)
   Rx(0.)(qreg)

   # Adjoint of Rx gate
   Rx(0.).Adj(qreg)

   # adjoint of 3-bit-controlled U3 gate
   # control bits are given by a python list.
   ctrlbits = [qreg0, qreg1, qreg2]  # creating a list of control bits
   ctrl(ctrlbits).U3(theta, phi, _lambda).Adj(qreg3)
  

I, X, Y, Z, H, S, T gate, Single qubit gate without parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gates in this section are single qubit gates without parameters.

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
   S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix},
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

Gates in this section are single qubit gates with one parameter.

Rx(theta), Ry(theta), Rz(theta) : Rotation around X, Y, Z axes

.. math::
   
   Rx(\theta) = e^{-i{\theta}X / 2} = \begin{pmatrix} cos(\frac{\theta}2) & - i sin(\frac{\theta}2) \\ - i sin(\frac{\theta}2) & cos(\frac{\theta}2) \end{pmatrix}
   
   Ry(\theta) = e^{-i{\theta}Y / 2} = \begin{pmatrix} cos(\frac{\theta}2) & - sin(\frac{\theta}2) \\ sin(\frac{\theta}2) & cos(\frac{\theta}2) \end{pmatrix}
   
   Rz(\theta) = e^{-i{\theta}Z / 2} = \begin{pmatrix} e^{-i{\theta}/2} & 0 \\ 0 & e^{i{\theta}/2} \end{pmatrix}


U1(theta) : Phase shift gate for a given angle. This gate comes from OpenQASM specification.
   
.. math::
   
   U_1(\lambda) = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 \\ 0 & e^{i\lambda} \end{pmatrix}


Expii, Expiz : Matrix exponential for I and Z matrices.

.. math::
   
   Expii(\theta) = e^{i I{\theta}} = \begin{pmatrix} e^{i\theta} & 0 \\ 0 & e^{i\theta} \end{pmatrix}
   
   Expiz(\theta) = e^{i Z{\theta}} = \begin{pmatrix} e^{i\theta} & 0 \\ 0 & e^{-i\theta} \end{pmatrix}

   
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
   Please use U1 gate as Rz gate if you need quantum circuits compatible with OpenQASM.
   

U2 gate, single qubit gate with 2 parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gates in this section are single qubit gates with two parameters.

U2(phi, lambda) : u2 gate defined in OpenQASM.  Global phase differs from the original definition.
   
.. math::
   
   U_2(\phi, \lambda) = U_3(\frac{\pi}2, \phi, \lambda) =
   \frac{1}{\sqrt{2}}
   \begin{pmatrix}
   e^{-i \frac{\phi + \lambda}2}
   & - e^{-i \frac{\phi - \lambda}2}
   \\ e^{i \frac{\phi - \lambda}2}
   & e^{i \frac{\phi + \lambda}2}
   \end{pmatrix}
   
.. code-block:: python

   # examples
   
   u2gate = U2(phi, _lambda)  # U2 gate

   cu2 = ctrl(qreg0).U2(phi, _lambda)(qreg1) # controlled U2 gate.
   u2dg = U2(phi, _lambda).Adj(qreg)         # Adjoint of U2 gate


U3 gate, single qubit gate with 3 parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gates in this section are single qubit gates with three parameters.

U3(theta, phi, lambda) : u3 gate defined in OpenQASM, global phase differs from the original definition.
   
.. math::
   
   U_3(\theta, \phi, \lambda) = 
   \begin{pmatrix}
   e^{-i \frac{\phi + \lambda}2} cos(\frac{\theta}2)
   & - e^{-i \frac{\phi - \lambda}2} sin(\frac{\theta}2)
   \\ e^{i \frac{\phi - \lambda}2} sin(\frac{\theta}2)
   & e^{i \frac{\phi + \lambda}2} cos(\frac{\theta}2)
   \end{pmatrix}
   
.. code-block:: python

   # examples
   
   u3gate = U3(theta, phi, _lambda)  # U3 gate

   cu3 = ctrl(qreg0).U3(theta, phi, _lambda)(qreg1)  # Controlled U3 gate
   u3dg = U3(theta, phi, _lambda).Adj(qreg)          # Adjoint of U3 gate


Macro gate
----------

Expi is the macro gate that Qgate currently implements.

Expi(theta)(gatelist)

- gatelist : list of pauli and/or identity gates.
- theta    : angle of rotation.

Expii gate is for rotation of a product of pauli and identity gates.  This gate is able to have (multiple-)controll bits, and adjoint is also available.

.. math::
   
   Expi(\theta)(gatelist) = e^{i \theta [P_0 \otimes P_1 \otimes P_2 \otimes ... \otimes P_N]}

where :math:`P_i` is a matrix product of operators that shares a target qreg.

Examples:

.. code-block:: python

   # examples
   gatelist = [Z(qreg0), Z(qreg1), X(qreg2), ... ]
   expigate = Expii(theta)(gatelist)                            # Expii gate with a given gatelist

   cexpi = ctrl(qreg).expii(theta)(gatelist)                    # Controlled Expi gate
   u3dg = U3(math.pi., math.pi / 4.., math.pi / 8.).Adj(qreg)   # Adjoint of Expi gate

   # exp(i * math.pi * X), identical to Rx(math.pi).
   Expi(math.pi)(X(qreg))

   # can have a sequence of pauli operators
   Expi(math.pi / 2.)(X(qreg0), Y(qreg1), Z(qreg2))
   
   # Can be a controlled gate
   ctrl(qreg0).expi(math.pi)(Y(qreg1))
   
   # Supports adjoint
   expi(math.pi).Adj(Y(qreg1))


2 qubit gate
------------

Swap is the only 2 qubit gate currently qgate implements.

Swap(qreg0, qreg1) : Swapping qreg0 and qreg1.

Swap does not have any control-bits nor adjoint.

.. code-block:: python

   # examples
   swap = Swap(qreg0, qreg1)
