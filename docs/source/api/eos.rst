Equation of State
=================

Thermodynamic equations of state mapping density to pressure for weakly
compressible fluid simulations.

.. automodule:: ddgclib.eos
   :members:
   :undoc-members:
   :show-inheritance:

Base class
----------

.. automodule:: ddgclib.eos._base
   :members:
   :undoc-members:
   :show-inheritance:

Tait-Murnaghan EOS
-------------------

Weakly compressible EOS for liquids (water, oil). Parameters: reference
density ``rho0``, reference pressure ``P0``, bulk modulus ``K``, and
exponent ``n``.

.. automodule:: ddgclib.eos._tait_murnaghan
   :members:
   :undoc-members:
   :show-inheritance:

Ideal Gas EOS
-------------

Ideal gas equation of state for compressible gas phases.

.. automodule:: ddgclib.eos._ideal_gas
   :members:
   :undoc-members:
   :show-inheritance:

Multiphase EOS dispatcher
-------------------------

Routes pressure updates to per-phase EOS instances. Compatible with the
integrator ``pressure_model`` interface.

.. automodule:: ddgclib.eos._multiphase_eos
   :members:
   :undoc-members:
   :show-inheritance:

Pressure update
---------------

.. automodule:: ddgclib.eos._update
   :members:
   :undoc-members:
   :show-inheritance:
