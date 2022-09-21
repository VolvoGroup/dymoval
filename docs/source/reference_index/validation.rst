Model Validation
================

A :ref:`ValidationSession<ValidationSession>` object stores a :ref:`Dataset <Dataset>` object that serves as a basis for validate your models.
Then, the user can append as many simulation results as he/she want to the same 
:ref:`ValidationSession<ValidationSession>` object that automatically computes validation metrics 
and store the results in the 
:py:attr:`~.ValidationSession.validation_results` attribute.

Note that a :ref:`ValidationSession<ValidationSession>` object is instantiated from a :ref:`Dataset <Dataset>` object and therefore 
the user can access all the attributes and methods of the :ref:`Dataset` directly from the :ref:`ValidationSession<ValidationSession>` object.


.. _ValidationSession:

ValidationSession class
-----------------------

.. currentmodule:: dymoval.validation

.. rubric:: Constructor

.. autosummary::

   ValidationSession

.. rubric:: Attributes
.. autosummary::

   ValidationSession.name
   ValidationSession.Dataset
   ValidationSession.auto_correlation
   ValidationSession.cross_correlation
   ValidationSession.validation_results

.. rubric:: Methods
.. autosummary::

   ValidationSession.append_simulation
   ValidationSession.drop_simulation
   ValidationSession.plot_simulations
   ValidationSession.plot_residuals
   ValidationSession.get_simulations_name
   ValidationSession.get_simulation_signals_list
   ValidationSession.clear

