Model Validation
================

The :ref:`ValidationSession` class stores a dataset and all the validation results against such a dataset.
A :ref:`ValidationSession` object is instantiated from a :ref:`Dataset` object. 
The user can append as many simulation results as he/she want and the class automatically computes the validation metrics.   


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

   ValidationSession.plot_residuals
   ValidationSession.drop_simulation

