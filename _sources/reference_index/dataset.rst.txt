Dataset handling
================

.. currentmodule:: dymoval.dataset

Measurement datasets are a central part in model validation and therefore we designed 
the :ref:`Dataset` that offer a number of useful :ref:`methods <datasetMethods>` to deal with them.

A typical workflow consists in casting your log-data into :ref:`Signal <signal>` objects and then use
the created :ref:`Signals <signal>` to instantiate a :ref:`Dataset <Dataset>` object.

.. _signal:

Signals
-----------
:ref:`Signal <signal>` are used to represent real-world signals. 

.. currentmodule:: dymoval.dataset

*Dymoval* :ref:`Signals <signal>` are *Typeddict* with the following keys

.. rubric:: Keys
.. autosummary::

   Signal.name
   Signal.values
   Signal.signal_unit
   Signal.sampling_period
   Signal.time_unit


.. rubric:: Functions

*Dymoval* offers few function for dealing with :ref:`Signals <signal>`. 
Such functions are the following

.. autosummary::

   validate_signals
   plot_signals


.. _Dataset:

Dataset class
-------------
The :ref:`Dataset`  is used to store and manipulate datasets.

Since to validate a model you need a datasets, objects of this class are used also to instantiate 
:ref:`ValidationSession <ValidationSession>` objects, and the passed :ref:`Dataset <Dataset>` object 
becomes an attribute of the newly created :ref:`ValidationSession <ValidationSession>` object. 

A :ref:`Dataset <Dataset>` object can be instantiated in two ways

#. Through a list of dymoval :ref:`Signals<signal>` (see :py:meth:`~dymoval.dataset.validate_signals` )
#. Through a *pandas* DataFrame with a specific structure (see :py:meth:`~dymoval.dataset.validate_dataframe`)
   
.. currentmodule:: dymoval.dataset

.. rubric:: Constructor
.. autosummary::

   Dataset

.. rubric:: Attributes
.. autosummary::

   Dataset.name
   Dataset.dataset
   Dataset.coverage
   Dataset.information_level

.. _datasetMethods:
.. rubric:: Manipulation methods
.. autosummary::

   Dataset.add_input
   Dataset.add_output
   Dataset.remove_signals
   Dataset.remove_means
   Dataset.remove_offset
   Dataset.remove_NaNs
   Dataset.apply
   Dataset.low_pass_filter
   Dataset.fft
   Dataset.trim

.. rubric:: Plotting methods
.. autosummary::

   Dataset.plot
   Dataset.plotxy
   Dataset.plot_coverage
   Dataset.plot_spectrum
   change_axes_layout

.. rubric:: Other methods
.. autosummary::

   Dataset.dump_to_signals
   Dataset.dataset_values
   Dataset.export_to_mat
   Dataset.dataset_values
   Dataset.signal_list
   validate_dataframe
   compare_datasets

