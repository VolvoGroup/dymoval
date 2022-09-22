Dataset handling
================

Measurement datasets are a central part in model validation and therefore we designed 
the :ref:`Dataset` that offer a number of useful :ref:`methods <datasetMethods>` to deal with them.

There are two ways for creating a :ref:`Dataset <Dataset>` object.

#. Through a list of :ref:`Signals <signal>`
#. Through a specific structure pandas DataFrame


.. _signal:

Signal type
-----------

If the signals that you want to use as dataset are not sampled with the same sampling period, 
then you must convert them into *dymoval* :ref:`Signal <signal>` objects.

A  :ref:`Dataset <Dataset>` object - which is what you need at the end - can be instantiated 
through a list of :ref:`Signal <signal>` objects.   

.. currentmodule:: dymoval.dataset

*Dymoval* :ref:`Signals <signal>` are *typeddict* with the following keys

.. rubric:: Keys
.. autosummary::

   Signal.name
   Signal.values
   Signal.signal_unit
   Signal.sampling_period
   Signal.time_unit


.. rubric:: Functions on signals

*Dymoval* offers few function for dealing with :ref:`Signals <signal>`. 
Such functions are the following

.. autosummary::

   signals_validation
   fix_sampling_periods
   plot_signals


.. _Dataset:

Dataset class
-------------
The :ref:`Dataset`  is used to store and manipulate datasets.
Since model validation requires a datasets, this class is used also to instantiate 
 :ref:`ValidationSession <ValidationSession>`, i.e. a :ref:`Dataset <Dataset>` object 
becomes an attribute of a :ref:`ValidationSession <ValidationSession>`. 

A :ref:`Dataset <Dataset>` object can be instantiated in two ways

#. Through a list of dymoval :ref:`Signals<signal>`,
#. Through a pandas DataFrame with a specific structure.

See :py:meth:`~dymoval.dataset.signals_validation` 
and :py:meth:`~dymoval.dataset.dataframe_validation` functions for more information.


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

   Dataset.remove_means
   Dataset.remove_offset
   Dataset.low_pass_filter
   Dataset.replace_NaNs

.. rubric:: Plotting methods
.. autosummary::

   Dataset.plot
   Dataset.plot_coverage
   Dataset.plot_amplitude_spectrum

.. rubric:: Other methods
.. autosummary::
   Dataset.get_signal_list
   Dataset.get_dataset_values
   Dataset.export_to_mat

.. rubric:: Functions over Datasets
.. autosummary::

   dataframe_validation
   compare_datasets

