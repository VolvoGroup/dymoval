Simulate your model
===================

Model validation requires *simulated data* along with the *dataset* (i.e. your target environment logs) and therefore you have to simulate your model.

How? Well, you shall feed it with the input signals contained in the :ref:`Dataset <Dataset>` object that you prepared so far and that contains the target environment logs.

To extract the input signals from your :ref:`Dataset <Dataset>` object you can 
use the method :py:meth:`~dymoval.dataset.Dataset.dataset_values` and then you can use them to feed your model 
as long as you are working in a Python environment.

Otherwise, you can export your :ref:`Dataset <Dataset>` object in the format you want and import it into your modeling tool.
To facilitate with this task, *Dymoval* allows you to dump :ref:`Dataset <Dataset>` objects into :ref:`Signal <Signal>` objects through the method :py:meth:`~dymoval.dataset.Dataset.dump_to_signals` but then, you will have to manually export in an appropriate format depending on your modeling tool. 

.. note::
    Given the popularity of Matlab, *Dymoval* has a builtin function that exports Signals directly in *.mat* format. 

Once you have simulated your model you should import the simulated data into Python and then you are now ready to validate your model, and guess what? 
*Dymoval* is here for that!

Go to the next Section to discover more. 

.. note::
    Exporting/importing signals from/to Python to/from your modeling tool may be fairly annoying. 
    For this reason, we recommend to compile your model into an FMU and use the packages like *pyfmu* or *fmpy* 
    to simulate your model directly from a Python environment, so you have everything in one place.

    Independently of your modeling tool (Simulink, Dymola, GT-Power, etc), you most likely 
    have an option for compiling models into FMU:s.    
    Check the documentation of your modeling tool. 

