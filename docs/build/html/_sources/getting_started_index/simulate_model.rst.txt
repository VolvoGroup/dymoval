Simulate your model
===================

To validate your model you should first simulate it since you have to collect model output data.
Yoy shall feed your model with *exactly* the input signals contained in the cleaned-up dataset.

To extract the input signals from your :ref:`Dataset` you can use the method :py:meth:`~dymoval.dataset.get_dataset_values`. 
Then, you should first save the extracted signals on disk in the format that you want and then you should import them 
into your modeling tool. 

.. note::
    Given the popularity of Matlab, dymoval has a builtin function that exports datasets directly in *.mat* format. 

Once you imported the input signals into your modeling tool, then you can run some simulations.

Once done, you should export your simulation results in a format that can be imported back into Python.

You are now ready to validate your model, and guess what? 
*Dymoval* is here for that reason!

.. note::
    Exporting/importing signals from/to Python may be fairly annoying. 
    For this reason, we recommend to compile your model into an FMU and use the package pyfmu
    to simulate your model directly from a Python environment, so you have everything in one place
    and you avoid to export/import signals continuously.

    Independently of your modeling tool (Simulink, Dymola, GT-Power, etc), you most likely 
    have an option for compiling models into FMU:s.    
    Check the documentation of your modeling tool. 





