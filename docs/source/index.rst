.. dymoval documentation master file, created by
   sphinx-quickstart on Wed Aug 31 12:11:21 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Dymoval (Dynamic Model Validation)
====================================

What is it?
-----------

*Dymoval* is a Python package for *analyzing datasets* and *validating models*. 

*Dymoval* validates models against some user-provided dataset regardless of the model format (*DNN, transfer function, ODE,* etc.) and the tool 
used for developing it (*Simulink, Modelica,* etc.). 
If your development process is done in a CI/CD environment, *Dymoval*'s functions can be easily included in a development pipeline to test your changes. 

*Dymoval* further provides a number of functions for dataset analysis and manipulation.  

Although it is a very helpful in engineering, its usage may come handy in other application domains as well. 

What is not.
------------
*Dymoval* **is not** a tool for developing models. 
You have to develop your models with the tool you prefer.

It is nor a tool for *System Identification* (but we don't exclude it can happen in the future ;-)).

*Dymoval* only checks if your models are good or not but you have to develop your models by yourself in the environment that you prefer.

Why dymoval?
------------

There plenty of amazing packages out there like *matplotlib*, *pandas*, *numpy*, *scipy*, etc for analyzing data,
compute statistics, and so on,  
but they are huge and the plethora of functionalities they offer may be overwhelming.

*Dymoval* has been built on top of these tools and it aims at providing an extremely easy and intuitive API that shall serve most of the tasks an engineer typically face in his/her daily job.

However, *Dymoval* leaves the door open: most of the functions return objects that can be used straight away with the aforementioned tools without any extra steps.
Hence, if you need more power, you always get an object that can be immediately handled by some other more powerful tool while using *Dymoval*. 

Main Features
-------------

Datasets analysis and manipulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Time and frequency analysis 
- Physical units
- Easy plot of signals
- Missing data handling
- Linear filtering
- Means and offsets removal
- Re-sampling

Model validation
^^^^^^^^^^^^^^^^
- Validation metrics:
	- R-square fit
	- Residuals auto-correlation
	- Input-Residuals cross-correlation 
- Coverage region
- Enable model unit-tests
- Work for both SISO and MIMO models
- Modeling tool independence
- Easily integrate with CI/CD pipelines.


Index
-----
.. toctree::
   :maxdepth: 2
   
   installation
   getting_started
   integration
   reference
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
