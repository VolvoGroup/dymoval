.. dymoval documentation master file, created by
   sphinx-quickstart on Wed Aug 31 12:11:21 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Dymoval (DYnamic MOdel VALidation)
====================================

What is it?
-----------

*Dymoval* is a Python package for  *validating models* and *analyzing datasets*. 

*Dymoval* validates model against some user-provided dataset regardless of their format (*DNN, transfer function, ODE,* etc.) and the tool  used for developing them (*Simulink, Modelica,* etc.). 
If your development process is done on a CI/CD environment, *dymoval*'s functions can be easily included in a development pipeline to test your changes. 
It can be also used to unit-test your models.

*Dymoval* further provides a number of functions for dataset analysis and manipulation.  

Although it is a very helpful in engineering, its usage may come handy in other application domains as well. 

What is not.
------------
*Dymoval* **is not** a tool for developing models. 
You have to develop your models with the tool you prefer.

It is nor a tool for *System Identification* (but we don't exclude it can happen in the future ;-)).

*Dymoval* only checks if your models are good or not, regardless of the tool you used for developing them. 

Why dymoval?
------------

There plenty amazing packages out there like pandas, *numpy*, *scipy*, etc for analyzing data,
compute statistics, and so on,  
but they are huge and the plethora of functionalities they offer may be overwhelming.

*Dymoval* has been thought with engineers in mind, and it offers just the essential tools 
for copying with the tasks an engineer needs to carry out by trying to keep an as simple as possible API.

Main Features
-------------

Model validation
****************

- Models unit-tests
- Validation metrics:
	- R-square fit (%)
	- Residuals auto-correlation
	- Input-Residuals cross-correlation 
- Coverage region
- Modeling tool independency
- Works for both SISO and MIMO models
- Easily integrates with CI/CD pipelines (Jenkins, GitLab, etc.) 


Datasets analysis and manipulation
**********************************

- Time and frequency analysis 
- Missing data handling
- Linear filtering
- Means and offsets removal
- Re-sampling

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
