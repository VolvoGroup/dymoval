Create, analyze and manipulate a dataset
========================================

Log-data format depends on many aspects such as the specific application domain, the logging system manufacturer and so on and so forth.
For example, financial institutes use different logging systems than aerospace industries.

Due to that, it is impossible to establish a dataset format that fits every domain.
Therefore, we need need to find a solution. 

*Dymoval* defines a :ref:`Signal<signal>` object type with 
a simple structure and should be able to represent any signal independently of its nature. 

As a first step to use *Dymoval*, each logged signal shall be cast into a *Dymoval* :ref:`Signal <signal>`. 
Once done, a list of such :ref:`Signals<signal>` can be used to create a :ref:`Dataset<Dataset>` object.

However, when dealing with datasets, several problems may arise: 

- not all the logged signals are sampled with the same sampling period, 
- data loggers are run continuously and for long time and this
  leads to very large log-data where only few time-windows contain interesting data.
- log-data are often affected 
  by other problems such as noisy measurements, missing data and so on,
- ...

just to cite a few. 
*Dymoval* provides a number of functions for dealing with :ref:`Dataset <Dataset>` object.
Such functions include re-sampling, plot, frequency analysis, filtering and so on. 

Once you have created and adjusted a :ref:`Dataset <Dataset>` object, then you are ready to simulate your model.  

