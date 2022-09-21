Create, analyze and manipulate a dataset
========================================

Log-data format depends on many aspects such as the specific application domain, the logging system manufacturer and so on and so forth.
For example, financial institutes use different logging systems than aerospace industries.

Due to that, it is impossible to establish a dataset format that fits every domain.
Therefore, we need need to find a solution. 

*Dymoval* defines a :ref:`signal` with 
a simple structure and should be able to represent any signal independently of its nature. 

As a first step to use *Dymoval*, each logged signal shall be cast into a *Dymoval* :ref:`Signal <signal>`. 
Once done, a list of such :ref:`Signals<signal>` can be used to create a *Dymoval* :ref:`object <Dataset>`.

However, when dealing with log-data, several problems may arise. Just to cite a few:

- not all the logged signals are sampled with the same sampling period, 
- data loggers are run continuously and for long time and this
  leads to very large log-data where only few time-windows contain interesting data.
- log-data are often affected 
  by other problems such as noisy measurements, missing data and so on,
- ...

*Dymoval* provides few functions for dealing with both :ref:`Signals<signal>` and  :ref:`Dataset <Dataset>` object.
Such functions include re-sampling, plot, frequency analysis, filtering and so on. 

Once you have created a :ref:`Dataset <Dataset>` object, then you are ready to simulate your model.  

.. note::
    If the signals in your log-data are sampled with the same sampling period,
    then you can import them directly in a pandas DataFrame with a specific structure without resorting to dymoval signals.
    See :ref:`Dataset` for more details.   
