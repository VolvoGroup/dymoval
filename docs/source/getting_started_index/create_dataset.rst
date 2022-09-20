Create, analyze and manipulate a dataset
========================================

Log-data format depends on many aspects such as the specific application domain, the logging system manufacturer and so on and so forth.
For example, financial institutes use different logging systems than aerospace industries.

Due to that, it is impossible to establish a dataset format that fits every domain.
Therefore, we need need to find a solution. 

*Dymoval* defines a :ref:`signal`  (it is actually a *Typeddict* datatype) with 
a simple structure and should be able to represent any signal independently of its nature. 

As a first step to use *dymoval*, each logged signal shall be cast into a *dymoval* :ref:`signal`. 
Once done, a list of such :ref:`Signals<signal>` can be used to instantiate a *dymoval* :ref:`Dataset`.

Another common problem with log-data, independently of the application domain and the equipment used, 
is that not all the logged signals are sampled with the same sampling period whereas a good dataset 
shall have all the signals sampled at the same rate. 

*Dymoval* provides a function for re-sampling :ref:`Signals<signal>`. 

.. note::
    If the signals in your log-data are sampled with the same sampling period,
    then you can import them directly in a pandas DataFrame with a specific structure without resorting to dymoval signals.
    See :ref:`Dataset` for more details.   

Furthermore, when running experiments, data loggers are run continuously and for long time. 
This leads to very large log-data where only few time-windows contain interesting data.
*Dymoval* allows you to trim log-data to keep only the portions with meaningful information.  

Finally, other than having signals sampled with different sampling periods, log-data are often affected 
by other problems such as noisy measurements, missing data and so on.

*Dymoval* offers a number of :ref:`methods <datasetMethods>` for copying with these kind of problems.
Once you created your :ref:`Dataset`, then you can plot your dataset,
you can analyze its frequency content and you can perform 
a bunch more operations including filtering, shifting, and so on.

Once you have a cleaned-up you dataset, then you are ready to simulate your model.  
