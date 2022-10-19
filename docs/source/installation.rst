Installation & configuration
============================

Installation
------------

*Dymoval* has zero releases and it is still under development. 
Therefore, at the moment to install it through `pip` you must clone the repo on `GitHub`_ 
and then run 


.. code-block::

	cd /path/to/where/you/cloned/this/repo
	pip install -e .

.. _GitHub: https://github.com/VolvoGroup/dymoval

Configuration
-------------
The configuration of `dymoval` is fairly straightforward since there are only 
two parameters that you can set. 

.. confval:: num_decimals
    :type: int
    :default: 4

    Number of decimal digits. 
    At the end of every function/method, `dymoval` round *float* numbers to a certain number of decimals.  

.. confval:: color_map
    :type: str
    :default: "tab10"

    The used `matplotlib` color map. Check `Matplotlib` docs for possible values. 

.. confval:: ax_height
    :type: float
    :default: 2.5

    The subplot height when you save the figure. The default values are chosen to keep a 16:9 ratio. 
.. confval:: ax_width
    :type: float
    :default: 4.445

    The subplot width when you save the figure. The default values are chosen to keep a 16:9 ratio. 


These parameters can be set through a :code:`~/.dymoval/config.toml`  file.
You have to create such a file manually.
A :code:`~/.dymoval/config.toml` could for example include the following content

.. code-block::

    num_decimals = 4
    color_map = "tab20"


