Installation & configuration
============================

Installation
------------

To install *dymoval* through *pip* simply run

.. code-block::

    pip install dymoval


Configuration
-------------
The configuration of *dymoval* is fairly straightforward since there are only 
two parameters that one can set. 

.. confval:: num_decimals
    :type: int
    :default: 3

    Number of decimal digits. 
    At the end of every function/method, *dymoval* round *float* numbers to a certain number of decimals.  

.. confval:: color_map
    :type: str
    :default: "tab10"

    The used *matplotlib* color map. Search for *Choosing Colormaps in Matplotlib* for possible values. 

These parameters can be set through a :code:`~/.dymoval/config.toml`  file.
You have to create such a file manually.
A :code:`~/.dymoval/config.toml` could for example include the following content

.. code-block::

    num_decimals = 4
    color_map = "tab20"


