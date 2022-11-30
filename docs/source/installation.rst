Installation & configuration
============================

Installation
------------

By running 

.. code-block::

   pip install dymoval

everything should work, but there are few things to keep in mind.

Typically :code:`conda` handles scientific packages better than `pip`, and given that many *dymoval* dependencies are scientific packages, it is suggested to install all the dependencies through :code:`conda` and then to install *dymoval* through :code:`pip`.

To do that, download the :code:`environment.yml` file from `here`_ and run


.. code-block::

   conda env update --name env_name --file environment.yml
   pip install dymoval

where *env_name* is the environment name where you want to install *dymoval*.
If not provided, *dymoval* will be installed in a new environment called :code:`dymoval`.

.. _here: https://github.com/VolvoGroup/dymoval/blob/main/environment.yml

Why not `conda install dymoval`?
--------------------------------

Unfortunately, it is not possible (yet?) to easily build :code:`conda` packages when the project is handled through a :code:`pyproject.toml` file, and therefore the *dymoval* package, which uses a :code:`pyproject.toml` file, is only available through :code:`pip`.


Installation from the repo
***************************
Clone the repo from `the_repo`_  and run

.. _the_repo: https://github.com/VolvoGroup/dymoval


.. code-block::
    
    cd /path/to/where/you/cloned/this/repo
    conda env update --name env_name --file environment.yml
    conda activate env_name
    pip install .

or

.. code-block::

	cd /path/to/where/you/cloned/this/repo
	pip install .


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

These parameters can be set through a :code:`~/.dymoval/config.toml`  file.
You have to create such a file manually.

A :code:`~/.dymoval/config.toml` could for example include the following content

.. code-block::

    num_decimals = 4
    color_map = "tab20"


