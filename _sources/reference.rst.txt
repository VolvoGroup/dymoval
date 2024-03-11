Reference Manual
================

As seen, the ingredients for validating a model are the *model* itself, a *dataset* and some *validation metrics*.
However, *Dymoval* is not a modeling tool and therefore its focus areas are the following

- :doc:`./reference_index/dataset`
- :doc:`./reference_index/validation`



Dymoval Architecture
--------------------

*Dymoval* has been thought to help engineers in analyzing datasets and in developing models by providing a validation tool.

Although there plenty of amazing packages out there like *pandas*, *numpy*, *matplotlib*, etc.,
they are huge and the plethora of functionalities they offer may be overwhelming.

Therefore, the idea is to combine the functionalities of the aforementioned tools in such a way 
engineers are not overwhelmed but at the same time they can manage to get their job done quickly and effectively.
However, we do not want to limit engineers, but we want to guarantee access to the tools Dymoval has been built upon in a seamless way.

Therefore, *Dymoval* classes are *composed* as shown in the picture below.

.. figure:: ./figures/Composition.svg
   :scale: 50 %

   *Dymoval*  structure. You can access every attributes and methods of an inner object from an outer object. 

This means that every outer package/module have full access to the classes/functions provided by the inner packages/modules.
For example, it is possible to access all :ref:`Dataset<Dataset>` methods from 
a :ref:`ValidationSession<ValidationSession>` object and all the *pandas* classes and methods from :ref:`Dataset<Dataset>` objects.

Furthermore, object methods are not *"inplace"* but they always return a modified version of the calling object. 
For example

.. code-block::

   >>> ds.remove_means() # won't change ds
   >>> ds = ds.remove_means() # you must re-assign ds


Finally, each plotting function returns a *matplotlib* figure so that the user can access all the *matplotlib* API for further manipulating the figure in case 
he/she is not happy with the results from *Dymoval*.

.. warning::

   Given that a outer objects do not contain only inner object types attributes, and given that such attributes are connected 
   to other attributes, it is **discouraged** to directly change inner class attributes with inner methods. 
   
   For example, we know that a :ref:`Dataset<Dataset>` object attribute is a *pandas* Dataframe, but if we change such a *pandas* DataFrame, then we shall
   update all the other :ref:`Dataset<Dataset>` attributes such as the *coverage region*, *Nan intervals*, etc. and that may become very messy.
   **Therefore, if you want to change any attribute of any Dymoval object, then use the Dymoval class methods (if any) or create a new class instance.**
   


Package structure
-----------------
*Dymoval*'s package is arranged in the following modules

.. currentmodule:: dymoval
.. autosummary::  

    dataset
    validation
    utils

.. toctree::
   :hidden:

   reference_index/dataset
   reference_index/validation

