Reference Manual
================

As seen, the ingredients for validating a model are the *model* itself, a *dataset* and some *validation metrics*.
However, *Dymoval* is not a modeling tool and therefore its focus areas are the following

- :doc:`./reference_index/dataset`
- :doc:`./reference_index/validation`



How Dymoval has thought
-----------------------

*Dymoval* has been thought to help engineers in analyzing datasets and in developing models by providing a validation tool.

Although there plenty of amazing packages out there like *pandas*, *numpy*, *matplotlib*, etc.,
they are huge and the plethora of functionalities they offer may be overwhelming.

Therefore, the idea is to combine the functionalities of the aforementioned tools in such a way 
engineers are not overwhelmed by the plethora of functionalities these tools offer,
but, at the same we do not want to limit engineers but we want to guarantee access to all the power that such tools can provide.

Therefore, *Dymoval* classes are *composed* as shown in the picture below.

.. figure:: ./figures/Composition.svg
   :scale: 50 %

   *Dymoval*  structure. You can access every attributes and methods of an inner object from an outer object. 

This means that every outer package/module have full access to the classes/functions provided by the inner packages/modules.
For example, it is possible to access all :ref:`Dataset<Dataset>` methods from 
a :ref:`ValidationSession<ValidationSession>` object and all the *pandas* classes and methods from :ref:`Dataset<Dataset>` objects.

However, given that a outer objects do not contain only inner object types attributes, and given that such attributes are connected 
to other attributes, it is **discouraged** to directly change inner attributes with inner methods. 

For example, we know that a :ref:`Dataset<Dataset>` object attribute is a *pandas* Dataframe, but if we change such a *pandas* DataFrame, then we shall
update all the other :ref:`Dataset<Dataset>` attributes such as the *coverage region*, *Nan intervals*, etc. and that may become very messy.   


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

