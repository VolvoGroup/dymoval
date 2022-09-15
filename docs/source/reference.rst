Reference Manual
================

As seen, the ingredients for validating a model are the *model* itself, a *dataset* and some *validation metrics*.
Given that *dymoval* is not a modeling tool, then its focus areas are the following

- :doc:`./reference_index/dataset`
- :doc:`./reference_index/validation`


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

Behind the scenes
-----------------
The Section :doc:`./reference_index/some_theory` describes how *dymoval* validate your models.