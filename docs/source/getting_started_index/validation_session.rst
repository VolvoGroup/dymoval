Validate your model
===================

It is now time to validate your model.

Here is where *Dymoval* comes into play


.. figure:: ../figures/ModelValidationDymoval.svg
   :scale: 50 %

   The model validation process with *Dymoval*. 
 
*Dymoval* validates your model in terms of 

- R-square fit
- Residuals auto-correlation norm
- Input-Residuals cross-correlation norm 

and it also return the **coverage region**.


Model validation is done through :ref:`ValidationSession` objects.

A :ref:`ValidationSession <ValidationSession>` object is created from a :ref:`Dataset <Dataset>` object.
More precisely, the :ref:`Dataset <Dataset>` object becomes an attribute of the created :ref:`ValidationSession <ValidationSession>` object 
and therefore you can access all the attributes and methods of the associated :ref:`Dataset object <Dataset>` directly from the 
created :ref:`ValidationSession <ValidationSession>` object. 

.. figure:: ../figures/Composition.svg
   :scale: 50 %

   *Dymoval*  structure. You can access every attributes and methods of an inner object from an outer object. 

Once a :ref:`ValidationSession <ValidationSession>` object is created, then you can append as many simulation results 
as you want to it and validation metrics are automatically computed against the stored dataset 
and stored in the attribute :py:attr:`~dymoval.validation.ValidationSession.validation_results`.

It is also possible to visually inspect the residuals through the :py:meth:`~dymoval.validation.ValidationSession.plot_residuals` method.

To know more about the selected validation metrics, feel free to google *R-square* and why alone it is not enough to validate a model. 
