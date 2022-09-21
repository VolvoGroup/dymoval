Validate your model
===================

It is now time to validate your model!
This is done through the :ref:`ValidationSession`.

A :ref:`ValidationSession <ValidationSession>` object is created from a :ref:`Dataset <Dataset>` object.
More precisely, the :ref:`Dataset <Dataset>` object becomes an attribute of the created :ref:`ValidationSession <ValidationSession>` object 
and therefore you can access all the attributes and methods of the associated :ref:`Dataset object <Dataset>` directly from the 
created :ref:`ValidationSession <ValidationSession>` object.

Once a :ref:`ValidationSession <ValidationSession>` object is created, then you can append as many simulation results 
as you want to it and validation metrics are automatically computed against the stored dataset 
and stored in the attribute :py:attr:`~dymoval.validation.validation_results`.

The currently implemented validation metrics in *Dymoval* are  

- R-square fit (%)
- Residuals auto-correlation
- Input-Residuals cross-correlation 

It is also possible to visually inspect the residuals through the :py:meth:`~dymoval.validation.plot_residuals` method.

To know more about the selected validation metrics, feel free to google *R-square* and why alone it is not enough to validate a model. 