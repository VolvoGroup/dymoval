Validate your model
===================

It is now time to validate your model.
For performing such a task, *Dymoval* uses :ref:`ValidationSession` objects.

A :ref:`ValidationSession` object is created by using a :ref:`Dataset <Dataset>` as basis.
Once instantiated, you can append as many simulated data as you want to the same :ref:`ValidationSession` object. 
The validation metrics are automatically computed for each simulated dataset against the common stored  :ref:`Dataset <Dataset>`.  


.. figure:: ../figures/ModelValidationDymoval.svg
   :scale: 50 %

   The model validation process with *Dymoval*. 
 

*Dymoval* validates your models in terms of 

- R-square fit
- Residuals auto-correlation norm
- Input-Residuals cross-correlation norm 

You can visually inspect both the simulations results with the :py:meth:`~dymoval.validation.ValidationSession.plot_simulations` method and the residuals with the :py:meth:`~dymoval.validation.ValidationSession.plot_residuals` method. 

The **coverage region** can be shown through the :py:meth:`~dymoval.dataset.Dataset.plot_coverage()` of the stored :ref:`Dataset <Dataset>`.

How to interpret the results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As a rule of thumbs, your model is as good as the r-squared index is high and the residuals correlation norms are small (possibly less than 1). 

Futhermore

- High values of the residuals auto-correlation -> your **disturbance**-to-output model needs improvement,
- High values of the input-residuals cross-correlation -> your **input**-to-output model needs improvement.


For more information on how to interpret r-squared fit and residuals, feel free to search the web or to read some good System Identification textbook. 

At this point, you may consider to use *Dymoval* for performing unit-tests on your models and create building pipelines if you are using a CI/CD environment for developing models. 


