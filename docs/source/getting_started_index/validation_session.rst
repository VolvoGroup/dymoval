Validate your model
===================

Once you have both a cleaned-up dataset and the corresponding simulation results, you can evaluate your model
through the :ref:`ValidationSession`.

You instantiate a ValidationSession object through a :ref:`Dataset` instance.
Once created, you can append as many simulation results as you want to the same :ref:`ValidationSession`. instance.

For each appended simulation result, validation metrics are automatically computed against the stored dataset.
In this way you can automatically see how "good" is your model. 
Remember that the dataset also include the coverage region that certifies where it is safe to use your model.

The currently implemented validation metrics in *dymoval* are  

- R-square fit (%)
- Residuals auto-correlation
- Input-Residuals cross-correlation 

*Dymoval* also allows you to visually inspect the residuals through the :py:meth:`~dymoval.validation.plot_residuals` method. 