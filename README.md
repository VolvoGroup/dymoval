What is it?
-
**Dymoval**  (**DY**namic **MO**del **VAL**idation) is a Python package for  *validating models* and *analyzing datasets*. 

Dymoval validates model against some user-provided dataset regardless of their format (DNN, transfer function, ODE, etc.) and the tool  used for developing them (Simulink, Modelica, etc.). 
If your development process is done on a CI/CD environment, dymoval's functions can be easily included in a development pipeline to test your changes. 

Dymoval further provides a number of functions for dataset analysis and manipulation.  

Although it is a very helpful in engineering, its usage may come handy in other application domains as well. 


 

## Main Features

**Model validation**

- Models unit-tests
- Validation metrics:
	- R-square fit (%)
	- Residuals auto-correlation
	- Input-Residuals cross-correlation 
- Coverage region
- Modeling tool independency
- Works for both SISO and MIMO models
- Easily integrates with CI/CD pipelines (Jenkins, GitLab, etc.) 



**Datasets analysis and manipulation**
- Time and frequency analysis 
- Missing data handling
- Linear filtering
- Means and offsets removal
- Re-sampling



## Installation


    pip install dymoval


## Usage

The best way to learn dymoval is to run the tutorial scripts and to take a look at the user guide. 
Nevertheless, we try to 

You must create a dymoval Dataset to test your models based on some log-data collected from the real-world. Such log-data could be some measurement time-series picked by some sensor during some experiments. To create a dymoval Dataset, you must specify which are the input and output signals, their sampling intervals and you can add optional attributes such as units. 

Once you have a dymoval Dataset, then you can analyze it both time and frequency domain, you can analyze its coverage  and you can manipulate it  by applying filters, removing means, etc. to make it suitable for your model and for validation purposes. 

The next step is to create an empty dymoval ValidationSession based on your (possibly) adjusted Dataset. 
Then, you shall simulate your model with the Dataset  input signals included in the current ValidationSession and then you shall append the simulation results to the same ValidationSession. 
The validation metrics are automatically computed. You can append as many simulations results as you want in the same ValidationSession. 

Note that your model is simulated externally to dymoval and therefore you can use the tool of your choice (Simulink, Modelica, CarSim, GT-suite, etc).
Dymoval only requires the time-series representative of the simulation results, it does not care which tool or model is used to generate it. 

## License
[BSD 3](https://github.com/pandas-dev/pandas/blob/main/LICENSE)