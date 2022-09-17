![Dymoval logo](https://github.com/VolvoGroup/dymoval/blob/main/docs/source/DymovalLogo.png)

What is it?
-
**Dymoval**  (**DY**namic **MO**del **VAL**idation) is a Python package for  *validating models* and *analyzing datasets*. 

Dymoval validates models against some user-provided dataset regardless of their format (DNN, transfer function, ODE, etc.) and the tool  used for developing them (Simulink, Modelica, etc.). 
If your development process is done on a CI/CD environment, dymoval's functions can be easily included in a development pipeline to test your changes. 

Dymoval further provides a number of functions for dataset analysis and manipulation.  

Although it is a very helpful in engineering, its usage may come handy in other application domains as well. 



## Main Features

 **Datasets analysis and manipulation**
- Time and frequency analysis 
- Missing data handling
- Linear filtering
- Means and offsets removal
- Re-sampling

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


## Installation


    The tool still needs some adjustments before it can be release in typical distribution channels. 
	Hence, at the moment it is only available on GitHub.


## Getting started

The best way to learn *dymoval* is to run the tutorial scripts but perhaps you may want to 
have a look at the [documentation](https://volvogroup.github.io/dymoval/).

## License
Dymoval is licensed under [BSD 3](https://github.com/VolvoGroup/dymoval/blob/main/LICENSE) license.