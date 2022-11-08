<div align="center">

<img src="https://github.com/VolvoGroup/dymoval/blob/main/docs/source/DymovalLogo3.png" data-canonical-src="https://github.com/VolvoGroup/dymoval/blob/main/docs/source/DymovalLogo3.png" width="800" class="center" />


</div>

### Build status
![pipeline](https://github.com/VolvoGroup/dymoval/actions/workflows/pipeline.yml/badge.svg)
![coverage badge](./coverage.svg)

### Tools
[![Hatch project](https://img.shields.io/badge/build-hatch-4051b5.svg)](https://github.com/pypa/hatch) 
[![code check - flake8](https://img.shields.io/badge/checks-flake8-green.svg)](https://pypi.org/project/flake8)
[![types - Mypy](https://img.shields.io/badge/types-mypy-orange.svg)](https://github.com/python/mypy) 
[![test - pytest](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](https://github.com/pytest-dev/pytest)
[![code style - black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
[![docs - sphinx](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://github.com/sphinx-doc/sphinx)
-----

## What is it?

**Dymoval**  (**DY**namic **MO**del **VAL**idation) is a Python package for  *validating models* and *analyzing datasets*. 

Dymoval validates models against some user-provided dataset regardless of the model format (*DNN, transfer function, ODE,* etc.) and the tool 
used for developing it (*Simulink, Modelica,* etc.). 
If your development process is done in a CI/CD environment, *Dymoval*'s functions can be easily included in a development pipeline to test your changes. 
It can be also used to unit-test your models.

*Dymoval* further provides a number of functions for dataset analysis and manipulation.  

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
	- R-square
	- Residuals auto-correlation
	- Input-Residuals cross-correlation 
- Coverage region
- Works for both SISO and MIMO models
- Modeling tool independency
- Easily integrates with CI/CD pipelines.


## Installation
By running 

    pip install dymoval

everything should workd, but there are few things to keep in mind.
Given that `conda` handles scientific packages better than `pip`, and given that many *dymoval* dependencies are scientific packages, it is suggested to install all the dependencies through `conda` and then to install *dymoval* through `pip`.

To do that, download the `environment.yml` file from [here](https://github.com/VolvoGroup/dymoval/blob/main/environment.yml) and run 

    conda env update --name env_name --file environment.yml
    pip install dymoval

where `env_name` is the environment name where you want to install *dymoval*.
If not provided, *dymoval* will be installed in a new environment called `dymoval`.

Unfortunately, it is not possible (yet?) to easily build `conda` packages when the project is handled thorugh a `pyproject.toml` file, and therefore the *dymoval* package, which uses a `pyproject.toml` file, is only available through `pip`.


#### From this repo
Clone this repo and run

	cd /path/to/where/you/cloned/this/repo
	conda env update --name env_name --file environment.yml
    conda actiate env_name
	pip install .

or 

	cd /path/to/where/you/cloned/this/repo
	pip install .


## Getting started

If you are already familiar with model validation, then the best way to get started with dymoval is to run the tutorial scripts that you can open with

	import dymoval as dmv
	dmv.open_tutorial()

For more info on model validation you can check the [docs](https://volvogroup.github.io/dymoval/). 


> **Note**
> The tutorial is stored in a Jupyter notebook, so you need an opener for that kind of files.

## License
Dymoval is licensed under [BSD 3](https://github.com/VolvoGroup/dymoval/blob/main/LICENSE) license.
