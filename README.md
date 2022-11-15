<div align="center">

<img src="https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalLogo.svg" data-canonical-src="[https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalLogo.svg](https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalLogo.svg)" width="800" class="center" />


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

**Dymoval**  (**Dy**namic **Mo**del **Val**idation) is a Python package to evaluate how "good" are your models.

It does not matter if your model is a Deep Neural Network, an ODE or something more complex, nor it is important if you use Modelica or Simulink or whatever as modeling tool. 
All you need to do is to feed *Dymoval* with real-world measurements and model-generated data and you will get a model quality evaluation in terms of r-squared fit, residuals norms and coverage region.


<div align="center" >
	<br>
	<br>
<img src="https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalNutshell.svg" data-canonical-src="[https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalNutshell.svg](https://github.com/VolvoGroup/dymoval/blob/main/docs/source/DymovalNutshell.svg)" width="600" class="center"  />
	<br>
	<br>
	<br>
</div>


If you are developing your models in a CI/CD environment, then *Dymoval* can help you in deciding if merging or not the latest model changes.
In-fact, *Dymoval* functions can also be included in development pipelines scripts, so the whole CI/CD process can be fully automatized. 

*Dymoval* finally provides you with some essential functions for dealing with dataset analysis and manipulation.  

Although the tool has been thought with engineers in mind, no one prevents you to use it in other application domains. 



## Main Features

**Model validation**

- Validation metrics:
	- R-square fit
	- Residuals auto-correlation
	- Input-Residuals cross-correlation 
- Coverage region
- Enable model unit-tests
- Work for both SISO and MIMO models
- Modeling tool independence
- Easily integrate with CI/CD pipelines.

 **Datasets analysis and manipulation**
- Time and frequency analysis 
- Missing data handling
- Linear filtering
- Means and offsets removal
- Re-sampling

## Installation
By running 

    pip install dymoval

everything should work, but there are few things to keep in mind.

Typically `conda` handles scientific packages better than `pip`, and given that many *dymoval* dependencies are scientific packages, it is suggested to install all the dependencies through `conda` and then to install *dymoval* through `pip`.

To do that, download the `environment.yml` file from [here](https://github.com/VolvoGroup/dymoval/blob/main/environment.yml) and run 

    conda env update --name env_name --file environment.yml
    pip install dymoval

where `env_name` is the environment name where you want to install *dymoval*.
If not provided, *dymoval* will be installed in a new environment called `dymoval`.

#### Why not `conda install dymoval`?
Unfortunately, it is not possible (yet?) to easily build `conda` packages when the project is handled through a `pyproject.toml` file, and therefore the *dymoval* package, which uses a `pyproject.toml` file, is only available through `pip`.


#### Installation From this repo
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


This will copy the `dymoval_tutorial.ipynb` jupyter notebook from your installation folder to your `home` folder.

For more info on what is model validation and what is currently implemented in *dymoval* along with the *dymoval* complete API, you can check the [docs](https://volvogroup.github.io/dymoval/). 


## License
Dymoval is licensed under [BSD 3](https://github.com/VolvoGroup/dymoval/blob/main/LICENSE) license.
