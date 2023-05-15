<div align="center">

<img src="https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalLogo.svg" data-canonical-src="[https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalLogo.svg](https://github.com/VolvoGroup/dymoval/blob/main/docs/source/figures/DymovalLogo.svg)" width="800" class="center" />


</div>

### Build status
![pipeline](https://github.com/VolvoGroup/dymoval/actions/workflows/pipeline.yml/badge.svg)
![coverage badge](./coverage.svg)

### Tools
[![Build - pdm](https://img.shields.io/badge/build-pdm-blueviolet)](https://pdm.fming.dev/latest/)
[![code check - flake8](https://img.shields.io/badge/checks-flake8-green.svg)](https://pypi.org/project/flake8)
[![types - Mypy](https://img.shields.io/badge/types-mypy-orange.svg)](https://github.com/python/mypy)
[![test - pytest](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](https://github.com/pytest-dev/pytest)
[![code style - black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![docs - sphinx](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://github.com/sphinx-doc/sphinx)
-----

## What is it?

**Dymoval**  (**Dy**namic **Mo**del **Val**idation) is a Python package for analyzing *datasets* and validate *models*.

It does not matter if your model is a Deep Neural Network, an ODE or something more complex, nor is important if you use Modelica or Simulink or whatever as modeling tool.
All you need to do is to feed *Dymoval* with real-world measurements and model-generated data and you will get a model quality evaluation in r-squared fit, residuals norms and coverage region.


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

 **Datasets analysis and manipulation**
- Time and frequency analysis
- Physical units
- Easy plotting of signals
- Missing data handling
- Linear filtering
- Means and offsets removal
- Re-sampling

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


## Installation
Clone this repo, go on the `dymoval` folder and type:

    pip istall .

If you also want to install `dev` and `build` tools, run

    pip istall ".[dev]"
    pip istall ".[build]"

The build tool included in `pyproject.toml` is `pdm` but you can use the build
tool that you prefer (to be installed separately).

<!-- comment
#### Installation From this repo
Clone this repo and run

	cd /path/to/where/you/cloned/this/repo
	conda env update --name env_name --file environment.yml
    conda actiate env_name
	pip install .

or

	cd /path/to/where/you/cloned/this/repo
	pip install .
-->

## Getting started

If you are already familiar with model validation, then the best way to get started with dymoval is to run the tutorial scripts that you can open with

	import dymoval as dmv
	dmv.open_tutorial()


This will copy the `dymoval_tutorial.ipynb` Jupyter notebook from your installation folder to your `home` folder.

For more info on what is model validation and what is implemented in *dymoval* along with the *dymoval* complete API, you can check the [docs](https://volvogroup.github.io/dymoval/).

> **Note**
> If your tutorial won't start, you can manually download the tutorial Jupyter notebook from this repo.

## License
Dymoval is licensed under [BSD 3](https://github.com/VolvoGroup/dymoval/blob/main/LICENSE) license.
