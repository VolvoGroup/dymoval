<div align="center">

<img src="https://github.com/VolvoGroup/dymoval/blob/main/docs/source/DymovalLogo3.png" data-canonical-src="https://github.com/VolvoGroup/dymoval/blob/main/docs/source/DymovalLogo3.png" width="800" class="center" />


</div>

### Build status
![pipeline](https://github.com/VolvoGroup/dymoval/actions/workflows/pipeline.yml/badge.svg)
![docs](https://github.com/VolvoGroup/dymoval/actions/workflows/docs.yml/badge.svg)
![coverage badge](./coverage.svg)

### Tools
[![Hatch project](https://img.shields.io/badge/build-hatch-4051b5.svg)](https://github.com/pypa/hatch) 
[![code check - flake8](https://img.shields.io/badge/lint-flake8-green.svg)](https://pypi.org/project/flake8)
[![types - Mypy](https://img.shields.io/badge/types-mypy-orange.svg)](https://github.com/python/mypy) 
[![test - pytest](https://img.shields.io/badge/test-pytest-brightgreen.svg)](https://github.com/pytest-dev/pytest)
[![format - black](https://img.shields.io/badge/format-black-000000.svg)](https://github.com/psf/black) 
[![doc - sphinx](https://img.shields.io/badge/doc-sphinx-blue.svg)](https://github.com/sphinx-doc/sphinx)
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
	- R-square fit (%)
	- Residuals auto-correlation
	- Input-Residuals cross-correlation 
- Coverage region
- Works for both SISO and MIMO models
- Modeling tool independency
- Easily integrates with CI/CD pipelines.


## Installation


The tool is still under development and therefore there are no release available yet.
However, if you want to give it a shot, you can just clone this repo and run

	cd /path/to/where/you/cloned/this/repo
	pip install -e .


## Getting started

If you are already familiar with model validation, then the best way to learn dymoval is to run the tutorial scripts that you can open with

	import dymoval as dmv
	dmv.open_tutorial()


**UPDATE** The tutorial won't work in its current status. You can take a look at the [documentation](https://volvogroup.github.io/dymoval/) in the meantime.


Note
----

The tutorial is stored in a Jupyter notebook, so you need an opener for that kind of files.

## License
Dymoval is licensed under [BSD 3](https://github.com/VolvoGroup/dymoval/blob/main/LICENSE) license.
