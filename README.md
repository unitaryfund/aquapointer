[![PyPI version](https://badge.fury.io/py/aquapointer.svg)](https://badge.fury.io/py/aquapointer)
[![Downloads](https://static.pepy.tech/personalized-badge/aquapointer?period=total&units=international_system&left_color=black&right_color=green&left_text=Downloads)](https://pepy.tech/project/aquapointer)
[![Repository](https://img.shields.io/badge/GitHub-5C5C5C.svg?logo=github)](https://github.com/unitaryfund/aquapointer)
[![Wellcome Leap](https://img.shields.io/badge/Supported%20By-Wellcome%20Leap-FF2C4C.svg)](https://wellcomeleap.org)
[![Discord Chat](https://img.shields.io/badge/dynamic/json?color=blue&label=Discord&query=approximate_presence_count&suffix=%20online.&url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2FJqVGmpkP96%3Fwith_counts%3Dtrue)](http://discord.unitary.fund)


# aquapointer
Package applying quantum algorithms to find locations of water molecules in a protein cavity.

## Getting started
You can use [this notebook](notebooks/aquapointer_demo.ipynb) to get started with aquapointer.

## Installing aquapointer
Install the latest released version of Aquapointer via the command `pip install aquapointer`.

Alternatively, the development install can be done by setting the working directory to the top level of the repository and running `pip install -e .`

You can also run the setup command from the top level of the repository, where the `setup.py` file is found
```
python setup.py install
```
Finally, a Dockerfile is availble for running the automated script in a container.

## Documentation

See https://aquapointer.readthedocs.io/en/latest/ for the latest documentation!

The documentation can also be found in the `docs` folder. To install the documentation you need to install `sphinx` and install the development environment.
```
pip install sphinx
pip install -e .
```

To build an html version of the documentation, go to the docs directory and run the sphinx-build command, i.e.,
```
cd docs
sphinx-build -b html source build
```

 Alternatively to the explicit sphinx-build command, you can use the `Makefile` shortcuts, such as `make html`.


## Contributing
Aquapointer is developed by [Unitary Fund](https://unitary.fund/), [PASQAL](https://www.pasqal.com/), and [Qubit Pharmaceuticals](https://www.qubit-pharmaceuticals.com/). You are welcome to contribute to the project, open an issue or a pull request.

You can join the [#aquapointer](https://discord.gg/cV4nEpMz) channel on the Unitary Fund Discord for community support.

## Funding
Work on aquapointer is supported by [Wellcome Leap](https://wellcomeleap.org/) as part of the Quantum for Bio Program ([Q4Bio](https://wellcomeleap.org/q4bio/program/)).

## License
Aquapointer is released under [GPL-3.0 license](https://github.com/unitaryfund/aquapointer#GPL-3.0-1-ov-file).
