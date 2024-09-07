[![PyPI version](https://badge.fury.io/py/aquapointer.svg)](https://badge.fury.io/py/aquapointer)
[![Downloads](https://static.pepy.tech/personalized-badge/aquapointer?period=total&units=international_system&left_color=black&right_color=green&left_text=Downloads)](https://pepy.tech/project/aquapointer)
[![Repository](https://img.shields.io/badge/GitHub-5C5C5C.svg?logo=github)](https://github.com/unitaryfund/aquapointer)
[![Wellcome Leap](https://img.shields.io/badge/Supported%20By-Wellcome%20Leap-FF2C4C.svg)](https://wellcomeleap.org)
[![Discord Chat](https://img.shields.io/badge/dynamic/json?color=blue&label=Discord&query=approximate_presence_count&suffix=%20online.&url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2FJqVGmpkP96%3Fwith_counts%3Dtrue)](http://discord.unitary.fund)


# aquapointer
An open source software package developed by the Unitary Fund team with consortium partners [Pasqal](https://www.pasqal.com/) and [Qubit Pharmaceuticals](https://www.qubit-pharmaceuticals.com/) and funding from [Wellcome Leap](https://wellcomeleap.org/).
The library is designed to be a computational tool used in pharmaceutical development, specifically for studying the placement of water molecules in protein cavities.

Proteins are complex molecules with cavities that can be occupied by water molecules, particularly in living tissue.
The presence of water molecules influences the binding of small molecules called ligands to specific protein sites, a problem of interest in drug discovery.
Protein solvation effects can be studied either by modeling the interactions experimentally, which is generally a costly and relatively inefficient process, or by using numerical models.
Classical numerical methods, such as Monte Carlo or molecular dynamics, can give some insight but the computational complexity of these methods can be too large for certain hard cases. 
An alternative approach is to find first the density distribution of water molecules, through methods such as the [3D Reference Interactive Site Model (3D-RISM)](https://pubmed.ncbi.nlm.nih.gov/23675899/). 
By looking at 2D slices of the 3D-RISM density function, we can define a discrete optimization problem (per slice) whose solutions correspond to positions of water molecules.

Aquapointer generates 2D slices of an input 3D-RISM density function, maps the slices to a QUBO problem, translates the QUBO to an analog pulse sequence or a digital circuit, and then calls the backend API and processes the results.
The analog workflow in Aquapointer uses [Pulser](https://github.com/pasqal-io/Pulser) for intermediate representations (IR) of the pulse sequences and for interfacing to supported backends, e.g. QuTip
The digital workflow uses [Qiskit](https://github.com/Qiskit) for IR and simulated backends.

![image demonstating the analog workflow in Aquapointer](/images/aquapointer_analogflow.png)

```python
water_postions = find_water_positions(canvases, executor, MockDevice, pulse_settings)
```

Since we first introduced Aquapointer, we have upgraded it to include 3D-RISM density processing, in the form of the `slicing` and `densitycanvas` modules.
The `slicing` module takes a 3D-RISM density file and transforms it into 2D slices along user-specified planes. 
The `densitycanvas` module contains classes and functions for transforming the 2D slices or generating them from a probability distribution and mapping the density distributions into a QUBO formulation.

![image demonstating the slicing workflow in Aquapointer](/images/aquapointer_slicing.png)

```python
canvases = canvases = density_slices_by_planes(grid, slicing_points)
for canvas in canvases:
    canvas.filter_density(filter_settings={"filter_function": filter_fn, "sigma": sigma})
    canvas.crop_canvas(center, size) 
```


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
You are welcome to contribute to the project by opening an issue or a pull request.

You can join the [#aquapointer](https://discord.gg/cV4nEpMz) channel on the Unitary Fund Discord for community support.

## Funding
Work on aquapointer is supported by [Wellcome Leap](https://wellcomeleap.org/) as part of the Quantum for Bio Program ([Q4Bio](https://wellcomeleap.org/q4bio/program/)).

## License
Aquapointer is released under [GPL-3.0 license](https://github.com/unitaryfund/aquapointer#GPL-3.0-1-ov-file).
