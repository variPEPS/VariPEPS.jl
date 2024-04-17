# VariPEPS.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10974460.svg)](https://zenodo.org/doi/10.5281/zenodo.10974460)

A Julia code for variational iPEPS using automatic differentiation based on the algorithms and techniques described in this lovely [review](https://arxiv.org/abs/2308.12358). 

The foundational tensor-structures used in this package are based on the [TensorKit.jl](https://jutho.github.io/TensorKit.jl/latest/) library. Check out the [software-website](https://quantumghent.github.io/software/) of the Quantum Group from Ghent to find other packages. 

This is version 0.1 of the code and the package is still subject to continuous change of both functional and aesthetic nature.

## Installation

download and develop.

## Scope of the code

The codes most basic uses are the contraction of an *infinite PEPS* using the *corner transfer-matrix renormalization group* (CTMRG) algorithm.
This also allows the calculation of the energy and other expectation values one might be interested in. Some basic models are implemented but the user can also 
specify their own models and desired expectation values. 

For the purpose of variationally searching for the ground state of a specified Hamiltonian the code also allows us to find the gradient of a specified *iPEPS* and a model
Hamiltonian. 

The basic model for each lattice will be listed here.

There are additional features in the code of which there will be a list here in the future.

## Examples
A few notebooks (in the examples folder) give some notes on how to use the code:

    1. `basic_functionality.ipynb`: Shows how to use the most basic functions in the code. We find the ground state of the Heisenberg Anti-ferromagnet as an example.
    2. `use_your_own_model.ipynb`: Shows how to use models that are not specified already in the code.
    3. `different_lattices.ipynb`: Will show how to use the code for lattices different from the square case.
    4. some additional tutorials (how to dynamically increase environment_bond dimension, how to enforce local symmetries...) will follow

## Citation

We are happy if you want to use the framework for your research. For the citation of our work we ask to use the following references (the publication with the method description, the Zenodo reference for this Git repository):
* J. Naumann, E. L. Weerda, M. Rizzi, J. Eisert, and P. Schmoll, variPEPS -- a versatile tensor network library for variational ground state simulations in two spatial dimensions (2023), [arXiv:2308.12358](https://arxiv.org/abs/2308.12358).
* J. Naumann, P. Schmoll, F. Wilde, and F. Krein, [variPEPS (Python version)](https://zenodo.org/doi/10.5281/zenodo.10852390), Zenodo.

The BibTeX code for these references are:
```bibtex
@misc{naumann23_varipeps,
    title =         {variPEPS -- a versatile tensor network library for variational ground state simulations in two spatial dimensions},
    author =        {Jan Naumann and Erik Lennart Weerda and Matteo Rizzi and Jens Eisert and Philipp Schmoll},
    year =          {2023},
    eprint =        {2308.12358},
    archivePrefix = {arXiv},
    primaryClass =  {cond-mat.str-el}
}

@software{varipeps_julia,
    author =        {Erik Weerda},
    title =         {{variPEPS (Julia version)}},
    howpublished =  {Zenodo},
    url =           {https://doi.org/10.5281/zenodo.10974460},
    doi =           {10.5281/zenodo.10974460},
}
```
