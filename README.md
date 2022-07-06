# Theory of overparametrization in QNNs

## 1. System requirements

### Operating systems

- Linux
- Mac
- Windows

### Software dependencies

- Pyhton 

- qibo==1.

- numpy==

- tensorflow==2.

### Non-standard hardware required

No

## 2. Installation guide

### Instructions

The Python package `qibo` can be installed using `pip`:

```
pip instal qibo==1.
```

This will install all the required dependencies. Then, you need to download the `.py` files in the folder `scr` of this repository.

### Typical instalation time in a desktop computer


## 3. Demo

### Instructions to run on data

If you want to run an instance of a Variational Quantum Eigensolver (VQE) on the transverse field Ising model using a Hamiltonian Variational Ansatz (HVA), you can open a terminal, go to the folder containing the `scr` folder, and run the `example.py` file, with the following command:

```
python3 example.py --nqubits 6 --nlayers 6 --steps 2000 --lambda 1.1 --nthreads 1
```

The arguments are:

`nqubits (int)`: number of qubits.

`nlayers (int)`: number of layers.

`steps (int)`: maximum number of allowed optimization steps for the Adam algorithm.

`lambda (float)`: strength of the tansverse field, `default==1`.

`nthreads (int)`: number of threads used to simulate the quantum circuits (it must not be larger than the number of logical cores of the CPU), `default==1`


### Expected output

The expected output is a tuple with:

i) `(float)` energy of the state (i.e. value of the cost function)

ii) `(qibo.state)` final quantum state as an state vector object from `qibo`

iii) `(numpy.ndarray:float)` energies during the optimization process


### Expected runtime for demo on a desktop computer


## 4. Instructions for use

### How to run the software on your data

To run this software, 

### Reproduction instructions


