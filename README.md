# Theory of overparametrization in QNNs

## 1. System requirements

### Operating systems

- Linux
- Mac
- Windows

### Software dependencies

- `Pyhton>=3.6 and < 3.8`

- `qibo==0.1.6`


### Non-standard hardware required

No

## 2. Installation guide

### Instructions

The Python package `qibo` can be installed using `pip`:

```
pip instal qibo==0.1.6
```

This will install all the required dependencies. Then, you need to download the `.py` files in the folder `scr` of this repository.

### Typical instalation time in a desktop computer

1-2 minutes

## 3. Demo

### Instructions to run on data

If you want to run an instance of a Variational Quantum Eigensolver (VQE) on the transverse field Ising model using a Hamiltonian Variational Ansatz (HVA), you can open a terminal, go to the folder containing this repository, and run the `example.py` file, with the following command:

```
python3 example.py --nqubits 6 --nlayers 6 --steps 2000 --lambda 1.1 --nthreads 1
```

The arguments are:

`nqubits (int)`: number of qubits.

`nlayers (int)`: number of layers of the circuit.

`steps (int)`: maximum number of allowed optimization steps for the Adam algorithm, `default=3000`.

`lambda (float)`: strength of the tansverse field, `default==1`.

`nthreads (int)`: number of threads used to simulate the quantum circuits (it must not be larger than the number of logical cores of the CPU), `default==1`.


### Expected output

The expected output is a tuple containing:

i) `(numpy.ndarray:float)` energies during the optimization process

ii) `(qibo.state)` final quantum state as an state vector object from `qibo`

iii) `(numpy.ndarray:float)` optimal angles found for the variational quantum circuit



### Expected runtime for demo on a desktop computer


## 4. Instructions for use

### How to run the software on your data

There are three main classes in this software, each corresponding to an example shown in the paper and each implemented on a different `.py` file, namely:

- `HVA_Ising`: implements a VQE using a Hamiltonian Variational Ansatz applyed to the transverse field Ising model.

- `HEA_QAQC`: implements a compilation of a unitary matrix using a Hardware Efficient Ansatz and the Quantum Assisted Quantum Compilng algorithm.

- `HEA_Autoencoder`: implements a quantum autoencoder using a Hardware Efficient Ansatz on a given training set.

In order to intialize the classes:

- 
```
from hva.py import HVA_Ising

HVA_Ising(nqubits, nlayers, lambda, periodic)
```

where:

`nqubits (int)`: number of qubits.
`nlayers (int)`: number of layers of the circuit, `default==1`.
`lambda (float)`: strength of the transverse field, `default==1`.
`periodic (bool)`: whether periodic or non-periodic boundary conditions apply, `default==True`.

- 
```
from hea.py import HEA_QAQC
HEA_QAQC()
```

- `HEA_Autoencoder()`

Once initialized, if you want to run a minimization using the Adam algorithm:

- 

This will return a tuple with

-

-


### Reproduction instructions

TBA


