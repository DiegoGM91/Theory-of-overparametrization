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

- `HVA_Ising`: implements a VQE using a Hamiltonian Variational Ansatz applied to the transverse field Ising model.

- `HEA_QAQC`: implements a compilation of a Haar random unitary matrix using a Hardware Efficient Ansatz and the Quantum Assisted Quantum Compilng algorithm.

- `HEA_Autoencoder`: implements a quantum autoencoder using a Hardware Efficient Ansatz on a given training set.

In order to intialize the classes:

- `HVA_Ising`
```
from hva.py import HVA_Ising

hva = HVA_Ising(nqubits, nlayers, lambda, periodic)
```

where:

`nqubits (int)`: number of qubits.

`nlayers (int)`: number of layers of the circuit, `default==1`.

`lambda (float)`: strength of the transverse field, `default==1`.

`periodic (bool)`: whether periodic or non-periodic boundary conditions apply, `default==True`.

- `HEA_QAQC`
```
from hea.py import HEA_QAQC

hea = HEA_QAQC(nqubits, nlayers, seed)
```

where:

`nqubits (int)`: number of qubits.

`nlayers (int)`: number of layers of the circuit, `default==1`.

`seed (int)`: random seed to generate the Haar random unitary, used for reproducibility, `default==1`.


- `HEA_Autoencoder`

```
from autoencoder.py import Autoencoder

autoencoder = Autoencoder(nqubits, nlayers, trash_space, training_states)
```

where:

`nqubits (int)`: number of qubits.

`nlayers (int)`: number of layers of the circuit.

`trash_space (int)`: number of qubits used as trashed space, `default==2`.

`training_states (list:int)`: ids of the states from the training set used, `default==[0,5,12,2}`.


Once initialized, if you want to run a minimization using the Adam algorithm:

- `hva.minimize(optimizer, options)`

- `hea.minimize(optimizer, options)`

- `autoencoder.minimize(optimizer, options)`

where:

`optimizer (str)`: optimizer used 


This will return a tuple with

-

-


### Reproduction instructions

TBA


