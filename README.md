# Theory of overparametrization in QNNs

## 1. System requirements

### Operating systems

- Linux
- Mac
- Windows

### Software dependencies

- `Pyhton>=3.6`

- `qibo==0.1.6`

- `numpy==`

- `tensorflow==2.`

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


## 3. Demo

### Instructions to run on data

If you want to run an instance of a Variational Quantum Eigensolver (VQE) on the transverse field Ising model using a Hamiltonian Variational Ansatz (HVA), you can open a terminal, go to the folder containing this repository, and run the `example.py` file, with the following command:

```
python3 example.py --nqubits 6 --nlayers 6 --steps 2000 --lambda 1.1 --nthreads 1
```

The arguments are:

`nqubits (int)`: number of qubits.

`nlayers (int)`: number of layers.

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

There are three main classes in this software, each corresponding to an example shown in the paper, namely:

- `HVA_Ising`:

- `HEA_QAQC`:

- `HEA_Autoencoder`

In order to intialize the classes:

- `HVA_Ising()`

- `HEA_QAQC()`

- `HEA_Autoencoder()`

Once initialized, if you want to run a minimization using the Adam algorithm:

- 

This will return a tuple with

-

-


### Reproduction instructions

TBA


