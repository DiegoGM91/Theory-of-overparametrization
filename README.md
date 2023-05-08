# Theory of overparametrization in QNNs

<div style="text-align: justify">
This repository contains the necessary code to reproduce all the numerical experiments presented in the manuscript ["Theory of overparametrization in quantum neural networks"](https://arxiv.org/abs/2109.11676), along with the raw data employed in the paper.
</div>

## 1. System requirements

### Operating systems

- Linux
- Mac
- Windows

### Software dependencies

- `Pyhton>=3.7`

- `qibo==0.1.6`

- `tensorflow>=2.4.1`

- (Optional) `qibojit==0.0.1`


### Non-standard hardware required

No

## 2. Installation guide

### Instructions

The Python package `qibo` can be installed using `pip`:

```
pip instal qibo==0.1.6
```

This will install all the required dependencies, execpt for `tensorflow`, which you can install using

```
pip install tensorflow
```

If you want to also install the fast `qibojit` backend that we used for the computation of the quantum Fisher information and Hessian matrices, use

```
pip install qibojit==0.0.1
```

Then, you need to download the `.py` files in the folder `scr` of this repository.

### Typical instalation time in a desktop computer

1 minute

## 3. Demo

### Instructions to run on data

<div style="text-align: justify">
If you want to run an instance of a Variational Quantum Eigensolver (VQE) on the transverse field Ising model with periodic boundary conditions using a Hamiltonian Variational Ansatz (HVA), you can open a terminal, go to the folder containing this repository, and run the `example.py` file, with the following command:
</div>

```
python3 example.py --nqubits 6 --nlayers 6 --nsteps 2000 --g 1.1
```

The arguments are:

`nqubits (int)`: number of qubits.

`nlayers (int)`: number of layers of the circuit.

`nsteps (int)`: maximum number of allowed optimization steps for the Adam algorithm, `default=3000`.

`g (float)`: strength of the tansverse field, `default==1`.


### Expected output

The expected output is a printed statement showing:

i) `(numpy.ndarray:float)` final residual energy (i.e. difference with the exact ground state energy)

ii) `(numpy.ndarray:float)` optimal angles found for the variational quantum circuit



### Expected runtime for demo on a desktop computer

5 minutes for an instance with 6 qubits, 6 layers and 3000 optimization steps.

## 4. Instructions for use

### How to run the software

<div style="text-align: justify">
There are three main classes in this software, each corresponding to an example shown in the paper and each implemented on a different `.py` file, namely:

- `HVA_Ising`: implements a VQE using a Hamiltonian Variational Ansatz applied to the transverse field Ising model.

- `HEA_QAQC`: implements a compilation of a Haar random unitary matrix using a Hardware Efficient Ansatz and the Quantum Assisted Quantum Compilng algorithm.

- `HEA_Autoencoder`: implements a quantum autoencoder using a Hardware Efficient Ansatz on a given training set.

In order to intialize the classes:

- `HVA_Ising`
```
from hva.py import HVA_Ising

hva = HVA_Ising(nqubits, nlayers, g, periodic)
```

where:

`nqubits (int)`: number of qubits.

`nlayers (int)`: number of layers of the circuit, `default==1`.

`g (float)`: strength of the transverse field, `default==1`.

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

`training_states (list:int)`: ids of the states from the training set used, `default==[0,5,12,2]`.


Once initialized, if you want to run a minimization using the Adam algorithm:

- `result = hva.minimize(optimizer, options)`

- `result = hea.minimize(optimizer, options)`

- `result = autoencoder.minimize(optimizer, options)`

where:

`optimizer (str)`: optimizer used for the minimization, `default==sgd` which stands for stochastic gradient descent.

`options (dic)`: dictionary with options accepeted by the optimizer. For the `sgd` algorithms, we use by default: `'optimizer':Adam`, `'learning_rate': 1e-2`, `'nepochs':3000`, `'nmessage': 1000`.


This will return a tuple with

- `(float)`: final residual energy (ie. difference with the exact ground state energy).

- `(numpy.ndarray:float)`: optimal angles found for the variational quantum circuit.

If you want to access the values of residual energies found during the optimization process, you can type:
</div>

```
hva.loss_record

hea.loss_record

autoencoder.loss_record
```


### Reproduction instructions

TBA


