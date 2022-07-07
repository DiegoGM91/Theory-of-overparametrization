#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
import qibo
from qibo.models import Circuit
from qibo import gates
from qibo.optimizers import optimize
qibo.set_backend("matmuleinsum") # qibo version==0.1.5
qibo.set_threads(1)


class Autoencoder:
    """
    Class for a quantum autoencoder.
    """
    
    def __init__(self, nqubits, layers, trash=2, train_set=[0,5,12,2]):
        """
        Args:
            nqubits (int): number of qubits.
            nlayers (int): number of layers of the ansatz.
            trash (int): number of trashed qubits.
            train_set (list:int:): indices of the training set states.
        """
        self.nqubits = nqubits
        self.train_set = train_set
        self.train_size = len(train_set)
        self.loss_record = []
        # Input state
        self.rho = np.zeros((self.train_size, 2**nqubits), dtype=np.complex128)
        for i,psi in enumerate(train_set):
            self.rho[i] = np.load(f'final_results/autoencoder/train_set/state_{psi}.npy')
        # Circuit
        self.hea = Circuit(nqubits)
        self.nlayers = layers
        self.trash = trash
        # Hardware-Efficient Ansatz
        pairs1 = [(i, i + 1) for i in range(0, nqubits-1, 2)]
        pairs2 = [(i, i + 1) for i in range(1, nqubits-1, 2)]
        self.hea.add((gates.RY(q, theta=0) for q in range(nqubits))) # Initial rotations
        self.hea.add((gates.RX(q, theta=0) for q in range(nqubits)))
        if self.nqubits % 2 == 0:
            for _ in range(layers):
                self.hea.add((gates.CZ(pair[0], pair[1]) for pair in pairs1))
                self.hea.add((gates.RY(q, theta=0) for q in range(nqubits)))
                self.hea.add((gates.RX(q, theta=0) for q in range(nqubits)))
                self.hea.add((gates.CZ(pair[0], pair[1]) for pair in pairs2))
                self.hea.add((gates.RY(q, theta=0) for q in range(1, nqubits-1)))
                self.hea.add((gates.RX(q, theta=0) for q in range(1, nqubits-1)))
        else:
            for _ in range(layers):
                self.hea.add((gates.CZ(pair[0], pair[1]) for pair in pairs1))
                self.hea.add((gates.RY(q, theta=0) for q in range(1, nqubits-1)))
                self.hea.add((gates.RX(q, theta=0) for q in range(1, nqubits-1)))
                self.hea.add((gates.CZ(pair[0], pair[1]) for pair in pairs2))
                self.hea.add((gates.RY(q, theta=0) for q in range(nqubits)))
                self.hea.add((gates.RX(q, theta=0) for q in range(nqubits)))     
        
    def _cost_function(self, parameters):
        """
        Cost function for the optimization of the variational circuit.
        
        Args:
            parameters (array): angles for the gates in the ansatz.
            
        Returns:
            (float): value of the cost function.
        """
        loss = 0
        self.hea.set_parameters(parameters)
        for psi in range(self.train_size):
            trial_state = self.hea(self.rho[psi])
            # Compute loss
            loss += 1 - tf.reduce_sum(tf.math.square(tf.math.abs(trial_state[0:2**(self.nqubits-self.trash)])))
        loss /= self.train_size
        self.loss_record.append(loss.numpy())
        
        return loss    
      
    def minimize(self, optimizer='sgd', options={'optimizer': 'Adam', 'learning_rate': 1e-2, 'nepochs':int(2e3), 'nmessage': 100}):
        """
        Optimize the varitational parameters.
        
        Args:
            optimizer (string): optimization method employed.
            options (dict): options admited by the optimizer.
            
        Returns:
            (float, array): minimum, optimal_parameters.
        """
        # Initial random angles
        np.random.seed(None)
        init_angles = tf.Variable(2*np.pi * np.random.rand(2*self.nlayers*(2*self.nqubits-2) + 2*self.nqubits), dtype=tf.float64)
        # Optimize
        min_energy, best_params, _ = optimize(self._cost_function, init_angles, method=optimizer, options=options)
        
        return min_energy, best_params
     
    def qfim(self, parameters, init_state, rows=None):
        """
        Compute the QFI matrix of the ansatz evaluated at parameters.
        
        Args:
            parameters (numpy.1darray): values of the parameters at which the Hessian is evaluated.
            rows (numpy.1darray): rows of the matrix to be computed (this argument is used to parallelized the com
                                  the computation. If None, the full matrix is computed).  
        Returns:
            (numpy.2darray): QFI matrix.
        """
        # Construct qfim
        nparams = len(parameters)
        qfim = np.zeros(shape=(nparams, nparams))
        self.hea.set_parameters(parameters)
        state_bra = np.conjugate(self.hea(init_state).numpy())
        if rows == None:
            rows = range(nparams)
        for i in rows:
            print('i: ', i)
            # Parameter shift
            parameters[i] += np.pi/2
            for j in range(i, nparams):
                # Parameter shift
                parameters[j] += np.pi/2
                self.hea.set_parameters(parameters)
                state = self.hea(init_state).numpy()
                # Add contribution
                contribution = np.abs(np.inner(state_bra, state))**2
                qfim[i][j] += contribution
                qfim[j][i] += contribution
                # Parameter shift
                parameters[j] -= np.pi
                self.hea.set_parameters(parameters)
                state = self.hea(init_state).numpy()
                # Add contribution
                contribution = np.abs(np.inner(state_bra, state))**2
                qfim[i][j] -= contribution
                qfim[j][i] -= contribution
                # Revert parameter shift
                parameters[j] += np.pi/2
            # Parameter shift
            parameters[i] -= np.pi
            for j in range(i, nparams):
                # Parameter shift
                parameters[j] += np.pi/2
                self.hea.set_parameters(parameters)
                state = self.hea(init_state).numpy()
                # Add contribution
                contribution = np.abs(np.inner(state_bra, state))**2
                qfim[i][j] -= contribution
                qfim[j][i] -= contribution
                # Parameter shift
                parameters[j] -= np.pi
                self.hea.set_parameters(parameters)
                state = self.hea(init_state).numpy()
                # Add contribution
                contribution = np.abs(np.inner(state_bra, state))**2
                qfim[i][j] += contribution
                qfim[j][i] += contribution
                # Revert parameter shift
                parameters[j] += np.pi/2
            # Revert parameter shift
            parameters[i] += np.pi/2
            # Correct the double addition of the diagonal contributions
            qfim[i][i] /= 2
        
        return -qfim/8