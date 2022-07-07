#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
from scipy.stats import unitary_group
import qibo
qibo.set_backend("matmuleinsum") # qibo version==0.1.5
qibo.set_threads(1)
from qibo import gates
from qibo.models import Circuit
from qibo.optimizers import optimize

class HEA_QAQC:
    """
    Class for the Hardware Efficient Ansatz applied to the problem of learning a unitary with QAQC.
    """
    
    def __init__(self, nqubits, layers=1, seed=1):
        """
        Args:
            nqubits (int): number of qubits.
            layers (int): number of layers of the ansatz.
            seed (int): seed for the generation of the random unitary.
        """
        self.nqubits = nqubits
        self.nlayers = layers
        self.loss_record = []
        # Create target random unitary
        np.random.seed(seed)
        self.u = unitary_group.rvs(2**self.nqubits)
        # Quantum circuit
        self.hea = Circuit(2*nqubits)
        # Create Bell Pairs
        self.hea.add((gates.H(q) for q in range(nqubits)))
        self.hea.add((gates.CNOT(q,q+nqubits) for q in range(nqubits)))
        # Hardware-Efficient Ansatz
        pairs1 = [(i, i + 1) for i in range(0, nqubits-1, 2)]
        pairs2 = [(i, i + 1) for i in range(1, nqubits-1, 2)]
        self.hea.add((gates.RY(q, theta=0) for q in range(nqubits))) # Initial rotations
        self.hea.add((gates.RX(q, theta=0) for q in range(nqubits))) # Initial rotations
        if self.nqubits % 2 == 0:
            for l in range(layers):
                self.hea.add((gates.CZ(pair[0], pair[1]) for pair in pairs1))
                self.hea.add((gates.RY(q, theta=0) for q in range(nqubits)))
                self.hea.add((gates.RX(q, theta=0) for q in range(nqubits)))
                self.hea.add((gates.CZ(pair[0], pair[1]) for pair in pairs2))
                self.hea.add((gates.RY(q, theta=0) for q in range(1, nqubits-1)))
                self.hea.add((gates.RX(q, theta=0) for q in range(1, nqubits-1)))
        else:
            for l in range(layers):
                self.hea.add((gates.CZ(pair[0], pair[1]) for pair in pairs1))
                self.hea.add((gates.RY(q, theta=0) for q in range(1, nqubits-1)))
                self.hea.add((gates.RX(q, theta=0) for q in range(1, nqubits-1)))
                self.hea.add((gates.CZ(pair[0], pair[1]) for pair in pairs2))
                self.hea.add((gates.RY(q, theta=0) for q in range(nqubits)))
                self.hea.add((gates.RX(q, theta=0) for q in range(nqubits)))
        # Add target unitary
        self.hea.add(gates.Unitary(self.u.conj(), *range(nqubits,2*nqubits), trainable=False))
        # Rotate to Bell basis
        self.hea.add((gates.CNOT(q,q+nqubits) for q in range(nqubits)))
        self.hea.add((gates.H(q) for q in range(nqubits)))     
                 
    def _cost_function(self, parameters):
        """
        Cost function for the optimization of the variational circuit.
        
        Args:
            parameters (array): angles for the gates in the ansatz.
            
        Returns:
            (float): value of the cost function.
        """
        # Set parameters and execute the circuit
        self.hea.set_parameters(parameters)
        trial_state = self.hea()
        # Compute and record loss
        loss = 1 - tf.math.abs(trial_state[0])**2
        self.loss_record.append(loss.numpy())
        
        return loss
    
    def _hess_loss(self, parameters):
        """
        State transfer loss function for the computation of the Hessian.
        
        Args:
            parameters (array): angles for the gates in the ansatz.
            
        Returns:
            (float): value of the cost function.
        """
        # Set parameters and execute the circuit
        self.hea.set_parameters(parameters)
        trial_state = self.hea()
        # Compute and record loss
        loss = tf.math.abs(trial_state[0])**2
        
        return loss.numpy()    
        
    def minimize(self, optimizer='sgd', options={'optimizer': 'Adam', 'learning_rate': 1e-2, 'nepochs':int(3e3), 'nmessage': 1000}):
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
        init_angles = tf.Variable(2 * np.pi * np.random.rand(2*self.nlayers*(2*self.nqubits-2) + 2*self.nqubits), dtype=tf.float64)
        # Optimize
        min_energy, best_params, _ = optimize(self._cost_function, init_angles, method=optimizer, options=options)
        
        return min_energy, best_params
    
    def hessian(self, parameters, rows=None):
        """
        Compute the Hessian matrix of the cost function evaluated at parameters.
        
        Args:
            parameters (numpy.1darray): values of the parameters at which the Hessian is evaluated.
            rows (numpy.1darray): rows of the matrix to be computed (this argument is used to parallelized the com
                                  the computation. If None, the full matrix is computed).  
        Returns:
            (numpy.2darray): Hessian matrix.
        """
        # Construct Hessian
        nparams = len(parameters)
        hess_matrix = np.zeros(shape=(nparams, nparams))
        if rows == None:
            rows = range(nparams)
        for i in rows:
            print('i: ', i)
            # Parameter shift
            parameters[i] += np.pi/2
            for j in range(i, nparams):
                # Parameter shift
                parameters[j] += np.pi/2
                # Add contribution
                contribution = self._hess_loss(parameters)
                hess_matrix[i][j] -= contribution
                hess_matrix[j][i] -= contribution
                # Parameter shift
                parameters[j] -= np.pi
                # Add contribution
                contribution = self._hess_loss(parameters)
                hess_matrix[i][j] += contribution
                hess_matrix[j][i] += contribution
                # Revert parameter shift
                parameters[j] += np.pi/2
            # Parameter shift
            parameters[i] -= np.pi
            for j in range(i, nparams):
                # Parameter shift
                parameters[j] += np.pi/2
                # Add contribution
                contribution = self._hess_loss(parameters)
                hess_matrix[i][j] += contribution
                hess_matrix[j][i] += contribution
                # Parameter shift
                parameters[j] -= np.pi
                # Add contribution
                contribution = self._hess_loss(parameters)
                hess_matrix[i][j] -= contribution
                hess_matrix[j][i] -= contribution
                # Revert parameter shift
                parameters[j] += np.pi/2
            # Revert parameter shift
            parameters[i] += np.pi/2
            # Correct the double addition of the diagonal contributions
            hess_matrix[i][i] /= 2
        
        return hess_matrix / 4
    
    def qfim(self, parameters, rows=None):
        """
        Compute the Hessian matrix of the cost function evaluated at parameters.
        
        Args:
            parameters (numpy.1darray): values of the parameters at which the Hessian is evaluated.
            rows (numpy.1darray): rows of the matrix to be computed (this argument is used to parallelized the com
                                  the computation. If None, the full matrix is computed).   
        Returns:
            (numpy.2darray): Hessian matrix.
        """
        # Construct qfim
        nparams = len(parameters)
        qfim = np.zeros(shape=(nparams, nparams))
        self.hea.set_parameters(parameters)
        state_bra = np.conjugate(self.hea().numpy())
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
                state = self.hea().numpy()
                # Add contribution
                contribution = np.abs(np.inner(state_bra, state))**2
                qfim[i][j] += contribution
                qfim[j][i] += contribution
                # Parameter shift
                parameters[j] -= np.pi
                self.hea.set_parameters(parameters)
                state = self.hea().numpy()
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
                state = self.hea().numpy()
                # Add contribution
                contribution = np.abs(np.inner(state_bra, state))**2
                qfim[i][j] -= contribution
                qfim[j][i] -= contribution
                # Parameter shift
                parameters[j] -= np.pi
                self.hea.set_parameters(parameters)
                state = self.hea().numpy()
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