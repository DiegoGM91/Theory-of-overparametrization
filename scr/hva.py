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
from qibo.hamiltonians import TFIM, Hamiltonian
qibo.set_backend("matmuleinsum") # qibo version==0.1.5
qibo.set_device("/CPU:0")
qibo.set_threads(1)


class VHA_Ising:
    """
    Class for the Variational Hamiltonian Ansatz for the Transverse Field Ising Model.
    """

    def __init__(self, nqubits, p=1, g=1, periodic=True):
        """
        Args:
            nqubits (int): number of qubits.
            p (int): depth parameter of the ansatz.
            g (float): strength of the transverse field.
            periodic (bool): whether periodic boundary conditions apply or not.
        """
        self.nqubits = nqubits
        self.periodic = periodic
        self.loss_record = []
        # Hamiltonian
        if periodic:
            self.hamiltonian = TFIM(nqubits, h=g)
        else:
            self.hamiltonian = self.non_periodic_ham(g)
        # Ground state energy
        self.exact_energy = np.real(self.hamiltonian.eigenvalues().numpy()[0])
        # Circuit
        self.vha = Circuit(nqubits)
        self.p = p
        for q in range(nqubits):
            self.vha.add(gates.H(q))
        for l in range(p):
            for q in range(nqubits-1):
                self.vha.add(gates.CNOT(q, q+1))
                self.vha.add(gates.RZ(q+1, theta=0))
                self.vha.add(gates.CNOT(q, q+1))
            if periodic:
                self.vha.add(gates.CNOT(0, nqubits-1))
                self.vha.add(gates.RZ(nqubits-1, theta=0))
                self.vha.add(gates.CNOT(0, nqubits-1))
            for q in range(nqubits):
                self.vha.add(gates.RX(q, theta=0))
                
    def non_periodic_ham(self, h):
        """
        TFIM Hamiltonian with non-periodic boundary conditions.
        
        Args:
            h (float): strength of the transverse field.
            
        Returns:
            (qibo.hamiltonians.Hamiltonian): non-periodic TFIM Hamiltonian.
        """
        import sympy as sy
        from qibo import matrices
        # Symbols
        Z_symbols = sy.symbols(f'Z:{self.nqubits}')
        X_symbols = sy.symbols(f'X:{self.nqubits}')
        # Symbolic hamiltonian
        symbolic_ham = -sum(Z_symbols[i] * Z_symbols[i + 1] for i in range(self.nqubits-1))
        symbolic_ham += -h * sum(X_symbols)
        # Symbol to matrix map
        symbol_map = {s: (i, matrices.Z) for i, s in enumerate(Z_symbols)}
        symbol_map.update({s: (i, matrices.X) for i, s in enumerate(X_symbols)})
   
        return Hamiltonian.from_symbolic(symbolic_ham, symbol_map)

    def _cost_function(self, parameters):
        """
        Cost function for the optimization of the variational circuit. The cost is given by the difference
        between the expected value of the energy and the exact ground-state energy.
        
        Args:
            parameters (array): angles for the gates in the ansatz.
            
        Returns:
            (float): value of the cost function.
        """
        # Correlate parameters
        correlated_parameters =  []
        if self.periodic:
            for i in range(int(parameters.shape[0])):
                correlated_parameters.extend(self.nqubits * [parameters[i]])
        else:
            for i in range(0,int(parameters.shape[0]),2):
                correlated_parameters.extend((self.nqubits-1) * [parameters[i]])
                correlated_parameters.extend(self.nqubits * [parameters[i+1]])  
        # Set parameters and execute the circuit
        self.vha.set_parameters(correlated_parameters)
        trial_state = self.vha()
        # Compute and record loss
        loss = self.hamiltonian.expectation(trial_state) - self.exact_energy
        self.loss_record.append(loss.numpy())

        return loss

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
        #tf.random.set_seed(None)
        init_angles = tf.Variable(np.pi * np.random.rand(2*self.p), dtype=tf.float64)
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
        if self.periodic:
            # Correlate parameters
            correlated_parameters =  []
            for i in range(int(parameters.shape[0])):
                correlated_parameters.extend(self.nqubits * [parameters[i]])
            correlated_parameters = np.array(correlated_parameters)
            # Construct Hessian
            nparams = 2*self.p
            hess_matrix = np.zeros(shape=(nparams, nparams))
            if rows == None:
                rows = range(nparams)
            for index_i in rows:
                for index_j in range(index_i, nparams):
                    for qubit_i in range(self.nqubits):
                        # Parameter shift
                        correlated_parameters[index_i*self.nqubits+qubit_i] += np.pi/2
                        for qubit_j in range(self.nqubits):
                            # Parameter shift
                            correlated_parameters[index_j*self.nqubits+qubit_j] += np.pi/2
                            self.vha.set_parameters(correlated_parameters)
                            state = self.vha()
                            # Add contribution
                            contribution = self.hamiltonian.expectation(state)
                            hess_matrix[index_i][index_j] += contribution
                            hess_matrix[index_j][index_i] += contribution
                            # Parameter shift
                            correlated_parameters[index_j*self.nqubits+qubit_j] -= np.pi
                            self.vha.set_parameters(correlated_parameters)
                            state = self.vha()
                            # Add contribution
                            contribution = self.hamiltonian.expectation(state)
                            hess_matrix[index_i][index_j] -= contribution
                            hess_matrix[index_j][index_i] -= contribution
                            # Revert parameter shift
                            correlated_parameters[index_j*self.nqubits+qubit_j] += np.pi/2
                        # Parameter shift
                        correlated_parameters[index_i*self.nqubits+qubit_i] -= np.pi
                        for qubit_j in range(self.nqubits):
                            # Parameter shift
                            correlated_parameters[index_j*self.nqubits+qubit_j] += np.pi/2
                            self.vha.set_parameters(correlated_parameters)
                            state = self.vha()
                            # Add contribution
                            contribution = self.hamiltonian.expectation(state)
                            hess_matrix[index_i][index_j] -= contribution
                            hess_matrix[index_j][index_i] -= contribution
                            # Parameter shift
                            correlated_parameters[index_j*self.nqubits+qubit_j] -= np.pi
                            self.vha.set_parameters(correlated_parameters)
                            state = self.vha()
                            # Add contribution
                            contribution = self.hamiltonian.expectation(state)
                            hess_matrix[index_i][index_j] += contribution
                            hess_matrix[index_j][index_i] += contribution
                            # Revert parameter shift
                            correlated_parameters[index_j*self.nqubits+qubit_j] += np.pi/2
                        # Revert parameter shift
                        correlated_parameters[index_i*self.nqubits+qubit_i] += np.pi/2
                # Correct the double addition of the diagonal contributions
                hess_matrix[index_i][index_i] /= 2
        
        else:           
            # Correlate parameters
            correlated_parameters =  []
            for i in range(0,int(parameters.shape[0]),2):
                correlated_parameters.extend((self.nqubits-1) * [parameters[i]])
                correlated_parameters.extend(self.nqubits * [parameters[i+1]]) 
            correlated_parameters = np.array(correlated_parameters)
            # Construct Hessian
            nparams = 2*self.p
            hess_matrix = np.zeros(shape=(nparams, nparams))  
            if rows == None:
                rows = range(nparams)
            for index_i in rows:
                if index_i % 2 == 0:
                    angles_i = self.nqubits -1
                    angles_k = self.nqubits
                    index_k = index_i -1
                else:
                    angles_i = self.nqubits
                    angles_k = self.nqubits-1
                    index_k = index_i
                for index_j in range(index_i, nparams):
                    if index_j % 2 == 0:
                        angles_j = self.nqubits -1
                        angles_m = self.nqubits
                        index_m = index_j-1
                    else:
                        angles_j = self.nqubits
                        angles_m = self.nqubits-1
                        index_m = index_j
                    for qubit_i in range(angles_i):                        
                        # Parameter shift
                        correlated_parameters[(index_i//2)*angles_i+(index_k//2+1)*angles_k+qubit_i] += np.pi/2
                        for qubit_j in range(angles_j):
                            # Parameter shift
                            correlated_parameters[(index_j//2)*angles_j+(index_m//2+1)*angles_m+qubit_j] += np.pi/2
                            self.vha.set_parameters(correlated_parameters)
                            state = self.vha()
                            # Add contribution
                            contribution = self.hamiltonian.expectation(state)
                            hess_matrix[index_i][index_j] += contribution
                            hess_matrix[index_j][index_i] += contribution
                            # Parameter shift
                            correlated_parameters[(index_j//2)*angles_j+(index_m//2+1)*angles_m+qubit_j] -= np.pi
                            self.vha.set_parameters(correlated_parameters)
                            state = self.vha()
                            # Add contribution
                            contribution = self.hamiltonian.expectation(state)
                            hess_matrix[index_i][index_j] -= contribution
                            hess_matrix[index_j][index_i] -= contribution
                            # Revert parameter shift
                            correlated_parameters[(index_j//2)*angles_j+(index_m//2+1)*angles_m+qubit_j] += np.pi/2
                        # Parameter shift
                        correlated_parameters[(index_i//2)*angles_i+(index_k//2+1)*angles_k+qubit_i] -= np.pi
                        for qubit_j in range(angles_j):
                            # Parameter shift
                            correlated_parameters[(index_j//2)*angles_j+(index_m//2+1)*angles_m+qubit_j] += np.pi/2
                            self.vha.set_parameters(correlated_parameters)
                            state = self.vha()
                            # Add contribution
                            contribution = self.hamiltonian.expectation(state)
                            hess_matrix[index_i][index_j] -= contribution
                            hess_matrix[index_j][index_i] -= contribution
                            # Parameter shift
                            correlated_parameters[(index_j//2)*angles_j+(index_m//2+1)*angles_m+qubit_j] -= np.pi
                            self.vha.set_parameters(correlated_parameters)
                            state = self.vha()
                            # Add contribution
                            contribution = self.hamiltonian.expectation(state)
                            hess_matrix[index_i][index_j] += contribution
                            hess_matrix[index_j][index_i] += contribution
                            # Parameter shift
                            correlated_parameters[(index_j//2)*angles_j+(index_m//2+1)*angles_m+qubit_j] += np.pi/2
                        # Reverse parameter shift
                        correlated_parameters[(index_i//2)*angles_i+(index_k//2+1)*angles_k+qubit_i] += np.pi/2
                # Correct the double addition of the diagonal contributions
                hess_matrix[index_i][index_i] /= 2
                
        return hess_matrix / 4
    
    
    def qfim(self, parameters, rows=None):
        """
        Compute the Quantum Fisher Information matrix of the ansatz evaluated at parameters.
        
        Args:
            parameters (numpy.1darray): values of the parameters at which the QFI matrix is evaluated.
            rows (numpy.1darray): rows of the matrix to be computed (this argument is used to parallelized the com
                                  the computation. If None, the full matrix is computed).
        Returns:
            (numpy.2darray): QFI matrix.
        """
        if self.periodic:
            # Correlate parameters
            correlated_parameters =  []
            for i in range(int(parameters.shape[0])):
                correlated_parameters.extend(self.nqubits * [parameters[i]])
            correlated_parameters = np.array(correlated_parameters)
            self.vha.set_parameters(correlated_parameters)
            state_bra = np.conjugate(self.vha().numpy())
            # Construct QFIM
            nparams = 2*self.p
            qfim = np.zeros(shape=(nparams, nparams))
            if rows == None:
                rows = range(nparams)
            for index_i in rows:
                for index_j in range(index_i, nparams):
                    for qubit_i in range(self.nqubits):
                        # Parameter shift
                        correlated_parameters[index_i*self.nqubits+qubit_i] += np.pi/2
                        for qubit_j in range(self.nqubits):
                            # Parameter shift
                            correlated_parameters[index_j*self.nqubits+qubit_j] += np.pi/2
                            self.vha.set_parameters(correlated_parameters)
                            state = self.vha().numpy()
                            # Add contribution
                            contribution = np.abs(np.inner(state_bra, state))**2
                            qfim[index_i][index_j] += contribution
                            qfim[index_j][index_i] += contribution
                            # Parameter shift
                            correlated_parameters[index_j*self.nqubits+qubit_j] -= np.pi
                            self.vha.set_parameters(correlated_parameters)
                            state = self.vha().numpy()
                            # Add contribution
                            contribution = np.abs(np.inner(state_bra, state))**2
                            qfim[index_i][index_j] -= contribution
                            qfim[index_j][index_i] -= contribution
                            # Revert parameter shift
                            correlated_parameters[index_j*self.nqubits+qubit_j] += np.pi/2
                        # Parameter shift
                        correlated_parameters[index_i*self.nqubits+qubit_i] -= np.pi
                        for qubit_j in range(self.nqubits):
                            # Parameter shift
                            correlated_parameters[index_j*self.nqubits+qubit_j] += np.pi/2
                            self.vha.set_parameters(correlated_parameters)
                            state = self.vha().numpy()
                            # Add contribution
                            contribution = np.abs(np.inner(state_bra, state))**2
                            qfim[index_i][index_j] -= contribution
                            qfim[index_j][index_i] -= contribution
                            # Parameter shift
                            correlated_parameters[index_j*self.nqubits+qubit_j] -= np.pi
                            self.vha.set_parameters(correlated_parameters)
                            state = self.vha().numpy()
                            # Add contribution
                            contribution = np.abs(np.inner(state_bra, state))**2
                            qfim[index_i][index_j] += contribution
                            qfim[index_j][index_i] += contribution
                            # Revert parameter shift
                            correlated_parameters[index_j*self.nqubits+qubit_j] += np.pi/2
                        # Revert parameter shift
                        correlated_parameters[index_i*self.nqubits+qubit_i] += np.pi/2
                # Correct the double addition of the diagonal contributions
                qfim[index_i][index_i] /= 2
        else:
            # Correlate parameters
            correlated_parameters =  []
            for i in range(0,int(parameters.shape[0]),2):
                correlated_parameters.extend((self.nqubits-1) * [parameters[i]])
                correlated_parameters.extend(self.nqubits * [parameters[i+1]]) 
            correlated_parameters = np.array(correlated_parameters)
            self.vha.set_parameters(correlated_parameters)
            state_bra = np.conjugate(self.vha().numpy())
            # Construct QFIM
            nparams = 2*self.p
            qfim = np.zeros(shape=(nparams, nparams))
            if rows == None:
                rows = range(nparams)
            for index_i in rows:
                if index_i % 2 == 0:
                    angles_i = self.nqubits -1
                    angles_k = self.nqubits
                    index_k = index_i -1
                else:
                    angles_i = self.nqubits
                    angles_k = self.nqubits-1
                    index_k = index_i
                for index_j in range(index_i, nparams):
                    if index_j % 2 == 0:
                        angles_j = self.nqubits -1
                        angles_m = self.nqubits
                        index_m = index_j-1
                    else:
                        angles_j = self.nqubits
                        angles_m = self.nqubits-1
                        index_m = index_j
                    for qubit_i in range(self.nqubits):
                        # Parameter shift
                        correlated_parameters[(index_i//2)*angles_i+(index_k//2+1)*angles_k+qubit_i] += np.pi/2
                        for qubit_j in range(self.nqubits):
                            # Parameter shift
                            correlated_parameters[(index_j//2)*angles_j+(index_m//2+1)*angles_m+qubit_j] += np.pi/2
                            self.vha.set_parameters(correlated_parameters)
                            state = self.vha().numpy()
                            # Add contribution
                            contribution = np.abs(np.inner(state_bra, state))**2
                            qfim[index_i][index_j] += contribution
                            qfim[index_j][index_i] += contribution
                            # Parameter shift
                            correlated_parameters[(index_j//2)*angles_j+(index_m//2+1)*angles_m+qubit_j] -= np.pi
                            self.vha.set_parameters(correlated_parameters)
                            state = self.vha().numpy()
                            # Add contribution
                            contribution = np.abs(np.inner(state_bra, state))**2
                            qfim[index_i][index_j] -= contribution
                            qfim[index_j][index_i] -= contribution
                            # Revert parameter shift
                            correlated_parameters[(index_j//2)*angles_j+(index_m//2+1)*angles_m+qubit_j] += np.pi/2
                        # Parameter shift
                        correlated_parameters[(index_i//2)*angles_i+(index_k//2+1)*angles_k+qubit_i] -= np.pi
                        for qubit_j in range(self.nqubits):
                            # Parameter shift
                            correlated_parameters[(index_j//2)*angles_j+(index_m//2+1)*angles_m+qubit_j] += np.pi/2
                            self.vha.set_parameters(correlated_parameters)
                            state = self.vha().numpy()
                            # Add contribution
                            contribution = np.abs(np.inner(state_bra, state))**2
                            qfim[index_i][index_j] -= contribution
                            qfim[index_j][index_i] -= contribution
                            # Parameter shift
                            correlated_parameters[(index_j//2)*angles_j+(index_m//2+1)*angles_m+qubit_j] -= np.pi
                            self.vha.set_parameters(correlated_parameters)
                            state = self.vha().numpy()
                            # Add contribution
                            contribution = np.abs(np.inner(state_bra, state))**2
                            qfim[index_i][index_j] += contribution
                            qfim[index_j][index_i] += contribution
                            # Revert parameter shift
                            correlated_parameters[(index_j//2)*angles_j+(index_m//2+1)*angles_m+qubit_j] += np.pi/2
                        # Revert parameter shift
                        correlated_parameters[(index_i//2)*angles_i+(index_k//2+1)*angles_k+qubit_i] += np.pi/2
                # Correct the double addition of the diagonal contributions
                qfim[index_i][index_i] /= 2
            
        return -qfim / 8