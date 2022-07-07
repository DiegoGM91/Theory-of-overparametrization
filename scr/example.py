#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from hva import HVA_Ising
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", help="Number of qubits", type=int)
parser.add_argument("--nlayers", help="Number of layers of the variational circuit", type=int)
parser.add_argument("--nsteps", default=int(3e3), help="Maximum number of optimization steps", type=int)
parser.add_argument("--g", default=1, help="Strength of the transverse field", type=float)


def main(nqubits, nlayers, nsteps, g):

    # We initialize the HVA
    hva = HVA_Ising(nqubits, nlayers, g)
    
    # We train the HVA
    print('Training HVA...')
    cost_function, optimal_angles = hva.minimize(options={'optimizer': 'Adam', 'learning_rate': 1e-2, 'nepochs':nsteps, 'nmessage': 100})


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)