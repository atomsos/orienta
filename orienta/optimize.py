"""Structure optimization. """

from os.path import isfile
import sys
import pickle
import time
from math import sqrt

import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase.parallel import world, barrier
from ase.io.trajectory import Trajectory
from ase.optimize.optimize import Dynamics as ase_Dynamics
Dynamics = ase_Dynamics


# class Dynamics(ase_Dynamics):
#     """
#     Modified Dynsmiacs
#     """
#     def call_observers(self):
#         recover_cell = False
#         if not self.atoms.pbc.any():
#             if self.atoms.cell:
#                 cell = self.atoms.cell.copy()
#             else:
#                 cell = None
#             self.atoms.cell = np.zeros((3, 3))
#             recover_cell = True
#         for function, interval, args, kwargs in self.observers:
#             call = False
#             # Call every interval iterations
#             if interval > 0:
#                 if (self.nsteps % interval) == 0:
#                     call = True
#             # Call only on iteration interval
#             elif interval <= 0:
#                 if self.nsteps == abs(interval):
#                     call = True
#             if call:
#                 function(*args, **kwargs)
#         if recover_cell:
#             self.atoms.cell = cell


class Optimizer(Dynamics):
    """Base-class for all structure optimization classes."""

    def __init__(
        self,
        atoms,
        restart,
        logfile,
        trajectory,
        master=None,
        append_trajectory=False,
        force_consistent=False,
    ):
        """Structure optimizer object.

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: str
            Filename for restart file.  Default value is *None*.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: Trajectory object or str
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K).  If force_consistent=None, uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.
        """
        Dynamics.__init__(
            self,
            atoms,
            logfile,
            trajectory,
            append_trajectory=append_trajectory,
            master=master,
        )

        self.force_consistent = force_consistent
        if self.force_consistent is None:
            self.set_force_consistent()

        self.restart = restart

        # initialize attribute
        self.fmax = None

        if restart is None or not isfile(restart):
            self.initialize()
        else:
            self.read()
            barrier()

    def todict(self):
        description = {
            "type": "optimization",
            "optimizer": self.__class__.__name__,
        }
        return description

    def initialize(self):
        pass

    def irun(self, fmax=0.05, steps=None):
        """ call Dynamics.irun and keep track of fmax"""
        self.fmax = fmax
        if steps:
            self.max_steps = steps
        return Dynamics.irun(self)

    def run(self, fmax=0.05, steps=None):
        """ call Dynamics.run and keep track of fmax"""
        self.fmax = fmax
        if steps:
            self.max_steps = steps
        return Dynamics.run(self)

    def converged(self, forces=None):
        """Did the optimization converge?"""
        if forces is None:
            forces = self.atoms.get_forces()
        if hasattr(self.atoms, "get_curvature"):
            return ((forces ** 2).sum(
                axis=1
            ).max() < self.fmax ** 2 and self.atoms.get_curvature() < 0.0)
        return (forces ** 2).sum(axis=1).max() < self.fmax ** 2

    def log(self, forces=None, e=None):
        if forces is None:
            forces = self.atoms.get_forces()
        fmax = sqrt((forces**2).sum(axis=1).max())
        if e is None:
            e = self.atoms.get_potential_energy(
                force_consistent=self.force_consistent
            )
        T = time.localtime()
        # import pdb; pdb.set_trace()
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                self.logfile.write(
                    '%s  %4s %8s %15s %12s\n' %
                    (' ' * len(name), 'Step', 'Time', 'Energy', 'fmax'))
                if self.force_consistent:
                    self.logfile.write(
                        '*Force-consistent energies used in optimization.\n')
            self.logfile.write('%s:  %3d %02d:%02d:%02d %15.6f%1s %12.4f\n' %
                               (name, self.nsteps, T[3], T[4], T[5], e,
                                {1: '*', 0: ''}[self.force_consistent], fmax))
            self.logfile.flush()

    def dump(self, data):
        if world.rank == 0 and self.restart is not None:
            pickle.dump(data, open(self.restart, 'wb'), protocol=2)

    def load(self):
        return pickle.load(open(self.restart, 'rb'))

    def set_force_consistent(self):
        """Automatically sets force_consistent to True if force_consistent
        energies are supported by calculator; else False."""
        try:
            self.atoms.get_potential_energy(force_consistent=True)
        except PropertyNotImplementedError:
            self.force_consistent = False
        else:
            self.force_consistent = True
