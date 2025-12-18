# -*- coding: utf-8 -*-
import warnings
import modlog

import numpy as np
from numpy.linalg import eigh, norm

from optimize import Optimizer
import pdb


logger = modlog.getLogger(__name__, "normal", "GASE_BFGS_LOGLEVEL")
np.set_printoptions(precision=4, suppress=True)  # 设置浮点精度


class CriticalPointException(Exception):
    pass


def normalized(v):
    n = np.linalg.norm(v)
    if n < 1e-5:
        return v
    return v/n


def get_real_cell(cell):
    from ase.cell import Cell
    if cell is None:
        return None
    if isinstance(cell, list):
        cell = np.array(cell)
    elif isinstance(cell, Cell):
        cell = cell.array
    assert isinstance(cell, np.ndarray)
    if cell.shape == (3, 3):
        if (norm(cell, axis=0) <= 0.1).any():
            return None
    if cell.shape == (3,):
        if (cell <= 0.1).any():
            return None
    return cell


def get_atoms_vector(atoms0, atoms1, cell=None):
    """
    return pos0 - pos1
    and minimize abs(dpos) when cell exists
    """
    if isinstance(atoms0, np.ndarray):
        pos0 = atoms0
        pos1 = atoms1
    else:
        pos0 = atoms0.get_positions()
        pos1 = atoms1.get_positions()
        cell = atoms0.get_cell()
    cell = get_real_cell(cell)
    dpos = pos0 - pos1
    if cell is None:
        return dpos, pos0
    res = []
    displace = []
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                _disp = x*cell[0] + y*cell[1] + z*cell[2]
                res.append(dpos + _disp)
                displace.append(_disp)
    res = np.array(res)
    idx = np.argmin(norm(res, axis=2), axis=0)
    vector = res[idx, np.arange(len(idx))]
    displace = np.array(displace)[idx]
    pos0 += displace
    return vector, pos0


class BFGS_SOPT(Optimizer):
    def __init__(self, atoms, anchor, critical_dri=0.04, restart=None, logfile='-', trajectory=None,
                 maxstep=0.10, master=None, debug=False):
        """Modified BFGS optimizer for Spherical Optimization.

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        anchor: Atoms object or positions
            The Atoms pairing with atoms as an anchor

        critical_dri: float
            Critical number of dr_i, if reached, the process will be
            initialized

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.04 Å).

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        debug: boolean
            debug option
        """
        if maxstep > 1.0:
            warnings.warn('You are using a much too large value for '
                          'the maximum step size: %.1f Å' % maxstep)
        assert anchor is not None, \
            'Partner is needed when Using Spherical Optimization'
        self.maxstep = maxstep
        self.anchor = anchor
        self.critical_dri = critical_dri
        self.debug = debug
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)

    def todict(self):
        d = Optimizer.todict(self)
        if hasattr(self, 'maxstep'):
            d.update(maxstep=self.maxstep)
        return d

    def initialize(self):
        self.H = None
        self.r0 = None
        self.f0 = None
        if hasattr(self.anchor, 'get_positions'):
            self.r_anchor = self.anchor.get_positions()
        else:
            assert np.array(self.anchor).shape == (len(self.atoms), 3)
            self.r_anchor = np.array(self.anchor)
        atoms = self.atoms
        cell = atoms.get_cell()
        vector, new_r = get_atoms_vector(
            atoms.get_positions(), self.r_anchor, cell=cell)
        self.R_square = float(np.square(vector).sum())
        self.index = np.argmax(np.fabs(vector).ravel())
        self._dpos = np.array([[0]])

    def read(self):
        self.H, self.r0, self.f0, self.maxstep = self.load()

    def step(self, f=None):
        index = self.index
        atoms = self.atoms
        cell = get_real_cell(atoms.get_cell())
        if f is None:
            f = atoms.get_forces()

        # import pdb; pdb.set_trace()
        f_real = f.copy()
        f_ravel = f.ravel()
        r_real = atoms.get_positions()
        r_anchor = self.r_anchor
        _, r = get_atoms_vector(r_real, r_anchor, cell=cell)
        r = r.ravel()

        # get new force by projecting radial part
        r_anchor = self.r_anchor.ravel()
        r_v = normalized(r - r_anchor)
        force_mod = (f_ravel - np.dot(f_ravel, r_v)
                     * r_v).reshape((-1, 3))
        # determine index of the removed one
        if self.debug:
            import pdb
            pdb.set_trace()

        def rm_index(r, index):
            assert index < len(r)
            return np.delete(r, index)

        def add_index(r, index=0, val=0):
            return np.insert(r, index, val)

        def get_real_positions(r_mod, r_anchor, R_square, index, sign=1, cell=None):
            """
            r_mod: 3n-1 vector, modified positions
            r_anchor: 3n vector, positions of anchor

            return: shape (-1, 3)
            """
            from ase.geometry import wrap_positions
            r_anchor_mod = rm_index(r_anchor, index)
            dr_mod = r_mod - r_anchor_mod
            if R_square - np.square(dr_mod).sum() < 0:
                raise ValueError('dr too large')
            val = np.sqrt(R_square - np.square(dr_mod).sum()) * \
                sign + r_anchor[index]
            real_r = add_index(r_mod, index, val).reshape((-1, 3))
            if cell is not None:
                real_r = wrap_positions(real_r, cell)
            return real_r

        def gradF(r, r_anchor, R_square, index, sign=1):
            dri = r[index] - r_anchor[index]
            dr_max = np.max(abs(r - r_anchor))
            dr_all = rm_index(r-r_anchor, index)
            if abs(dri) < self.critical_dri and abs(dri) < dr_max:
                raise CriticalPointException('dri too small')
            gradf = -1 * dr_all / dri
            return gradf

        r_mod = rm_index(r, index)
        sign = 1
        real_positions = get_real_positions(
            r_mod, r_anchor, self.R_square, index, sign=sign, cell=cell)
        if abs(real_positions.ravel()[index] - r[index]) > 1e-5:
            sign = -1

        gradf = gradF(r, r_anchor, self.R_square, index, sign)
        f = rm_index(f_ravel, index) + f_ravel[index] * gradf
        r0 = self.r0
        if r0 is not None:
            r0 = rm_index(self.r0, index)
        self.update(r_mod, f, r0, self.f0)
        omega, V = eigh(self.H)
        dr = np.dot(V, np.dot(f, V) / np.fabs(omega))
        tmpdr = add_index(dr, index, 0).reshape((-1, 3))
        steplengths = np.linalg.norm(tmpdr, axis=1)
        dr = self.determine_step(dr, steplengths)

        # dr_value = np.dot(gradf, dr)
        # dr = add_index(dr, index, dr_value)
        # index_pos = r[index] + dr_value
        # while abs(index_pos - get_real_positions(r+dr, r_anchor, self.R_square, index, sign)) > 1e-5:
        #     dr_value = get_real_positions(r+dr, r_anchor, self.R_square, index, sign) - r[index]
        #     dr[index] = dr_value
        #     # steplengths = (dr.reshape((-1,3))**2).sum(1)**0.5
        #     steplengths = np.linalg.norm(dr.reshape((-1,3), axis=0)
        #     dr = self.determine_step(dr.reshape((-1,3)), steplengths).ravel()
        #     newr = r + dr
        #     index_pos = newr[index]
        # xdr =
        # new_r = get_real_positions(r_mod+dr, r_anchor, self.R_square, index,
        #                            sign=sign, cell=cell)
        # vector, new_r = get_atoms_vector(new_r, r_real, cell=cell)
        # steplengths = np.linalg.norm(vector.reshape((-1,3)), axis=1)
        # dr = self.determine_step(dr, steplengths)
        unavailable = True
        while unavailable:
            try:
                new_r = get_real_positions(
                    r_mod+dr, r_anchor, self.R_square, index,
                    sign=sign, cell=cell)
                vector, new_r = get_atoms_vector(new_r, r_real, cell=cell)
                steplengths = np.linalg.norm(vector.reshape((-1, 3)), axis=1)
                dr = self.determine_step(dr, steplengths)
                unavailable = False
            except ValueError:
                dr *= 0.80
        newpos = new_r.reshape((-1, 3))
        dpos = newpos - r_real
        self._dpos = dpos
        atoms.set_positions(newpos)
        vector, new_r = get_atoms_vector(
            atoms.get_positions(), self.r_anchor, cell=cell)
        newR_square = np.square(vector).sum()
        if abs(newR_square - self.R_square) > 0.1:
            raise ValueError()
        self.r0 = r.copy()
        self.f0 = f.copy()
        self.dump((self.H, self.r0, self.f0, self.maxstep))
        # force_mod = add_index(f, index, 0).reshape((-1,3))
        # print('modf:\n', force_mod)
        return force_mod

    def determine_step(self, dr, steplengths):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        maxsteplength = np.max(steplengths)
        if maxsteplength >= self.maxstep:
            dr *= self.maxstep / maxsteplength

        return dr

    def update(self, r, f, r0, f0):
        """
        There are two standard choices for the initial approximation of B in BFGS:
        Either you choose B=∥g0∥δI where
            ∥g0∥ is the gradient in the very first iteration and
            δ a "typical step size" from xk to xk+1
        Or you choose B=(y1T·y1/y1T·s1) I
            using the standard notation used for BFGS.
        """
        if self.H is None:  # first step
            alpha_default = 70.0  # default set to 70.p
            # option1:
            g0, delta = norm(f), 0.10
            alpha1 = g0 / delta
            # option2:
            upper = np.dot(f, f)
            lower = np.dot(f, r)
            alpha2 = abs(upper / lower)
            # option3:
            alpha3 = (alpha1 + alpha2) / 2.0
            alpha_min = 30.0

            res = []
            for a in [alpha1, alpha2, alpha3]:
                if a > alpha_min:
                    res.append(a)
            if len(res) == 0:
                res.append(alpha_min)
            else:
                res.append(alpha_default)
            alpha = min(res)
            self.H = np.eye(len(r)) * alpha
            self._alphas = {
                'alpha1': alpha1,
                'alpha2': alpha2,
                'alpha3': alpha3,
                'alpha': alpha
            }
            # print("Alpha: ", alpha1, alpha2, alpha3, 'alpha:', alpha)
            return
        dr = r - r0

        if np.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        df = f - f0
        a = np.dot(dr, df)
        dg = np.dot(self.H, dr)
        b = np.dot(dr, dg)
        self.H -= np.outer(df, df) / a + np.outer(dg, dg) / b

    def replay_trajectory(self, traj):
        """Initialize hessian from old trajectory."""
        if isinstance(traj, str):
            from ase.io.trajectory import Trajectory
            traj = Trajectory(traj, 'r')
        self.H = None
        atoms = traj[0]
        r0 = atoms.get_positions().ravel()
        f0 = atoms.get_forces().ravel()
        for atoms in traj:
            r = atoms.get_positions().ravel()
            f = atoms.get_forces().ravel()
            self.update(r, f, r0, f0)
            r0 = r
            f0 = f

        self.r0 = r0
        self.f0 = f0

    def irun(self, fmax=0.05, steps=100000000):
        """Run structure optimization algorithm as generator. This allows, e.g.,
        to easily run two optimizers at the same time.

        Examples:
        >>> opt1 = BFGS(atoms)
        >>> opt2 = BFGS(StrainFilter(atoms)).irun()
        >>> for _ in opt2:
        >>>     opt1.run()
        """

        if hasattr(self, "force_consistent", ) and self.force_consistent is None:
            self.set_force_consistent()
        self.fmax = fmax
        step = 0
        CriticalPointExceptionCounter = 0
        while step < steps:
            origin_f = self.atoms.get_forces()
            origin_e = self.atoms.get_potential_energy()
            if self.converged(origin_f):
                yield True
                return
            try:
                f = self.step(origin_f)
                # if self.converged(f):
                #     yield True
                #     return
                CriticalPointExceptionCounter = 0
            except CriticalPointException:
                CriticalPointExceptionCounter += 1
                if CriticalPointExceptionCounter >= 3:
                    print("CriticalPointExceptionCounter too much", exit)
                    yield False
                    return
                logger.debug('Critical Point reached, initialize, step {0}, index: {1}'.format(
                    step, self.index))
                self.initialize()
                logger.debug(
                    'initialized, will use index {0}'.format(self.index))
                continue
            self.log(f, origin_e)
            if self.converged(f):
                yield True
                return
            self.call_observers()
            self.nsteps += 1
            step += 1
            yield False
        yield False

    def run(self, fmax=0.05, steps=100000000):
        """Run structure optimization algorithm.

        This method will return when the forces on all individual
        atoms are less than *fmax* or when the number of steps exceeds
        *steps*.
        FloK: Move functionality into self.irun to be able to run as
              generator."""

        for converged in self.irun(fmax, steps):
            pass
        name = self.__class__.__name__
        if converged:
            self.logfile.write(
                name+": Converged with {0} steps.\n".format(self.nsteps))
        else:
            self.logfile.write(
                name+": Not Converged with {0} steps.\n".format(self.nsteps))
        return converged

    def log(self, forces=None, e=None):
        import time
        from math import sqrt
        if forces is None:
            forces = self.atoms.get_forces()
        fmax = sqrt((forces**2).sum(axis=1).max())
        rmax = np.max(norm(self._dpos, axis=1))
        if e is None:
            e = self.atoms.get_potential_energy(
                force_consistent=self.force_consistent
            )
        T = time.localtime()
        # import pdb; pdb.set_trace()
        if self.logfile is not None:
            name = self.__class__.__name__
            if self.nsteps == 0:
                _head = 'Starting BFGS-SOPT at %02d:%02d:%02d\n' % (
                    T[3], T[4], T[5])
                _head += 'R_square: %.2f; Alphas: %d, %d, %d, Alpha: %d\n' % (
                    self.R_square,
                    self._alphas['alpha1'], self._alphas['alpha2'],
                    self._alphas['alpha3'], self._alphas['alpha'])
                _head += '%s  %4s %8s %15s %8s %8s\n' % (
                    ' ' * len(name), 'Step', 'Time', 'Energy', 'fmax', 'rmax')
                self.logfile.write(_head)
                if self.force_consistent:
                    self.logfile.write(
                        '*Force-consistent energies used in optimization.\n')
            self.logfile.write('%s:  %3d %02d:%02d:%02d %15.6f%1s %8.4f %8.4f\n' %
                               (name, self.nsteps, T[3], T[4], T[5], e,
                                {1: '*', 0: ''}[self.force_consistent], fmax, rmax))
            self.logfile.flush()


class BFGS_SOPT_Double(BFGS_SOPT):

    def __init__(self, atoms, anchor1, anchor2, critical_dri=0.04, restart=None, logfile='-', trajectory=None,
                 maxstep=0.10, master=None, debug=False):
        self.atoms = atoms
        self.anchor1 = anchor1
        self.anchor2 = anchor2
        self.critical_dri = critical_dri
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)

    def initialize(self):
        self.H0 = None
        self.r0 = None
        self.f0 = None
        self.r_anchor1 = self.anchor1.get_positions().ravel()
        self.r_anchor2 = self.anchor2.get_positions().ravel()
        r0 = self.atoms.get_positions().ravel()
        self.R1_square = np.square(r0 - self.r_anchor1).sum()
        self.R2_square = np.square(r0 - self.r_anchor2).sum()
        cell = self.atoms.get_cell()
        vector, new_r = get_atoms_vector(
            self.atoms.get_positions(), self.r_anchor, cell=cell)
        ind = np.argpartition(np.fabs(vector), -2)[-2:]
        self.indexes = ind
        self._dpos = np.array([[0]])

    def step(self, f):
        def rm_index(r, indexes):
            indexes = sorted(indexes.copy(), reverse=True)
            for i in indexes:
                r = np.delete(r, i)
            return r
        def add_index(r, indexes, vals):
            indexes = sorted(indexes.copy(), reverse=True)
            for i in indexes:
                r = np.insert(r, i, vals[i])

        f = f.ravel()
        r = self.atoms.get_positions().ravel()
        indexes = self.indexes
        newr = rm_index(r, indexes)
        newf = rm_index(f, indexes)
        for j in indexes:
            newf -= f[j] * gradF(j)

        r0 = self.r0
        if r0 is not None:
            r0 = rm_index(self.r0, self.indexes)
        self.update(newr, newf, r0, self.f0)
        omega, V = eigh(self.H)
        dr = np.dot(V, np.dot(f, V) / np.fabs(omega))
        tmpdr = add_index(dr, self.indexes, 0).reshape((-1, 3))
        steplengths = np.linalg.norm(tmpdr, axis=1)
        dr = self.determine_step(dr, steplengths)


        self.r0 = r.copy()
        self.f0 = f.copy()
        vector, new_r = get_atoms_vector(
            atoms.get_positions(), self.r_anchor, cell=cell)
        newR_square = np.square(vector).sum()
        if abs(newR_square - self.R_square) > 0.1:
            raise ValueError()

    def update(self, r, f, r0, f0):
        super(BFGS_SOPT_Double, self).update(r, f, r0, f0)

    def determine_step(self, dr, steplengths):
        super(BFGS_SOPT_Double, self).determine_step(dr, steplengths)
