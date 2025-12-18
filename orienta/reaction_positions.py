"""
repulsion on a unit sphere
"""


import time
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

import modlog
import numpy as np
from numpy.linalg import norm
# import gase.generator.utils

import pdb

# from gase.generator.vsepr import transform_to_spherical_pos

logger = modlog.getLogger(__name__)


def uniformly_sample_on_sphere(origin_point, total_sampling: int = 100,
                               radius: float = 1.0):
    """
    Generate candidate positions uniformly on the spherical surface
        reference: https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
    Input:
        origin_point: xyz coordinates of the center of the sphere will be sampled
        total_sampling: number of the candidates required
        radius: radius of the sphere
    Output:
        sampling results: np.ndarray

    """
    res = []
    while total_sampling > 0:
        total_sampling -= 1
        p = np.random.normal(size=(3,))
        p /= norm(p)
        res.append(p * radius + origin_point)
    return np.array(res)


def run_REP_spherical_opt(
        candidates, atoms, anchor, pbc, cell, 
        repulsion_order, exist_points, 
        max_workers=1, verbose=False):
    all_results = []
    for point in candidates:
        points = exist_points.tolist() + [point.tolist()]
        kwds = {
            'points': points,
            'atoms': atoms,
            'anchor': anchor,
            'pbc': pbc,
            'cell': cell,
            'repulsion_order': repulsion_order,
            'fmax': 1E-5,
            'maxstep': 0.6,
            'verbose': verbose,
        }
        res = REP_spherical_opt(**kwds)
        all_results.append(res)
    return all_results


def run_REP_spherical_opt_multiprocessing(
        candidates, atoms, anchor, pbc, cell, 
        repulsion_order, exist_points, 
        max_workers=1, verbose=False):
    task_handles = []
    all_results = []
    import multiprocessing
    with multiprocessing.Pool(max_workers) as pool:
        for point in candidates:
            points = exist_points.tolist() + [point.tolist()]
            kwds = {
                'points': points,
                'atoms': atoms,
                'anchor': anchor,
                'pbc': pbc,
                'cell': cell,
                'repulsion_order': repulsion_order,
                'fmax': 1E-5,
                'maxstep': 0.6,
                'verbose': verbose,
            }
            task_handles.append(pool.apply_async(REP_spherical_opt, kwds=kwds))
        for handle in task_handles:
            all_results.append(handle.get())
    return all_results


def get_low_repulsion_points(atoms, atom_indices, exist_points=None,
                             total_sampling: int = 100, bond_length: float = 1.0,
                             repulsion_order: float = 5.0, position_resolution: float = 0.5,
                             max_workers: int = 4, normal_ratio: float = 0.15,
                             min_need=None, verbose: bool = False, **kwargs):
    """
    Get all low repulsion points for an atom in atoms given conditions.
    SPECIAL TREATMENT:
        single atom, on the the sampling result
        C_inf axis, contract to one if they are in the eqator
    Input:
        atoms: Atoms like object(AtomsShell)
        atom_indices:
            * int, index of the atom required to get low repulsion points
            * int list, index of a list of atom, the averaged position will be the center
        exist_points: list of the points already exist
        total_sampling: number of the sampling
        bond_length: float, length between low repulsion points
            and the position of the atom_indices
        repulsion_order: float, order of the repulsion expression
        position_resolution: float, points in range of
            position_resolution will be considered as the same position
        max_workers: int, avaiable when use multiprocessing
        normal_ratio: float, lower than this ratio will be regarded as rare and dropped.
        min_need: minimum number of positions needed
    Output:
        (np.ndarray, np.ndarray), (position_res, energy_res),
            list of low repulsion points in coordinates.
    """
    startTime = time.time()
    natoms = len(atoms.numbers)
    if exist_points is None:
        exist_points = np.array([])
    exist_points = np.array(exist_points).reshape((-1, 3))
    npoints = len(exist_points) + 1
    if isinstance(atom_indices, tuple):
        atom_indices = list(atom_indices)
    anchor = np.mean(atoms.positions[atom_indices].reshape((-1, 3)), axis=0)
    # process single atom situation
    candidates = uniformly_sample_on_sphere(anchor, total_sampling=total_sampling,
                                            radius=bond_length)
    if natoms == 1 and npoints == 1:
        position_res, energy_res = candidates[0].reshape((1, 1, 3)), [0.]
        return position_res, energy_res
    task_handles = list()
    position_res = []
    energy_res = []
    energy_with_numbers_res = []
    counter_res = []
    converged_counter = 0
    total_steps = 0
    pbc = atoms.pbc
    cell = atoms.cell
    if max_workers > 1:
        all_results = run_REP_spherical_opt_multiprocessing(
            candidates, atoms, anchor, pbc, cell, repulsion_order, exist_points, max_workers, verbose)
    else:
        all_results = run_REP_spherical_opt(
            candidates, atoms, anchor, pbc, cell, repulsion_order, exist_points, verbose)
    for outds in all_results:
        if not outds['converged']:
            continue
        rspositions = outds['positions']
        energy = outds['energy']
        total_steps += outds['steps']
        converged_counter += 1
        rspositions = rspositions.reshape((-1, 3))
        exist = False
        energy_with_numbers = get_pseudo_atoms(
            rspositions, atoms, anchor,
            pbc, cell, repulsion_order, with_numbers=True).get_potential_energy()
        for i, rspos in enumerate(position_res):
            if norm(rspositions - rspos) < position_resolution:
                exist = True
                counter_res[i] += 1
                break
        if not exist:
            position_res.append(rspositions)
            energy_res.append(energy)
            energy_with_numbers_res.append(energy_with_numbers)
            counter_res.append(1)
    # if True:
    #     atoms.positions = np.vstack((atoms.positions, np.array(position_res).reshape((-1, 3))))
    #     atoms.numbers = atoms.numbers.tolist() + [0] * len(position_res)
    #     atoms.symbols = atoms.symbols + ['X'] * len(position_res)
    #     atoms.write("Q.xyz", format="xyz")
    # pdb.set_trace()
    converged_ratio = converged_counter / total_sampling
    average_steps = total_steps / total_sampling
    logger.info(
        f"converged_ratio: {converged_ratio}, average_steps: {average_steps}")
    if converged_ratio < 0.5:
        logger.warning(f"converged_ratio: {converged_ratio} less than 0.5")
    results = get_nonredundant_reaction_positions(
        atoms, total_sampling,
        position_res, energy_with_numbers_res, counter_res,
        normal_ratio, min_need)
    endTime = time.time()
    logger.debug(f"time: {endTime-startTime}")
    return results


def get_nonredundant_reaction_positions(atoms, total_sampling: int,
                                        position_res, energy_res, counter_res,
                                        normal_ratio: float = 0.10, min_need=None,
                                        merge_by_energy: bool = True, energy_gap=0.2):
    """
    Remove redundant reaction positions
    Input:
        atoms: Atoms like object,
        position_res: results of positions
        energy_res: results of energy
        counter_res: results of counting
        normal_ratio: lower than this ratio will be regarded as rare and dropped.
        min_need: minimum positions needed
    """
    # order res by energy
    energy_order = sorted(range(len(position_res)),
                          key=lambda x: energy_res[x])
    position_res = np.array(position_res)[energy_order]
    energy_res = np.array(energy_res)[energy_order]
    counter_res = np.array(counter_res)[energy_order]
    # drop rare events
    normal_event = np.array(counter_res) / total_sampling >= normal_ratio
    if min_need:
        normal_event[:min_need] = True
    position_res, energy_res, counter_res = position_res[normal_event], \
        energy_res[normal_event], counter_res[normal_event]
    logger.info(
        f'Before energy merge:\n \
          position_res:  {position_res}; \n\
          energy_res:  {energy_res}; \n\
          counter_res: {counter_res}')
    length = len(position_res)
    # if min_need:
    #     length = min_need
    # if min_need != 'all':
    #     if len(energy_res) > 0:
    #         E_min = energy_res[0]
    #         length = 1
    #         while len(energy_res) > length and \
    #                 abs(E_min - energy_res[length]) < 1e-2:
    #             length += 1
    #     if min_need is not None:
    #         length = max(int(min_need), length)
    # if C_inf and all positions are on the equator, set length to 1
    position_res = position_res[:length]
    energy_res = energy_res[:length]
    counter_res = counter_res[:length]

    # pdb.set_trace()
    # print(energy_res)

    if merge_by_energy:
        energy_label = - float('inf')
        keep_label = []
        new_counter_res = []
        for i in range(len(energy_res)):
            e = energy_res[i]
            c = counter_res[i]
            if abs(e - energy_label) < energy_gap:
                keep_label.append(False)
                new_counter_res[-1] += c
            else:
                keep_label.append(True)
                energy_label = e
                new_counter_res.append(c)
        position_res = position_res[keep_label]
        energy_res = energy_res[keep_label]
        counter_res = new_counter_res
        # merge counter
        logger.info(
            f'position_res:  {position_res}; \nenergy_res:  {energy_res}; \ncounter_res: {counter_res}')
    return position_res, energy_res


class PseudoAtoms():
    """docstring for PseudoAtoms"""

    def __init__(self, positions, calc=None, pbc=False, cell=None):
        super(PseudoAtoms, self).__init__()
        self.positions = np.array(positions).reshape((-1, 3))
        self.pbc = pbc
        self.cell = cell
        self.calc = calc
        if calc:
            calc.atoms = self

    def __ase_optimizable__(self):
        return True

    def __len__(self):
        return len(self.positions)

    def has(self, name):
        return False

    @property
    def constraints(self):
        return []

    @constraints.setter
    def constraints(self, c):
        pass

    def iterimages(self):
        positions = np.vstack(
            [self.positions.reshape((-1, 3)), self.calc.repulsion_points])
        res = PseudoAtoms(positions, None, self.pbc, self.cell)

        def get_atomic_numbers():
            return [0] * len(self.positions) + self.calc.repulsion_numbers.tolist()

        res.get_atomic_numbers = get_atomic_numbers
        yield res

    def get_positions(self):
        return self.positions

    def get_atomic_numbers(self):
        return [0] * len(self.positions)

    @property
    def numbers(self):
        return self.get_atomic_numbers()

    def get_cell(self):
        return np.zeros((3, 3))

    def get_calculator(self):
        return None

    @property
    def info(self):
        return {}

    def set_positions(self, positions):
        self.positions = np.array(positions).reshape((-1, 3))

    def get_potential_energy(self, **kwargs):
        return self.calc.get_potential_energy(**kwargs)

    def get_forces(self, **kwargs):
        return self.calc.get_forces()


class PseudoCalc():
    """docstring for PseudoCalc"""

    def __init__(self, atoms: object, kernel_position: np.ndarray,
                 repulsion_points=None,
                 repulsion_numbers=None,
                 repulsion_order: float = 5.,
                 max_repulsion_distance: float = 10.,
                 with_numbers: bool = False):
        super(PseudoCalc, self).__init__()
        self.atoms = atoms
        self.kernel_position = kernel_position
        self.max_repulsion_distance = max_repulsion_distance
        self.with_numbers = with_numbers

        if repulsion_points is None:
            repulsion_points = np.array([])
            repulsion_numbers = np.array([])
        repulsion_points = np.array(repulsion_points.copy()).reshape((-1, 3))
        self.set_repulsion_points(repulsion_points, repulsion_numbers)
        # if len(repulsion_numbers) == len(repulsion_points):
        #     self.repulsion_numbers = repulsion_numbers
        # else:
        #     self.repulsion_numbers = [1.0] * len(repulsion_points)
        # self.repulsion_numbers = np.array(self.repulsion_numbers)
        self.repulsion_order = repulsion_order

    def get_pseudo_masses_matrix(self):
        """
        get masses matrix
        """
        pseudo_masses = self.repulsion_numbers.copy() / 100. + 1.0
        return pseudo_masses

    # def get_potential_energy_v1(self, **kwargs):
    #     import gase.structure.coordinations
    #     numbers = self.atoms.numbers
    #     atoms_positions = self.atoms.positions.copy()
    #     if self.with_numbers:
    #         atoms_positions = atoms_positions * numbers
    #     positions = np.vstack((atoms_positions, self.repulsion_points))
    #     distance_matrix = gase.structure.coordinations.compute_distance_matrix(
    #         positions)
    #     distance_matrix[distance_matrix < 1e-3] = float('inf')
    #     E = 0.5 * np.sum(1 / (distance_matrix**self.repulsion_order))
    #     return E

    def get_potential_energy_v2(self, **kwargs):
        from coordinations import compute_dist_X_Y
        # real_positions = self.atoms.positions
        atoms_positions = self.atoms.positions.copy()
        if self.with_numbers:
            numbers = self.repulsion_numbers
        else:
            numbers = 1
        pseudo_positions = self.repulsion_points
        cross_distance_matrix = compute_dist_X_Y(
            atoms_positions, pseudo_positions)
        E = np.sum(numbers / (cross_distance_matrix **
                              self.repulsion_order * self.get_pseudo_masses_matrix()))
        return E

    def get_potential_energy(self, **kwargs):
        return self.get_potential_energy_v2(**kwargs)

    def get_forces(self, **kwargs):
        positions = np.array(self.atoms.positions).reshape((-1, 3))
        natoms = len(positions)
        nrepl = len(self.repulsion_points)
        repulsion_points = np.array([np.vstack((positions,
                                                self.repulsion_points))] * natoms).reshape((natoms, -1, 3))
        vx = -(repulsion_points - positions.reshape((natoms, -1, 3)))
        vx_div = np.power(np.sum(np.square(vx), axis=2),
                          (self.repulsion_order + 2.) / 2.).reshape((natoms, -1, 1))
        vx_div += np.hstack((np.diag([1] * natoms),
                             np.zeros((natoms, nrepl)))).reshape((natoms, natoms + nrepl, 1))
        vx /= vx_div
        vx *= self.repulsion_order
        forces = np.sum(vx, axis=1).reshape((-1, 3))
        return forces

    # def get_forces_new(self, **kwargs):
    #     positions = np.array(self.atoms.positions).reshape((-1, 3))
    #     natoms = len(positions)
    #     nrepl = len(self.repulsion_points)
    #     repulsion_points = np.array([np.vstack((positions,
    #                                        self.repulsion_points))]*natoms).reshape((natoms, -1, 3))
    #     vx1 = -(repulsion_points - positions.reshape((natoms, -1, 3)))
    #     vx1_div = np.power(np.sum(np.square(vx1), axis=2),
    #                        (12+2.)/2.).reshape((natoms, -1, 1))
    #     vx1_div += np.hstack((np.diag([1]*natoms),
    #                           np.zeros((natoms, nrepl)))).reshape((natoms, natoms+nrepl, 1))
    #     vx1 /= vx1_div
    #     vx1 *= 12
    #     vx2 = -(repulsion_points - positions.reshape((natoms, -1, 3)))
    #     vx2_div = np.power(np.sum(np.square(vx2), axis=2),
    #                        (6+2.)/2.).reshape((natoms, -1, 1))
    #     vx2_div += np.hstack((np.diag([1]*natoms),
    #                           np.zeros((natoms, nrepl)))).reshape((natoms, natoms+nrepl, 1))
    #     vx1 /= vx2_div
    #     vx1 *= -6
    #     forces = np.sum(vx1, axis=1).reshape((-1, 3)) + \
    #         np.sum(vx2, axis=1).reshape((-1, 3))
    #     return forces

    def set_repulsion_points(self, repulsion_points, repulsion_numbers):
        self.repulsion_points = repulsion_points.copy()
        self.repulsion_numbers = np.array(repulsion_numbers).copy()
        # logger.debug(f"repulsion_points: {repulsion_points}")
        cell = self.atoms.cell
        pbc = self.atoms.pbc
        kernel_position = self.kernel_position
        if pbc is True or isinstance(pbc, np.ndarray) and pbc.any():
            lelement = range(-1, 2)
            for i, j, k in itertools.product(lelement, lelement, lelement):
                if i == 0 and j == 0 and k == 0:
                    continue
                xpositions = repulsion_points + \
                    cell[0] * i + cell[1] * j + cell[2] * k
                self.repulsion_points = np.vstack(
                    (self.repulsion_points, xpositions))
            self.repulsion_numbers = np.vstack(
                [self.repulsion_numbers] * 27).flatten()
        self.repulsion_points = np.array(
            self.repulsion_points).reshape((-1, 3))
        dist_to_kernel_position = norm(
            self.repulsion_points - kernel_position, axis=1)
        repulsion_index = dist_to_kernel_position < self.max_repulsion_distance
        self.repulsion_points = self.repulsion_points[repulsion_index]
        self.repulsion_numbers = self.repulsion_numbers[repulsion_index]


def REP_spherical_opt(points, atoms, anchor, pbc=False, cell=None,
                      repulsion_order: float = 5., range_threshold: float = 6.,
                      maxstep: float = 0.4, fmax: float = 1E-4,
                      verbose: bool = False,
                      **kwargs):
    positions = atoms.get_positions()
    if isinstance(anchor, int):
        anchor = positions[anchor]
    pseudo_atoms = get_pseudo_atoms(
        points, atoms, anchor,
        pbc, cell, repulsion_order)
    res = opt_points(pseudo_atoms, points, anchor,
                     maxstep, fmax, verbose)
    return res


def get_pseudo_atoms(points, atoms, anchor,
                     pbc=False, cell=None,
                     repulsion_order: float = 5., with_numbers=False):
    """
    generate a pseudo atoms with all parameters,
    compatible with REP_spherical_opt
    """
    positions = atoms.get_positions()
    numbers = atoms.numbers
    if isinstance(anchor, int):
        anchor = positions[anchor]
    points = np.array(points).reshape((-1, 3))
    pseudo_atoms = PseudoAtoms(positions=points, pbc=pbc, cell=cell)
    pseudo_calc = PseudoCalc(atoms=pseudo_atoms,
                             kernel_position=anchor,
                             repulsion_points=positions,
                             repulsion_numbers=numbers,
                             repulsion_order=repulsion_order,
                             with_numbers=with_numbers
                             )
    pseudo_atoms.calc = pseudo_calc
    return pseudo_atoms


def opt_points(pseudo_atoms, points, anchor,
               maxstep: float = 0.4, fmax: float = 1E-4,
               verbose: bool = False):
    """
    use Spherical Optimization to optimize points on the sphere
    Input:
        pseudo_atoms: prepared pseudo_atoms with pseudo_calculator
        points: points which will be optimized
        anchor: anchor of the points, will be repeated len(points) times
        maxstep: max step of movement in optimization
        fmax: force creteria of convergence
    """
    from bfgs_sopt import BFGS_SOPT
    logfile = None
    trajectory = None
    if verbose:
        logfile = '-'
        trajectory = 'traj.traj'
    points = np.array(points).reshape((-1, 3))
    anchor = np.array([anchor] * len(points)).reshape((-1, 3))
    opt = BFGS_SOPT(atoms=pseudo_atoms, anchor=anchor, logfile=logfile,
                    trajectory=trajectory, maxstep=maxstep)
    origin_length = norm(points - anchor, axis=1)
    # print('start sopt')
    steps = 200
    res = {
        'converged': False
    }
    if opt.run(steps=steps, fmax=fmax):
        pseudo_positions = pseudo_atoms.positions
        energy = pseudo_atoms.calc.get_potential_energy()
        length = norm(pseudo_positions - anchor, axis=1)
        if norm(length - origin_length) < 0.1:
            res = {
                'converged': True,
                'positions': pseudo_positions,
                'energy': energy,
                'steps': opt.nsteps,
            }
        else:
            logger.debug('distance error')
    else:
        logger.debug('not converged')
    return res
