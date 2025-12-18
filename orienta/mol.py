"""


test if the molecule is independent into two



"""

import chemdata
import numpy as np
import pdb


def compute_independent_indices_with_connection_matrix(connection_matrix):
    """
    get independent indices
    Input:
        connection_matrix: np.ndarray of bool, natoms * natoms,
                           True if connected and False if not connected
    Output:
        list of index list: independent atoms indices
    """
    connection_matrix = np.array(connection_matrix)
    assert np.logical_or(connection_matrix == 0, connection_matrix == 1).all(
    ), 'connection_matrix should be 0/1 or True/False matrix'
    connection_matrix = connection_matrix.astype(bool)
    length = len(connection_matrix)
    assert connection_matrix.ndim == 2 and \
        connection_matrix.shape == (length, length)
    independent_molecules = {
        0: [0],
    }
    n_independent_molecules = 0
    reverse_index = [0, ]
    # pdb.set_trace()
    for i in range(1, length):
        connected_atom_index = np.arange(i)[connection_matrix[i, :i]]
        if len(connected_atom_index) == 0:
            n_independent_molecules += 1
            independent_molecules[n_independent_molecules] = [i]
            reverse_index.append(n_independent_molecules)
            continue
        groups = list(set(np.array(reverse_index)[
            connected_atom_index].tolist()))
        reverse_index.append(-1)
        if len(groups) == 1:
            group_id = groups[0]
            independent_molecules[group_id].append(i)
            reverse_index[i] = group_id
        else:
            # belongs to multiple group, merge these group together
            merging_groupid = groups[0]
            merging_indices = [i]
            for _id in groups:
                merging_indices += independent_molecules[_id]
                del independent_molecules[_id]
            independent_molecules[merging_groupid] = merging_indices
            for _id in merging_indices:
                reverse_index[_id] = merging_groupid
    for _id, _mollist in independent_molecules.items():
        _mollist.sort()
    return independent_molecules, reverse_index


def get_independent_molecules_by_distance(atoms, connection_threshold=4.0):
    """
    Get independent molecules
    Input:
        atoms: Atoms like object, the atoms tested.
        connection_threshold: float, 4.0, 
            over connection_threshold will be regarded as non connecting.
    Output:
        True/False, single atoms list
    """
    from . import coordinations
    cell = None
    numbers = atoms.get_numbers()
    if atoms.pbc.any():
        cell = atoms.cell
        mask = np.arange(3)[np.logical_not(atoms.pbc)]
        cell[mask] = 0.
    dist_matrix = coordinations.compute_distance_matrix(
        atoms.positions, cell=cell)
    connection_matrix = dist_matrix < connection_threshold
    res, _ = compute_independent_indices_with_connection_matrix(
        connection_matrix)
    return len(res) != 1, res


def get_independent_indices(atoms, connection_matrix=None):
    if connection_matrix is None:
        connection_matrix = get_bond_connecting_matrix(atoms)
    independent_molecules, reverse_index = \
        compute_independent_indices_with_connection_matrix(connection_matrix)
    return independent_molecules, reverse_index


def get_independent_molecules(atoms, connection_matrix=None):
    if connection_matrix is None:
        connection_matrix = get_bond_connecting_matrix(atoms)
    independent_molecules, reverse_index = get_independent_indices(
        atoms, connection_matrix)
    splited_atoms = []
    # pdb.set_trace()
    for index, mol_indices in independent_molecules.items():
        satoms = atoms[mol_indices]
        splited_atoms.append(satoms)
    return splited_atoms, list(independent_molecules.values())


def get_covalent_matrix(atoms, extra_length=0.35):
    covalent_matrix = np.array([[chemdata.get_element_covalent(i) +
                                 chemdata.get_element_covalent(j)
                                 for i in atoms.numbers] for j in atoms.numbers])
    covalent_matrix += extra_length
    return covalent_matrix


def get_bond_connecting_matrix(atoms, extra_length=0.35):
    """
    return bond connecting matrix
    0, 1 for connecting or not
    """
    from gase.structure.coordinations \
        import compute_distance_matrix
    cell = None
    if atoms.cell[0].dot(np.cross(atoms.cell[1], atoms.cell[2])) > 0:
        cell = atoms.cell
    dist_matrix = compute_distance_matrix(atoms.positions, cell=cell)
    covalent_dist_matrix = get_covalent_matrix(atoms, extra_length=0) * 1.3
    connection_matrix = dist_matrix < covalent_dist_matrix
    np.fill_diagonal(connection_matrix, 0)
    return connection_matrix


def get_molecule_diameter(atoms):
    """
    return the diameter of atoms, i.d. the maximum bond distance between two atom
    Using floyd algorithm
    Attention:
        if the atoms contain two or more atoms, it will fail
    Input:
        atoms: Atoms
    Output:
        int, diameter of atoms
    """
    natoms = len(atoms.numbers)
    dist = np.zeros((natoms, natoms))
    dist.fill(np.inf)
    # np.fill_diagonal(dist, 0)
    dist[get_bond_connecting_matrix(atoms)] = 1
    path = np.zeros((natoms, natoms))
    path.fill(-1)
    path_times = np.zeros((natoms, natoms))
    # floyd algorithm
    for k in range(natoms):
        for i in range(natoms):
            for j in range(natoms):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    path[i][j] = k
                    path_times[i][j] += 1
    print('path: ', path)
    print('path_times: ', path_times)
    print('dist: ', dist)
    return int(np.max(dist))


def in_same_molecule(atoms, atomi: int, atomj: int, connection_matrix=None) -> bool:
    """
    is atomi, atomj the same molecule
    """
    if connection_matrix is None:
        connection_matrix = get_bond_connecting_matrix(atoms)
    _, reverse_index = compute_independent_indices_with_connection_matrix(
        connection_matrix)
    return reverse_index[atomi] == reverse_index[atomj]


def is_same_molecule(atoms1, atoms2):
    from .stero_structure import is_same_stero_structure
    if hasattr(atoms1, 'charge'):
        if not atoms1.charge == atoms2.charge:
            return False
    elif hasattr(atoms1, 'get_initial_charges'):
        if not atoms1.get_initial_charges().sum() == atoms2.get_initial_charges().sum():
            return False
    return is_same_stero_structure(atoms1, atoms2, 'weighted_USR')


