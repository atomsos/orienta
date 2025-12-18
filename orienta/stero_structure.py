"""
This module implements the social-permutation-invariant (SPRINT) coordinates
based similarity algorithm.

References
----------
Physical Review Letters, 107(8), 107(8), 085504-5.

"""


import os
import numpy as np

BASEDIR = os.path.dirname(os.path.abspath(__file__))

DEBUG = False


def is_same_stero_structure(atoms0, atoms1, method='USR', threshold=0.95):
    assert method in ['USR','weighted_USR', ] # 'SPRINT']
    if len(atoms0.numbers) != len(atoms1.numbers) or \
            sorted(atoms0.get_atomic_numbers()) != sorted(atoms1.get_atomic_numbers()):
        if DEBUG:
            print('not same num & kinds of atom')
        return False
    if method == 'weighted_USR':
        similarity = similarity_weighted_USR_kernel(atoms0, atoms1)
    elif method == 'USR':
        similarity = similarity_USR_kernel(atoms0, atoms1)
    # elif method == 'SPRINT':
    #     similarity = similarity_SPRINT_kernel(atoms0, atoms1)
    if DEBUG:
        print('similarity:', round(similarity, 2), 'threshold:', threshold)
    if similarity > threshold:
        return True
    return False


def similarity_kernel(v1, v2):
    """
    The kernal function for computing similarity scores.

    Parameters
    ----------
    v1 : array_like, (n, )
        An n-elements vector.
    v2 : array_like, (n, ) or (m, n)
        An n-elements vector or a matrix of m rows and n columns.

    Returns
    -------
    s : float or array_like
        A float if `v2` is an n-elements vector or a m-elements vector if `v2`
        is a matrix.

    """
    return 1.0 / (1.0 + __mean_abs_error__(v1, v2))


def get_atoms_fingerprints(atoms, method="SPRINT", *args):
    """
    Return the fingerprints for the given atoms.

    Parameters
    ----------
    atoms : Atoms
        The target atoms.
    method : str
        The fingerprints method. Defaults to 'SPRINT'. Other options are: 'USR'
        and 'weighted_USR'.
    args : tuple
        Additional arguments for the fingerprints method.

    Returns
    -------
    v : array_like
        The fingerprints vector.

    """
    assert(method in ['SPRINT', 'USR', 'weighted_USR'])
    if method == "SPRINT":
        return get_SPRINT_vector(
            atoms.cart_coords, get_atoms_interatomic_radii(atoms))
    elif method == "USR":
        v, vext = get_USR_features(atoms, *args)
        if vext is None:
            return v
        else:
            return np.append(v, vext)
    elif method == "weighted_USR":
        return get_weighted_USR_features(atoms)


def get_finperprints_dimension(natoms, method="SPRINT", extra=None):
    """
    Return the dimension of the fingerprints.

    Parameters
    ----------
    natoms : int
        The total number of atoms.
    method : str
        The fingerprints method.
    extra : str
        The extra descriptors for creating another set of features.

    Returns
    -------
    ndim : int
        The dimension for the given kind of fingerprints.

    """
    assert method in ['SPRINT', 'USR', 'weighted_USR']
    if method == "SPRINT":
        return natoms
    elif method == "USR":
        if extra is None or extra == 'none':
            return 12
        else:
            return 24
    elif method == "weighted_USR":
        return 12


def get_atoms_similarity(a, b, method="SPRINT", *args):
    """
    Return the geometry similarity of the atoms a and b.

    Parameters
    ----------
    a, b : Atoms
        Two non-periodic atoms.
    method : str
        The method used to calculate the similarity.
    args : tuple
        Additional arguments for the fingerprints method.

    Returns
    -------
    similarity : float
        The similarity between `a` and `b`.

    """
    if len(a) != len(b):
        return 0.0
    if sorted(a.get_atomic_numbers()) != sorted(b.get_atomic_numbers()):
        return 0.0
    va = get_atoms_fingerprints(a, method, *args)
    vb = get_atoms_fingerprints(b, method, *args)
    return similarity_kernel(va, vb)


def similarity_matrix(atoms, method="SPRINT", args=()):
    """
    Return the similarity matrix given a set of atoms.

    Parameters
    ----------
    atoms : Sized
        A set of atoms.
    method : str
        The method for calculating the similarities.
    args : tuple
        Additional arguments for the fingerprints method.

    Returns
    -------
    sim_mat : array_like
        The similarity matrix.

    """
    n = len(atoms)
    if n <= 1:
        raise ValueError("At least 2 atoms should be given!")
    natoms = len(atoms)
    s = np.eye(n)
    ndim = get_finperprints_dimension(natoms, method, *args)
    v = np.zeros((n, ndim))
    for i in range(n):
        v[i] = get_atoms_fingerprints(atoms[i], method, args)
    for i in range(n):
        s[i, i + 1:] = similarity_kernel(v[i], v[i + 1:])
        s[i + 1:, i] = s[i, i + 1:]
    return s


# def similarities_benchmark():
#     """
#     The benchmark test codes for these similarity algorithms.
#     """
#     import os
#     from time import time
#     from tabulate import tabulate
#
#     atoms = XYZ.from_file(os.path.join(
#         testdir(), "similarity", "RhB18.xyz")).atoms
#
#     tic = time()
#     sim_mat_USR = similarity_matrix(atoms, "USR", ("Rh", ))
#     t_USR = time() - tic
#
#     tic = time()
#     sim_mat_weighted_USR = similarity_matrix(atoms, "weighted_USR")
#     t_weighted_USR = time() - tic
#
#     tic = time()
#     sim_mat_SPRINT = similarity_matrix(atoms, "SPRINT")
#     t_SPRINT = time() - tic
#
#     def create_table(matrix):
#         data = []
#         n = len(matrix)
#         headers = ["Index"] + list(map(str, range(1, n + 1)))
#         for i in range(n):
#             row = [i + 1]
#             row.extend(matrix[i])
#             data.append(row)
#         return tabulate(data, headers=headers, floatfmt=".3f")
#
#     def test_output(matrix, total_time, algorithm):
#         print("Algorithm: {}".format(algorithm))
#         print("Time: {:.3f} seconds".format(total_time))
#         print(create_table(matrix))
#         print("\n")
#
#     test_output(sim_mat_USR, t_USR, "USR")
#     test_output(sim_mat_weighted_USR, t_weighted_USR, "Weighted USR")
#     test_output(sim_mat_SPRINT, t_SPRINT, "SPRINT")


# USR.py from tgmin2
def get_USR_features(atoms, extra=None):
    """
    Return the USR feature vector of the given atoms.

    Parameters
    ----------
    atoms : Atoms
        The target atoms.
    extra : str or int or Sized
        Extra descriptors for creating another set of features.

    Returns
    -------
    USR_vector, ext_vector : array_like
        The standard USR feature vector and the extra USR feature vector.
        If ``extra`` is None, ext_vector will also be None.

    """
    def get_vector(v1, v2, v3, v4, coords):
        vector = np.zeros(12)
        k = 0
        for v in [v1, v2, v3, v4]:
            di = np.linalg.norm(v - coords, axis=1)
            vector[k: k+3] = np.mean(di), np.std(di), skewness(di)
            k += 3
        return vector

    cart_coords = atoms.get_positions()
    x = cart_coords.mean(axis=0)
    d = np.linalg.norm(x - cart_coords, axis=1)
    y = cart_coords[np.argmin(d)]
    z = cart_coords[np.argmax(d)]
    d = np.linalg.norm(z - cart_coords, axis=1)
    w = cart_coords[np.argmax(d)]

    USR_vector = get_vector(x, y, z, w, cart_coords)
    if extra is not None:
        if isinstance(extra, str):
            ii = atoms.indices_from_symbol(extra)
        elif isinstance(extra, int):
            ii = [extra]
        elif isinstance(extra, (tuple, list)):
            ii = extra
        else:
            raise ValueError("")
        ext_vector = get_vector(x, y, z, w, cart_coords[np.ix_(ii)])
    else:
        ext_vector = None
    return USR_vector, ext_vector


def similarity_USR_kernel(a, b, extra=None):
    """
    The kernal function for computing the similarity of atoms a and b using
    the Ultrafast Shape Recognization (USR) algorithm.

    Parameters
    ----------
    a, b : Atoms or tuple or list
        Two non-periodic atoms or their feature vectors.
    extra : str or int or Sized
        Extra descriptors for computing USR vectors if ``a`` and ``b`` are
        atoms objects. Otherwise this is ignored.

    Returns
    -------
    sim : float
        The similarity.

    """

    from ase import Atoms

    def _process(o, s):
        err = "Unsupported input argument type: {}".format(type(s))
        if isinstance(o, (tuple, list)):
            if len(o) == 1 and len(o[0]) == 12:
                return o[0], None
            elif len(o) == 2:
                return o[0], o[1]
            else:
                raise ValueError(err)
        elif hasattr(o, 'positions'):
            return get_USR_features(o, extra=extra)
        elif isinstance(o, np.ndarray):
            if len(o) == 12:
                return o, None
            elif len(o) == 24:
                return o[:12], o[12:]
            else:
                raise ValueError("Unsupported array size: {:d}".format(len(o)))
        else:
            raise ValueError(err)

    va_USR, va_ext = _process(a, "a")
    vb_USR, vb_ext = _process(b, "b")
    mae = __mean_abs_error__(va_USR, vb_USR)
    if va_ext is None:
        return 1.0 / (1.0 + mae)
    else:
        mae = 0.5 * (mae + __mean_abs_error__(va_ext, vb_ext))
        return 1.0 / (1.0 + mae)


def get_weighted_USR_features(atoms):
    """
    Return the mass-weighted USR feature vector.

    Parameters
    ----------
    atoms : Atoms
        The target atoms.

    Returns
    -------
    USR_vector : array_like
        The mass-weighted USR feature vector.

    """
    cart_coords = atoms.get_positions()
    x = cart_coords.mean(axis=0)
    d = np.linalg.norm(x - cart_coords, axis=1)
    y = cart_coords[np.argmin(d)]
    z = cart_coords[np.argmax(d)]
    d = np.linalg.norm(z - cart_coords, axis=1)
    w = cart_coords[np.argmax(d)]
    m = atoms.get_masses()
    m /= m.mean()
    USR_vector = np.zeros(12)
    k = 0
    for v in [x, y, z, w]:
        di = np.linalg.norm(v - cart_coords, axis=1)
        di *= m
        USR_vector[k: k+3] = np.mean(di), np.std(di), skewness(di)
        k += 3
    return USR_vector


def similarity_weighted_USR_kernel(a, b):
    """
    The kernal function for computing the similarity of atoms a and b using
    the mass-weighted Ultrafast Shape Recognization (USR) algorithm.

    Parameters
    ----------
    a, b : Atoms or array_like
        Two non-periodic atoms or two feature vectors.

    Returns
    -------
    sim : float
        The similarity.

    """

    from ase import Atoms

    def _process(o, s):
        if isinstance(o, Atoms) or hasattr(o, 'get_positions'):
            return get_weighted_USR_features(o)
        elif isinstance(o, (tuple, list, np.ndarray)):
            return np.asarray(o)
        else:
            raise ValueError("{} should be a vector or a Atoms!".format(s))

    ua = _process(a, "a")
    ub = _process(b, "b")
    if len(ub.shape) == 2:
        axis = 1
    else:
        axis = None
    if DEBUG:
        print('ua:', ua, '\nub:', ub)
    return 1.0 / (1.0 + np.mean(np.abs(ua-ub), axis=axis))


# SPRINT.py from tgmin2
def get_SPRINT_vector(coords=None, rc=None, n=6, m=12, order=1, aij=None):
    """
    Return the topology SPRINT coordinates of a atoms. The contact matrix may
    be smoothed with the following eqn:

    aij = \frac{1 - (rij / rcij)^n}{1 - (rij / rcij)^m}

    where rij is the distance between site i and j and rcij is the default bond
    distance between site i and j.

    Parameters
    ----------
    coords : array_like, (n, 3) or (3n, ).
        The coordinates of a atoms.
    rc : float or array_like
        The default bond lengths. This can be a float or a (n,n) matrix.
    n, m : int, optional
        The exponential order.
    order : int, optional
        The walks of length. Defaults to 1.
    aij : array_like, optional
        The precomputed aij matrix. Defaults to None. If given, `coords` will be
        ignored.

    Returns
    -------
    vector : array_like
        The SPRINT coordinates. The shape is (n, ).

    """
    if aij is None:
        if len(coords.shape) == 1:
            coords = coords.reshape((-1, 3))

        natoms = len(coords)
        rij = __pairwise_euclidean_distances__(coords)
        rr = rij / rc
        aij = __divide__(1.0 - rr ** n, 1.0 - rr ** m)
    else:
        assert aij.shape[0] == aij.shape[1]
        natoms = len(aij)
    order = max(order, 1)
    v, w = np.linalg.eigh(aij)
    s = 1.0 / np.power(v.max(), order - 1) * np.sqrt(natoms)
    SPRINTs = np.dot(aij, np.abs(w[:, -1])) * s
    return np.sort(SPRINTs, kind='mergesort')


# def SPRINT_benchmark():
#     """
#     The benchmark run of using SPRINT coordinates as the fingerpints to
#     distinguish the 389 Co@B22- clusters obtained by TGMin2.
#
#     Job 276 and 351 are failed with unexpected reasons.
#     17 duplicates are found using the SPRINT fingerprints while 29 duplicates
#     are found using the USR fingerprints.
#
#     Job 225 (350) should be the most stable structure, but USR marks it as a
#     duplicates of Job 205.
#     """
#     from GRRMS.atoms import Atoms
#     import os
#
#     ar = np.load(os.path.join('tgmin_test', "similarity", "CoB22p.npz"))
#     xyz = ar["xyz"]
#     ids = ar["id"]
#     species = ["Co"] + ["B"] * 22
#     natoms = len(species)
#     r = np.atleast_2d([get_pyykko_radii(s) for s in species])
#     rc = r + r.T
#     ntotal = len(ids)
#     thres = 0.975
#
#     USR_v = np.zeros((ntotal, 12))
#     USR_dup = []
#     spr_v = np.zeros((ntotal, natoms))
#     spr_dup = []
#
#     for i in range(ntotal):
#         coords = xyz[i]
#         USR_v[i] = get_USR_features(Atoms(species, coords))[0]
#         spr_v[i] = get_SPRINT_vector(coords, rc)
#
#     for i in range(ntotal - 1):
#         USR_s = 1.0 / (1.0 + __mean_abs_error__(USR_v[i], USR_v[i + 1:]))
#         USR_indices = np.where(USR_s > thres)[0]
#         USR_n = len(USR_indices)
#         if USR_n > 0:
#             USR_dup.extend([j + i + 1 for j in USR_indices])
#
#         spr_s = 1.0 / (1.0 + __mean_abs_error__(spr_v[i], spr_v[i + 1:]))
#         spr_indices = np.where(spr_s > thres)[0]
#         if len(spr_indices) > 0:
#             spr_dup.extend([j + i + 1 for j in spr_indices])
#
#         if USR_n == 0:
#             continue
#
#         idi = ids[i]
#         print("Job id = %3d" % idi)
#
#         for k, j in enumerate(range(i + 1, ntotal)):
#             USR_isdup = USR_s[k] > thres
#             spr_isdup = spr_s[k] > thres
#
#             if (not USR_isdup) and (not spr_isdup):
#                 continue
#
#             print("    id = %3d, SPRINT: %5s (%.3f), USR: %5s (%.3f)" % (
#                 ids[j], spr_isdup, spr_s[k], USR_isdup, USR_s[k]))
#
#     spr_dup = set(map(int, spr_dup))
#     USR_dup = set(map(int, USR_dup))
#
#     print("SPRINT duplictes: %2d/%3d" % (len(spr_dup), ntotal))
#     print("USR    duplictes: %2d/%3d" % (len(USR_dup), ntotal))


def get_atoms_interatomic_radii(atoms, pyykko=True):
    """
    Return the interatomic covalent radii matrix, `rmat`. `rmat[i, j]` is the
    sum of the covalent radii of site i and j.

    Parameters
    ----------
    atoms : Atoms
        A atoms.
    pyykko : bool
        If True, the Pyykko radii will be used.

    Returns
    -------
    rmat : array_like
        The interatomic covalent radii matrix.

    """
    rr = np.atleast_2d(get_atoms_radii(atoms, pyykko))
    return rr.T + rr


def get_atoms_radii(atoms, pyykko=True):
    """
    Return the covalent radii for all atoms of the atoms.

    Parameters
    ----------
    atoms : Atoms
        A atoms.
    pyykko : bool
        If True, the Pyykko radii will be used.

    Returns
    -------
    radii : array_like
        The covalent radii for each atom of the given atoms.

    """
    return np.asarray(
        map(get_pyykko_radii if pyykko else get_covalent_radius,
            atoms.species))


def get_covalent_radius(element):
    """ 
    Return the covalent radius of the given element.

    Parameters
    ----------
    element : Element or str
        An Element object or an atom symbol string like 'H', 'Cl'.

    Returns
    -------
    r : float
        The covalent radius of the element.

    Raises
    ------
    ValueError
        If the given element is incorrect.

    """
    import json
    try:
        with open(os.path.join(BASEDIR, "atomdb.json")) as fd:
            rcov_db = json.loads(fd)
    except Exception as exception:
        raise NotImplementedError(
            "Failed to read the covalent radii database!")
    if element.__class__.__name__ == 'Element':
        element = element.symbol
    try:
        return rcov_db.get(element)["covalent_radius"]
    except KeyError:
        raise ValueError("{} is not an element!".format(element))


def get_pyykko_radii(element, order=0):
    """
    Return the Pyykko radii of the given element.

    Parameters
    ----------
    element : Element or str
        An Element object or an atom symbol string like 'H', 'Cl'.
    order : int
        The bond order.

    Returns
    -------
    radii : Sized
        The three Pyykko radii.

    Raises
    ------
    ValueError
        If the given element is incorrect.

    """
    import json
    try:
        with open(os.path.join(BASEDIR, "atomdb.json")) as fd:
            rcov_db = json.loads(fd)
    except Exception as exception:
        raise NotImplementedError(
            "Failed to read the covalent radii database!")

    assert 0 <= order <= 2
    if element.__class__.__name__ == 'Element':
        element = element.symbol
    try:
        return rcov_db.get(element)["pyykko"][order]
    except KeyError:
        raise ValueError("{} is not an element!".format(element))


# mathutils.py from tgmin2
def skewness(vector):
    """
    This function returns the cube root of the skewness of the given vector.

    Parameters
    ----------
    vector : array_like
        A vector.

    Returns
    -------
    skewness : float
        The skewness of the vector.

    References
    ----------
    http://en.wikipedia.org/wiki/Skewness
    http://en.wikipedia.org/wiki/Moment_%28mathematics%29

    """
    v = np.asarray(vector)
    sigma = np.std(v)
    s = np.mean((v - v.mean())**3.0)
    eps = 1E-8
    if np.abs(sigma) < eps or np.abs(s) < eps:
        return 0.0
    else:
        return s / (sigma**3.0)


def __mean_abs_error__(v1, v2):
    """
    Return the mean absolution error.

    Parameters
    ----------
    v1 : array_like, (n, )
        A vector with `n` elements.
    v2 : array_like, (n, ) or (m, n)
        A vector with `n` elements or a matrix with `m` rows and `n` columns.

    Returns
    -------
    mae : float or array_like
        The mean absolute error between `v1` and `v2`. May be a float if `v2` is
        a vector or a vector with `m` elements if `v2` is a (m, n) matrix.

    """
    if len(v1.shape) >= 2:
        raise ValueError("`v1` should be a vector!")
    if len(v2.shape) == 2:
        axis = 1
    else:
        axis = None
    return np.mean(np.abs(v1 - v2), axis=axis)


def __divide__(a, b):
    """
    Ignore / 0, __divide__( [-1, 0, 1], 0 ) -> [0, 0, 0].

    References
    ----------
    stackoverflow.com/questions/26248654/numpy-return-0-with-__divide__-by-zero

    """
    with np.errstate(__divide__='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        if not isinstance(c, np.ndarray):
            c = 0.0
        else:
            c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


def __pairwise_euclidean_distances__(x):
    """
    Compute the Euclidean distance matrix from a vector array X.

    Parameters
    ----------
    x : array_like, [n, n]
        An array of feature vectors.

    Returns
    -------
    d : array_like, [n, n]
        A distance matrix D such that D_{i, j} is the distance between the ith
        and jth vectors of the given matrix X.

    """
    x = np.asarray(x)
    assert len(x.shape) == 2
    n = len(x)
    d = np.zeros((n, n))
    for i in range(n - 1):
        d[i, i + 1: n] = np.linalg.norm(x[i] - x[i + 1: n], axis=1)
        d[i + 1: n, i] = d[i, i + 1: n]
    return d

