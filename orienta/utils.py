import os
import numpy as np
from numpy.linalg import norm
import tempfile
import ase.data
import ase.data.dbh24 as dbh24
import ase.io
from ase.constraints import FixAtoms, FixBondLength

import itertools
import modlog
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

import chemio
from gase.aseshell import AtomsShell
from gase.generator.reaction_positions import get_low_repulsion_points, REP_spherical_opt
import gase.ts.neb

import gase.db.use_sqlite3 as database

# from verifyneb import verifyMEP, verifyActiveSite
import pdb


global atoms_pool, reaction_map, run1_runed, run2_runed, neb_images
global _terminate

dry_run = True
dry_run = False
atoms_pool = {}
neb_images = {}
reaction_map = {}
run1_runed = []
run2_runed = []
_terminate = False


def get_tmpdir():
    try:
        tmpdir = tempfile.mkdtemp(prefix='/dev/shm/')
    except:
        tmpdir = tempfile.mkdtemp()
    print("Temp dir:", tmpdir)
    return tmpdir


def initialize_db():
    global atoms_pool, reaction_map, run1_runed, run2_runed, neb_images
    for name in ['atoms_pool', 'reaction_map', 'runed', 'neb_images', 'unopted_neb_images']:
        database.create_table(name)
    atoms_pool = dict(database.read('atoms_pool'))
    neb_images = dict(database.read('neb_images'))

    reaction_map = database.read('reaction_map', 'rmap') or {}
    run1_runed = database.read('runed', 'run1_runed') or []
    run2_runed = database.read('runed', 'run2_runed') or []


def terminate():
    global _terminate
    _terminate = True


def test_is_surface(atoms):
    cell = atoms.get_cell()
    if cell[0].dot(np.cross(cell[1], cell[2])) <= 0.1:
        return False
    # positions = atoms.positions


def get_bond_length(ele1, ele2):
    if isinstance(ele1, str):
        n1 = ase.data.chemical_symbols.index(ele1)
    else:
        n1 = int(ele1)
    if isinstance(ele1, str):
        n2 = ase.data.chemical_symbols.index(ele2)
    else:
        n2 = int(ele2)
    l1 = ase.data.covalent_radii[n1]
    l2 = ase.data.covalent_radii[n2]
    return l1 + l2


def interpolate(start, end, intermediate=None,
                nimages=7, min_rot=None):
    import gase.ts.img_utils
    start = start.copy()
    end = end.copy()
    if min_rot is None:
        min_rot = True
        if is_surface_reaction(start):
            min_rot = False

    cell = start.get_cell()
    # cellx, celly, cellz = cell[0], cell[1], cell[2]
    dpos = end.positions - start.positions
    for i, d in enumerate(dpos):
        for j in range(3):
            if d[j] > cell[j][j] * 0.75:
                end.positions[i][j] -= cell[j][j]
            elif d[j] < -cell[j][j] * 0.75:
                end.positions[i][j] += cell[j][j]

    images = gase.ts.img_utils.interpolate_images(
        start, end, intermediate=intermediate,
        nimages=nimages, min_rot=min_rot)
    for i in range(0, len(images)):
        calc_arrays = {}
        img = images[i]
        img.calc_arrays = calc_arrays
        img.reset_calc()
    return images


def sopt_interpolate(A, B, nimages,
                     calc_arrays, exec_arrays,
                     sopt_fmax=0.5, min_rot=True):
    # from gase.build import
    A, B = A.copy(), B.copy()

    sopt_fmax = 0.5
    sA, sB = A, B
    lA, lB = 'A', 'B'
    for x in [A, B]:
        x.calc_arrays = calc_arrays
        x.exec_arrays = exec_arrays
        x.reset_calc()
        x.get_forces()
    A, B = gase.ts.neb.run_sopt(
        A, B, trajectory='/dev/null',
        sopt_steps=30, sopt_fmax=sopt_fmax,
        min_rot=min_rot)
    B, A = gase.ts.neb.run_sopt(
        B, A, trajectory='/dev/null',
        sopt_steps=30, sopt_fmax=sopt_fmax,
        min_rot=False)

    sopt_fmax = 1.0
    simages = ['start', A.copy()]
    eimages = [B.copy()]
    while len(simages) + len(eimages) < nimages + 1:
        dpos = (B.positions - A.positions) / \
            (nimages - len(simages) - len(eimages) + 2)
        newatoms = A.copy()
        newatoms.positions += dpos
        R2 = norm(newatoms.positions - sA.positions)
        R22 = norm(newatoms.positions - sB.positions)
        print("start R2:", lA, R2, lB, R22)
        newatoms.get_forces()
        newatoms, _ = gase.ts.neb.run_sopt(
            newatoms, sB, trajectory='/dev/null',
            sopt_steps=30, sopt_fmax=sopt_fmax,
            min_rot=False)
        R2 = norm(newatoms.positions - sA.positions)
        R22 = norm(newatoms.positions - sB.positions)
        print("end   R2:", lA, R2, lB, R22)
        simages.append(newatoms.copy())
        A = newatoms
        A, B = B, A
        sA, sB = sB, sA
        lA, lB = lB, lA
        simages, eimages = eimages, simages
        if lA == 'A':
            ase.io.write('tmptest.traj', (simages +
                         list(reversed(eimages)))[1:])
        else:
            ase.io.write('tmptest.traj', (eimages +
                         list(reversed(simages)))[1:])

    if simages[0] != 'start':
        simages, eimages = eimages, simages
    eimages.reverse()
    images = (simages + eimages)[1:]
    Es = [_.get_potential_energy() for _ in images]
    idx = np.argmax(Es)
    ts = images[idx]
    return images, ts


def sopt_bisect(A, B, nimages):
    pass


def _sopt_interpolate(A, B, nimages,
                      calc_arrays, exec_arrays,
                      fmax=0.5):
    """
    interpolate with sopt
    """
    A = A.copy()
    B = B.copy()
    for x in [A, B]:
        x.calc_arrays = calc_arrays.copy()
        x.exec_arrays = exec_arrays.copy()
        x.reset_calc()
        x.get_forces()
    images = [A.copy()]
    istart = 1
    while istart < nimages - 1:
        dpos = (B.get_positions() - A.get_positions()) / (nimages - istart)
        newatoms = A.copy()
        newatoms.positions += dpos
        newatoms.calc_arrays = calc_arrays.copy()
        newatoms.exec_arrays = exec_arrays.copy()
        newatoms.reset_calc()
        f = newatoms.get_forces()
        if f.flatten().dot(dpos.flatten()) > 0:
            anchor = A
            flag = True
        else:
            anchor = B
            flag = False
        # opt = BFGS_SOPT(newatoms, anchor, fmax=fmax)
        # opt.run()
        newatoms, _ = gase.ts.neb.run_sopt(
            newatoms, anchor, trajectory='-',
            sopt_steps=30, sopt_fmax=5.0)
        images.append(newatoms.copy())
        A = newatoms.copy()
        istart += 1
    images.append(B.copy())
    return images


def get_non_redundant_positions(input_positions):
    if input_positions is None:
        return None
    oldpos = None
    output_positions = []
    for pos in input_positions:
        if oldpos is None or \
                norm(pos - oldpos) > 0.1:
            oldpos = pos
            output_positions.append(pos)
    return np.array(output_positions)


# def get_assemble_atoms(atoms):
#     from ase import Atoms
#     from ase.calculators.singlepoint import SinglePointCalculator
#
#     numbers = atoms.get_numbers()
#     positions = atoms.get_positions()
#     cell = atoms.get_cell()
#     energy = atoms.get_energy()
#     forces = atoms.get_forces()
#     # pdb.set_trace()
#     charges = [atoms.charge] + [0] * (len(atoms) - 1)
#     atoms = Atoms(numbers=numbers,
#                   positions=positions,
#                   charges=charges,
#                   cell=cell)
#     calculator = SinglePointCalculator(
#         atoms=atoms, energy=energy, forces=forces)
#     atoms.calculator = calculator
#     return atoms


default_opt_calc_arrays = {
    'name': 'Gaussian',
    'command': 'opt=(cartesian) cam-b3lyp 6-31+G(d, p)',
}

if dry_run:
    default_opt_calc_arrays = {
        'name': 'Gaussian',
        'command': 'opt=(cartesian, loose, maxcycle=1) cam-b3lyp 6-31+G(d, p)',
    }
default_force_calc_arrays = {
    'name': 'Gaussian',
    'command': 'force cam-b3lyp 6-31+G(d, p)'
}
default_exec_arrays = {
    'max_core': 16,
    'max_memory': 9,
    # 'dest_dir': get_tmpdir(),
    'dest_dir': '/tmp',
}


def get_gase(atoms,
             calc_arrays=default_force_calc_arrays,
             exec_arrays=default_exec_arrays):
    import copy
    arrays = copy.deepcopy(atoms.arrays)
    arrays = chemio.read(arrays)
    arrays['calc_arrays'] = calc_arrays
    arrays['exec_arrays'] = exec_arrays
    constraints = atoms.constraints.copy()
    atoms = AtomsShell(arrays=arrays)
    atoms.constraints = constraints
    _ = atoms.calc
    return atoms


class OptimizationError(Exception):
    pass


def optimize_structure(
        atoms,
        opt_calc_arrays=default_opt_calc_arrays,
        exec_arrays=default_exec_arrays):
    newatoms = get_gase(atoms,
                        calc_arrays=opt_calc_arrays,
                        exec_arrays=exec_arrays)
    # pdb.set_trace()
    old_positions = newatoms.get_positions()
    # try:
    if True:
        newatoms.get_forces(apply_constraint=False)
        opt_positions = newatoms.calc.get_all_positions()
        if opt_positions is not None:
            opt_positions = np.vstack(([old_positions], opt_positions))
    # except Exception as e:
    #     raise OptimizationError(e)
    # newatoms=get_assemble_atoms(newatoms)
    return newatoms, opt_positions


def sopt_optimize_structure(
        target, anchor,
        run_double=False,
        trajectory=None,
        force_calc_arrays=default_force_calc_arrays,
        exec_arrays=default_exec_arrays):
    # target = get_gase(target,
    #                   force_calc_arrays,
    #                   exec_arrays)
    # anchor = get_gase(anchor,
    #                   force_calc_arrays,
    #                   exec_arrays)
    target, anchor = gase.ts.neb.run_sopt(
        target, anchor, trajectory=trajectory,
        sopt_steps=50, sopt_fmax=0.2)
    if run_double:
        target, anchor = anchor, target
        target, anchor = gase.ts.neb.run_sopt(
            target, anchor, trajectory=trajectory,
            sopt_steps=50, sopt_fmax=0.2)
        target, anchor = anchor, target
    return target, anchor


def setup_images(images, fixed_atoms=None,
                 force_calc_arrays=default_force_calc_arrays,
                 exec_arrays=default_exec_arrays):
    for img in images:
        img.calc_arrays = force_calc_arrays
        img.exec_arrays = exec_arrays
        if fixed_atoms is not None and len(fixed_atoms) > 0:
            constraints = [FixAtoms(fixed_atoms)]
            img.fixed_atoms = fixed_atoms
            img.constraints = constraints
        img.reset_calc()
        _ = img.calc
    # return images


def optimize_images(
        images, fixed_atoms=None,
        force_calc_arrays=default_force_calc_arrays,
        exec_arrays=default_exec_arrays,
        neb_steps=50, dry_run=False,
):
    import gase.ts.neb
    setup_images(images, fixed_atoms=fixed_atoms,
                 force_calc_arrays=force_calc_arrays,
                 exec_arrays=exec_arrays)
    images, converged, analysis = gase.ts.neb.run_neb_images(
        images, force_calc_arrays,
        neb_steps=neb_steps, neb_maxstep=0.10)
    return images, converged, analysis


def pool_contain_atoms(atoms, pool):
    from gase.structure.mol import is_same_molecule
    issame = False
    oldname = None
    for name, oldatoms in pool.items():
        if isinstance(oldatoms, dict):
            oldatoms = oldatoms['atoms']
        if is_same_molecule(oldatoms, atoms):
            issame = True
            oldname = name
            break
    return issame, oldname


def split_atoms(atoms):
    from gase.structure.mol import get_independent_molecules
    return get_independent_molecules(atoms)


def has_cell(atoms):
    cell = atoms.get_cell()
    return cell[0].dot(np.cross(cell[1], cell[2])) > 0.0


def adjust_solid_surface(atoms):
    if not has_cell(atoms):
        return
    positions = atoms.get_positions()
    cellz = atoms.get_cell()[2][2]
    if cellz > 0:
        while (positions[:, 2] > cellz - 1).any():
            zz = positions[:, 2][positions[:, 2] > cellz - 1]
            positions[:, 2] += cellz - zz.min()
            positions[:, 2][positions[:, 2] >= cellz] -= cellz
    if atoms.calc and \
            hasattr(atoms.calc, '_stored_positions') and \
            atoms.calc._stored_positions is not None:
        atoms.calc._stored_positions = positions
    atoms.set_positions(positions)
    return


def get_randstring(N):
    import random
    import string
    randstring = ''.join(random.choice(string.ascii_lowercase)
                         for _ in range(N))
    return randstring


def get_mol_name(mol, pool_list=None):
    formula = mol.get_chemical_formula()
    mol_name = formula
    if pool_list is None:
        N = 10
        randstring = get_randstring(N)
        mol_name = formula + '_' + randstring
    else:
        if mol_name in pool_list:
            for i in range(10000):
                _name = f"{mol_name}_{i}"
                if not _name in pool_list:
                    mol_name = _name
                    break
    return mol_name


def get_products_data(atoms):
    global atoms_pool
    final_atoms_list, _vals = split_atoms(atoms)
    final_labels = []
    if len(final_atoms_list) == 1:
        final_atoms_list = [atoms]
    for mol in final_atoms_list:
        issame, oldname = pool_contain_atoms(mol, atoms_pool)
        if issame:
            final_labels.append(oldname)
        else:
            mol_name = get_mol_name(mol)
            atoms_pool[mol_name] = mol
            database.insert('atoms_pool', mol_name, mol)
            final_labels.append(mol_name)
    return final_labels


def get_candidate_pairs(atoms1, atoms2, same_mol, molstate1, molstate2):
    import itertools
    natoms1 = len(atoms1)
    natoms2 = len(atoms2)
    active_sites1 = molstate1['selected']
    active_sites2 = molstate2['selected']
    if active_sites1.dtype == bool:
        active_sites1 = np.arange(natoms1)[active_sites1].tolist()
        active_sites2 = np.arange(natoms2)[active_sites2].tolist()

    if not same_mol:
        return itertools.product(active_sites1, active_sites2)
        # return itertools.product(range(len(atoms1)), range(len(atoms2)))
    else:
        return itertools.combinations_with_replacement(active_sites1, 2)
        # return itertools.combinations_with_replacement(range(len(atoms1)), 2)


def move_atoms(atoms, idx, idxpos, rotpos, linevec):
    # move atoms
    pos = atoms.positions[idx]
    vec2 = (rotpos - pos)
    l_vec2 = norm(vec2)

    atoms.set_positions(atoms.positions + (idxpos - pos))
    vec2 = np.array([0, 1, 0]) if l_vec2 < 1e-3 else vec2 / l_vec2
    # rotpos +=(idxpos - pos)

    # rotate atoms
    if len(atoms) > 1:
        # rotpos=meanpos=atoms.positions.mean(axis=0)
        pos = atoms.positions[idx]
        angle = np.arccos(vec2.dot(linevec)) * 180 / np.pi
        # print(angle)
        r_vec = np.cross(vec2, linevec)
        l_r_vec = norm(r_vec)
        r_vec /= l_r_vec
        if norm(r_vec) > 1e-3:
            atoms.rotate(angle, r_vec, center=idxpos)
    return atoms


# def merge_atoms_v1(atoms1, atoms2, c1, c2, bond_length=5):
#     atoms1: Atoms=atoms1.copy()
#     atoms2: Atoms=atoms2.copy()
#
#     mc1=atoms1.positions.mean(axis=0)
#     p1=atoms1.positions[c1]
#     mc2=atoms2.positions.mean(axis=0)
#     p2=atoms2.positions[c2]
#     vec=(p1 - mc1)
#     l_vec=norm(vec)
#     vec=np.array([1, 0, 0]) if l_vec < 1e-3 else vec / l_vec
#
#     # move atoms2
#     newp2=mc1 + vec * (l_vec + bond_length)
#     atoms2.set_positions(atoms2.positions + (newp2 - p2))
#     # atoms=atoms1 + atoms2
#
#     # rotate atoms2
#     if len(atoms2) > 1:
#         mc2=atoms2.positions.mean(axis=0)
#         p2=atoms2.positions[c2]
#         vec2=(mc2 - p2)
#         l_vec2=norm(vec2)
#         vec2=np.array([0, 1, 0]) if l_vec2 < 1e-3 else vec2 / l_vec2
#         angle=np.arccos(vec2.dot(vec)) * 180 / np.pi
#         r_vec=np.cross(vec2, vec)
#         if norm(r_vec) > 1e-3:
#             atoms2.rotate(angle, r_vec, center=newp2)
#     atoms=atoms1 + atoms2
#     return atoms


def merge_atoms(atoms1, atoms2, c1, c2,
                pr1=None, pr2=None, bond_length=5):
    from ase import Atom
    atoms1 = atoms1.copy()
    atoms2 = atoms2.copy()
    if isinstance(c1, int):
        c1 = [c1]
    pc1 = atoms1.positions[c1].mean(axis=0)
    # if isinstance(c2, int):
    #     c2 = [c2]
    # pc2 = atoms2.positions[c2].mean(axis=0)
    pc2 = atoms2.positions[c2]
    if pr1 is None or pr2 is None:
        p1s, e1s = get_low_repulsion_points(atoms1, c1)
        pr1 = p1s[0][0]
        p2s, e2s = get_low_repulsion_points(atoms2, c2)
        pr2 = p2s[0][0]
        # atoms2 +=Atom('He', pr2)

    vec = pr1 - pc1
    l_vec = norm(vec)
    vec = np.array([1, 0, 0]).astype(np.float) if l_vec < 1e-3 else vec / l_vec

    # bond_length=5
    idxpos = pc1 + bond_length * vec
    rotpos = pr2
    atoms2 = move_atoms(atoms2, c2, idxpos, rotpos, -vec)
    atoms = atoms1 + atoms2
    return atoms, pr1, pr2


def _run2(atoms1, atoms2, c1, c2, label=''):
    if _terminate:
        return None
    print("Running", label)
    natoms1 = len(atoms1)
    if isinstance(c1, int):
        ele1 = atoms1.symbols[c1]
        ele2 = atoms2.symbols[c2]
        bond_length = get_bond_length(ele1, ele2)
    else:
        ele11, ele12 = atoms1.symbols[c1[0]], atoms1.symbols[c1[1]]
        ele2 = atoms2.symbols[c2]
        bond_length1 = get_bond_length(ele11, ele2)
        bond_length2 = get_bond_length(ele12, ele2)
        bond_length = min(bond_length1, bond_length2)
        bond_length = max(bond_length, 1.0)
    init_start, pr1, pr2 = merge_atoms(atoms1, atoms2, c1, c2,
                                       bond_length=max(2.0*bond_length, 2.0))
    init_end, _, _ = merge_atoms(atoms1, atoms2,
                                 c1, c2, pr1, pr2,
                                 bond_length=bond_length)
    return init_start, init_end


# def attack_bond(atoms1, atoms2, site1list, site2):
#     formula1 = atoms1.get_chemical_formula()
#     formula2 = atoms2.get_chemical_formula()
#     # ele1 = atoms1.get_chemical_symbols()[c1]
#     # ele2 = atoms2.get_chemical_symbols()[c2]
#     # reaction_string = "{}:{}({}), {}:{}({})".format(
#     #     formula1, ele1, c1, formula2, ele2, c2)
#     # print("running reaction:", reaction_string)
#     natoms1 = len(atoms1)
#     # pdb.set_trace()


def sopt_md_images_once(start, end, dest_dir='.'):
    import automd
    atoms = start.copy()
    fname = os.path.join(dest_dir, 'tmp.xyz')
    start.write(fname)
    top, itp = automd.generate_gromacs_topfile_itpfile(
        fname, dest_dir=dest_dir, use_geom_bond=True,
        use_geom_angle=True, use_geom_dihedral=True)
    images = []
    length = norm(end.positions - start.positions) / 20
    print("Length:", length)
    while True:
        # calc atoms
        atoms.calc_arrays = {
            'name': 'Gromacs', 'dest_dir': dest_dir,
            'topfile': top, 'itpfile': itp}
        atoms.reset_calc()
        _ = atoms.calc
        atoms, _ = gase.ts.neb.run_sopt(
            atoms, end,
            trajectory='/dev/null', logfile='-',
            sopt_steps=10, sopt_fmax=0.2)
        images.append(atoms.copy())
        ase.io.write("tmpimages.traj", images)
        # try break
        if norm(atoms.positions - end.positions) < 1e-3:
            break
        vec = end.positions - atoms.positions
        if norm(vec) < length:
            dpos = vec
        else:
            nvec = vec / norm(vec)
            dpos = nvec * length
        atoms.positions += dpos
    return images


def sopt_md_images(start, end, dest_dir='.'):
    rootpath = os.getcwd()
    images1 = sopt_md_images_once(start, end, dest_dir=dest_dir)
    os.chdir(rootpath)
    ase.io.write('images1.traj', images1)
    images2 = sopt_md_images_once(end, start, dest_dir=dest_dir)
    os.chdir(rootpath)
    ase.io.write('images2.traj', images2)
    return images1, images2


def run_ts(images, force_calc_arrays, ts_calc_arrays, exec_arrays):
    Es = [_.get_potential_energy() for _ in images]
    idx = np.argmax(Es)
    ts = images[idx]
    start, end = images[0], images[-1]
    ts, opted_pos = optimize_structure(
        ts, ts_calc_arrays, exec_arrays)
    # images = interpolate(start, ts, nimages//2+1) + \
    #     interpolate(ts, end, nimages//2)[1:]
    images = [start, ts, end]
    converged = (ts.calc.status == 'completed')
    return converged, ts, images


def _run_init_start_end(
        init_start, init_end, reactant_pool,
        reaction_string='',
        fixed_atoms=list(),
        active_points=list(),
        nimages=15, min_rot=None,
        force_calc_arrays=default_force_calc_arrays,
        opt_calc_arrays=default_opt_calc_arrays,
        exec_arrays=default_exec_arrays):
    # constraint opt
    optimized = False
    end = init_end.copy()
    constrainted_end_opt_positions = None
    # fixed_atoms = []

    if min_rot is None:
        min_rot = False
        if is_surface_reaction(init_start):
            min_rot = True
    while not optimized:
        # end.fixed_atoms = np.zeros(len(init_end), bool)
        # end.fixed_atoms[fixed_atoms] = True
        # end.fixed_atoms[active_points] = True
        print("active_points1", active_points)
        fixed_points = []
        # fixed_points = list(set(fixed_atoms + active_points))
        active1, active2 = active_points
        if not isinstance(active1, list):
            active1 = [active1]
        if not isinstance(active2, list):
            active2 = [active2]

        end.constraints = [FixBondLength(a1, a2) for a1, a2 in itertools.product(active1, active2)]
        # end.fixed_atoms = fixed_points
        end, opted_pos = optimize_structure(
            end, opt_calc_arrays, exec_arrays)
        constrainted_end = end.copy()
        if constrainted_end_opt_positions is None:
            constrainted_end_opt_positions = opted_pos
        elif opted_pos is not None:
            constrainted_end_opt_positions = np.vstack(
                (constrainted_end_opt_positions, opted_pos))
        if _terminate:
            exit(0)
        # fixed_points = fixed_atoms + active_points
        end.constraints = [FixAtoms(fixed_points)]
        print("end.constraints", end.constraints)
        maxforce = norm(end.get_forces(apply_constraint=True), axis=1).max()
        print("Max force:", maxforce)
        if maxforce < 0.2:
            print("constrainted force converged")
            optimized = True
        products_atoms, _vals = split_atoms(end)
        if len(products_atoms) > 1:
            print("constrainted splited")
            optimized = True
    # unconstraint opt
    unconstrainted_end_opt_positions = None
    end.constraints = []
    end.fixed_atoms = fixed_atoms
    if len(products_atoms) == 1:
        optimized = False
        while not optimized:
            # end.fixed_atoms = np.zeros(len(init_end), bool)
            # end.fixed_atoms[fixed_atoms] = True
            end.fixed_atoms = fixed_atoms
            end, opted_pos = optimize_structure(
                end, opt_calc_arrays, exec_arrays)
            unconstrainted_end_opt_positions = opted_pos if unconstrainted_end_opt_positions is None else np.vstack(
                (unconstrainted_end_opt_positions, opted_pos))
            # end, _ = optimize_structure(
            #     end, opt_calc_arrays, exec_arrays)
            # end.constraints = [FixAtoms(fixed_atoms)]
            active1, active2 = active_points
            end.constraints = [FixBondLength(a1, a2) for a1, a2 in itertools.product(active1, active2)]
            print("active_points:", active_points, "end.constraints", end.constraints)
            maxforce = norm(end.get_forces(
                apply_constraint=True), axis=1).max()
            print("Max force:", maxforce)
            if maxforce < 0.2:
                print("unconstrainted force converged")
                optimized = True
            products_atoms, _vals = split_atoms(end)
            if len(products_atoms) > 1:
                print("unconstrainted splited")
                optimized = True

    # for endmol in products_atoms:
    #     issame, oldname = pool_contain_atoms(endmol, reactant_pool)
    #     if issame:
    #         print("Ignore reaction, products and reactants have same mol",
    #               reaction_string)
    #         return {
    #             'valid': False,
    #             'ends': [init_start, end],
    #             'images': interpolate(init_start, end, nimages=nimages, min_rot=min_rot),
    #             'reason': 'products and reactants have same mol',
    #             'products_atoms': products_atoms,
    #         }

    # c2 +=len(atoms1)
    # # images=get_images(atoms, c1, c2)
    # start=atoms1 + atoms2
    # end=atoms
    start = init_start.copy()
    if fixed_atoms:
        start.fixed_atoms = fixed_atoms
        start.constraints = [FixAtoms(fixed_atoms)]

    # # run force calc for reorg atoms
    start, _ = optimize_structure(
        start, force_calc_arrays,
        exec_arrays)

    # run sopt
    # start, end = sopt_optimize_structure(
    #     start, end,
    #     run_double=False,
    #     force_calc_arrays=force_calc_arrays,
    #     exec_arrays=exec_arrays)

    # images = interpolate(start, end, nimages=nimages, min_rot=min_rot)

    # drop too close positions
    constrainted_end_opt_positions = get_non_redundant_positions(
        constrainted_end_opt_positions)
    unconstrainted_end_opt_positions = get_non_redundant_positions(
        unconstrainted_end_opt_positions)
    return {
        'valid':  True,
        'ends': [start, end],
        'images': 'interpolate',
        'products_atoms': products_atoms,
        'constrainted_end_opt_positions': constrainted_end_opt_positions,
        'unconstrainted_end_opt_positions': unconstrainted_end_opt_positions,
        'constrainted_end': constrainted_end,
    }


def dont_run1_atoms(name, atoms):
    global run1_runed
    if name in run1_runed:
        return True
    return True
    # return False


def dont_run2_atoms(name, atoms, name2, atoms2):
    global run2_runed, run1_runed
    if [name, name2] in run2_runed or [name2, name] in run2_runed:
        print([name, name2], 'calculated')
        return True
    return False


def verify_activate_site_callback(mols):
    global atoms_pool
    atoms_pool.update(mols)
    for name, molstate in mols.items():
        database.update('atoms_pool', name, molstate)


def get_solid_surface_idx(atoms, bottom=False, border=False):
    import chemdata
    numbers = atoms.numbers.tolist()
    positions = atoms.get_positions()
    cellz = atoms.get_cell()[2][2]
    cell = atoms.get_cell()
    cx = atoms.get_cell()[0]
    cy = atoms.get_cell()[1]
    cz = atoms.get_cell()[2]
    if cellz > 0:
        while (positions[:, 2] > cellz - 1).any():
            zz = positions[:, 2][positions[:, 2] > cellz - 1]
            positions[:, 2] += cellz - zz.min()
            positions[:, 2][positions[:, 2] >= cellz] -= cellz
    if bottom:
        positions[:, 2] = -positions[:, 2]
    natoms = len(positions)
    res = []
    # x +- cx; y +- cy
    cell = atoms.get_cell()
    cellx, celly = cell[0], cell[1]
    allnumbers = []
    allpositions = None
    for i in [0, -1, 1]:
        for j in [0, -1, 1]:
            npos = positions + i * cellx + j * celly
            if allpositions is None:
                allpositions = npos
            else:
                allpositions = np.vstack([allpositions, npos])
            allnumbers += numbers
    numbers = allnumbers
    positions = allpositions
    radii = np.array([chemdata.get_element_covalent(_) for _ in numbers])

    for i in range(natoms):
        pos = positions[i]
        idx = positions[:, 2] > pos[2]
        xpos = positions[idx]
        dpos = xpos - pos
        ndpos = norm(dpos, axis=1)
        zdpos = dpos[:, 2]
        costheta = zdpos / ndpos
        sintheta = np.sqrt(1 - costheta ** 2)
        sinradii = radii[idx] / ndpos
        cosradii = np.sqrt(1 - sinradii ** 2)
        # t0 = theta - radii
        sint0 = sintheta * cosradii - costheta * sinradii
        sin0 = 0.500
        sin00 = 0.520
        ontop = True
        if len(sint0[sint0 < sin00]) >= 3 or (sint0 < sin0).any():
            ontop = False
        # if i == 12: pdb.set_trace()
        # print(i, ontop, sint0[sint0 < sin0])
        if ontop:
            res.append(i)

    if border:
        for k in res.copy():
            pk = positions[k]
            p_cell = pk.dot(np.linalg.inv(cell))
            if not ((p_cell[:2] < 0.1).any() or (p_cell[:2] > 0.9).any()):
                res.remove(k)

    return res


def is_surface_reaction(atoms1, atoms2=None):
    def cellV(c): return c[0].dot(np.cross(c[1], c[2]))
    if cellV(atoms1.cell) > 0:
        return True
    if atoms2 and cellV(atoms2.cell) > 0:
        return True
    return False


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('atoms')
    args = parser.parse_args()
    import ase.io
    from ase.constraints import FixAtoms
    atoms = ase.io.read(args.atoms)
    res = get_solid_surface_idx(atoms, bottom=False)
    print(res)
    atoms.numbers[res] = 3
    atoms.constraints = [FixAtoms(res)]
    atoms.edit()
