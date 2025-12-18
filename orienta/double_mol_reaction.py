import numpy as np
from numpy.linalg import norm

from reaction_positions import get_low_repulsion_points, REP_spherical_opt
from utils import get_bond_length
import pdb


def _move_atoms(atoms, idx, idxpos, rotpos, linevec):
    # move atoms
    if isinstance(idx, int):
        idx = [idx]
    pos = atoms.positions[idx].mean(axis=0)
    vec2 = (rotpos - pos)
    l_vec2 = norm(vec2)

    atoms.set_positions(atoms.positions + (idxpos - pos))
    vec2 = np.array([0, 1, 0]) if l_vec2 < 1e-3 else vec2 / l_vec2
    # rotpos +=(idxpos - pos)

    # rotate atoms
    if len(atoms) > 1:
        # rotpos=meanpos=atoms.positions.mean(axis=0)
        pos = atoms.positions[idx].mean(axis=0)
        angle = np.arccos(vec2.dot(linevec)) * 180 / np.pi
        # print(angle)
        r_vec = np.cross(vec2, linevec)
        l_r_vec = norm(r_vec)
        r_vec /= l_r_vec
        if norm(r_vec) > 1e-3:
            atoms.rotate(angle, r_vec, center=idxpos)
    return atoms


def merge_atoms(atoms1, atoms2, c1, c2,
                pr1=None, pr2=None, bond_length=5,
                acp_bond_length=None):
    from ase import Atom
    atoms1 = atoms1.copy()
    atoms2 = atoms2.copy()
    if isinstance(c1, int):
        c1 = [c1]
    if isinstance(c2, int):
        c2 = [c2]
    # pc1 = atoms1.positions[c1].mean(axis=0)
    # pc2 = atoms2.positions[c2].mean(axis=0)
    coeff1 = [3 if atoms1[c].symbol == 'H' else 1 for c in range(len(atoms1))]
    coeff2 = [3 if atoms2[c].symbol == 'H' else 1 for c in range(len(atoms2))]
    pc1 = sum(atoms1.positions[c] *coeff1[c] for c in c1) / sum(coeff1[c] for c in c1)
    pc2 = sum(atoms2.positions[c] *coeff2[c] for c in c2) / sum(coeff2[c] for c in c2)
    pc1 += np.random.random(3) * 0.1
    pc2 += np.random.random(3) * 0.1
    print(pc1, pc2, [coeff1[c] for c in c1], [coeff2[c] for c in c2])
    if pr1 is None or pr2 is None:
        p1s, e1s = get_low_repulsion_points(
            atoms1, c1, bond_length=acp_bond_length, min_need=1)
        pr1 = p1s[0][0]
        p2s, e2s = get_low_repulsion_points(
            atoms2, c2, bond_length=acp_bond_length, min_need=1)
        # if len(p2s) == 0: pdb.set_trace()
        pr2 = p2s[0][0]
        # atoms2 +=Atom('He', pr2)

    vec = pr1 - pc1
    l_vec = norm(vec)
    vec = np.array([1, 0, 0]).astype(np.float) if l_vec < 1e-3 else vec / l_vec

    # bond_length=5
    idxpos = pc1 + bond_length * vec
    rotpos = pr2
    atoms2 = _move_atoms(atoms2, c2, idxpos, rotpos, -vec)
    atoms = atoms1 + atoms2
    return atoms, pr1, pr2


def run_2atoms_with_sites(
        atoms1, atoms2, c1, c2, inputbond_length=None, label='',
        fixed1=list(), fixed2=list(), use_drag=False):
    from chemdata import get_element_covalent
    # print("Running", label)
    natoms1 = len(atoms1)
    if isinstance(c1, int):
        ele11, ele12 = atoms1.symbols[c1], None
        l1 = get_element_covalent(ele11)
    else:
        ele11, ele12 = atoms1.symbols[c1[0]], atoms1.symbols[c1[1]]
        l1 = (get_element_covalent(ele11) + get_element_covalent(ele12)) / 2
    if isinstance(c2, int):
        ele21, ele22 = atoms2.symbols[c2], None
        l2 = get_element_covalent(ele21)
    else:
        ele21, ele22 = atoms2.symbols[c2[0]], atoms2.symbols[c2[1]]
        l2 = (get_element_covalent(ele21) + get_element_covalent(ele22)) / 2
    bond_length = (l1 + l2)
    # bond_length = get_bond_length(ele1, ele2)
    # ele2 = atoms2.symbols[c2]
    # bond_length1 = get_bond_length(ele11, ele2)
    # bond_length2 = get_bond_length(ele12, ele2)
    # bond_length = min(bond_length1, bond_length2)
    bond_length = max(bond_length, 0.6)
    bond_length = inputbond_length or bond_length
    _bondlength = max(0.5+bond_length, 1.5)
    # _bondlength = bond_length + 0.5
    # bond_length = _bondlength = 2.24
    print("Bond length:", bond_length, _bondlength)
    init_start, pr1, pr2 = merge_atoms(
        atoms1, atoms2, c1, c2,
        bond_length=_bondlength,
        acp_bond_length=bond_length)
    init_end, _, _ = merge_atoms(
        atoms1, atoms2,
        c1, c2, pr1, pr2,
        bond_length=bond_length * 0.90)
    fixed_atoms = []
    if isinstance(c1, int):
        fixed_atoms.append(c1)
    else:
        fixed_atoms += c1

    if isinstance(c2, int):
        fixed_atoms.append(c2+natoms1)
        idx = c2 + natoms1
    else:
        fixed_atoms += [_+natoms1 for _ in c2]
        idx = [_+natoms1 for _ in c2]

    fixed_atoms += fixed1
    fixed_atoms += [_+natoms1 for _ in fixed2]
    drag_init_end, drag_images = None, None
    return init_start, init_end, drag_init_end, drag_images


def run2(name1, name2, _dict1, _dict2, settings,
         is_surface_reaction=False, reaction_pool=dict()):
    import itertools
    import utils
    atoms1 = _dict1['atoms']
    atoms2 = _dict2['atoms']
    fixed_atoms = _dict1['fixed_atoms']
    equivalent_atoms1 = _dict1['equivalent_atoms']
    rev_idx1 = equivalent_atoms1['reverse_index']
    equivalent_atoms2 = _dict2['equivalent_atoms']
    rev_idx2 = equivalent_atoms2['reverse_index']

    natoms1 = len(atoms1)
    natoms2 = len(atoms2)
    active_site1 = np.arange(natoms1)[_dict1['selected_sites']].tolist()
    active_site2 = np.arange(natoms2)[_dict2['selected_sites']].tolist()
    print("active_site1", active_site1, "active_site2", active_site2)
    if name1 != name2:
        active_site_pairs = list(itertools.product(active_site1, active_site2))
    else:
        active_site_pairs = list(itertools.combinations_with_replacement(
            active_site1, 2))

    print("active_site_pairs", active_site_pairs)
    res = []
    # attack atom
    # import pdb; pdb.set_trace()
    restmp = []
    for site1, site2 in active_site_pairs:
        # test if name exist
        name = '{}:{} + {}:{}'.format(name1, site1, name2, site2)
        if name in reaction_pool:
            continue
        # search equivalent
        flag = False
        for _n, _s1, _s2 in restmp:
            if rev_idx1[site1] == rev_idx1[_s1] and rev_idx2[site2] == rev_idx2[_s2]:
                flag = True
                break
        if flag:
            continue
        restmp.append([name, site1, site2])
    res += restmp

    # attack bond
    restmp = []
    print("Attack bond", list(itertools.combinations(active_site1, 2)))
    for site11, site12 in itertools.combinations(active_site1, 2):
        # test bond length
        bond_length = utils.get_bond_length(
            atoms1.numbers[site11], atoms1.numbers[site12])
        print(atoms1.get_distance(site11, site12), bond_length)
        if atoms1.get_distance(site11, site12) > bond_length * 1.2 :
            print("Distance too long")
            continue  # distance too long
        # test equivalent
        flag = False
        print(site11, site12)
        # for _n, [_s11, _s12], _ in restmp:
        #     if rev_idx1[site11] == rev_idx1[_s11] and rev_idx1[site12] == rev_idx1[_s12] or \
        #             rev_idx1[site11] == rev_idx1[_s12] and rev_idx1[site12] == rev_idx1[_s11]:
        #         flag = True  # This has been searched
        #         break
        if flag:
            continue

        # combine with single site of atoms2
        for site2 in active_site2:
            # prevent repeat name
            name = f'{name1}:{site11}-{site12} + {name2}:{site2}'
            if name in reaction_pool:
                continue
            # detect equivalent
            flag = False
            for _n, _s1, _s2 in restmp:
                if _s1 == [site11, site12] and type(_s2) is int and rev_idx2[site2] == rev_idx2[_s2]:
                    flag = True
                    break
            if flag:
                continue
            restmp.append([name, [site11, site12], site2])
        # bond attack bond
        for site21, site22 in itertools.permutations(active_site2, 2):
            # for site21, site22 in itertools.combinations(active_site2, 2):
            # already have same name
            name = f'{name1}:{site11}-{site12} + {name2}:{site21}-{site22}'
            if name in reaction_pool:
                continue
            # test bond length
            bond_length = utils.get_bond_length(
                atoms2.numbers[site21], atoms2.numbers[site22])
            if atoms2.get_distance(site21, site22) > bond_length * 1.1:
                continue
            # search equivalent
            flag = False
            for _n, _s1, _s2 in restmp:
                if _s1 == [site11, site12] and type(_s2) is list:
                    _s21, _s22 = _s2
                    if rev_idx2[_s21] == rev_idx2[site21] and rev_idx2[_s22] == rev_idx2[site22] or \
                            rev_idx2[_s21] == rev_idx2[site22] and rev_idx2[_s22] == rev_idx2[site21]:
                        flag = True
                        break
            if flag:
                continue
            # good ones
            restmp.append([name, [site11, site12], [site21, site22]])
    res += restmp
    # print(res)

    # start merging
    # import pdb; pdb.set_trace()
    for name, site1, site2 in res:
        init_start, init_end, drag_init_end, drag_images = run_2atoms_with_sites(
            atoms1, atoms2, site1, site2, name, fixed1=fixed_atoms)
        # min_rot = True
        # if (init_start.cell > 0 ).any():
        #     min_rot = False
        # if images is None:
        #     images = utils.interpolate(init_start, init_end, nimages=nimages, min_rot=min_rot)
        outdict = {
            'valid_by_gase': True,
            'allowed': None,
            'reactants': [name1, name2],
            'natoms': [len(atoms1), len(atoms2)],
            'active_sites': [site1, site2],
            'mol_sites': [site1, site2 + len(atoms1) if isinstance(site2, int) else [_ + len(atoms1) for _ in site2]],
            'fixed_atoms': fixed_atoms,
            'ends': [init_start, init_end],
            'drag_init_end': drag_init_end,
            'drag_images': drag_images,
            'images': 'interpolate',
            # 'is_surface': is_surface_reaction,
        }
        yield name, outdict


if __name__ == '__main__':
    drag_atom_to_dest
