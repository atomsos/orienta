import os
import ase.io
from ase.optimize.bfgs import BFGS


def read_mol(mol):
    if os.path.exists(mol):
        import ase.io
        return ase.io.read(mol, index=-1)
    else:
        import ase.build
        return ase.build.molecule(mol)


def bfgs_opt(mol):
    from ase.optimize import bfgs
    opt = bfgs(mol)
    opt.run()


def test_broken(mol):
    from .mol import get_bond_connecting_matrix, get_independent_molecules
    splited_atoms, _ = get_independent_molecules(mol)
    return len(splited_atoms) > 1


def constrainted_opt_until_bond_break(mol):
    opt = BFGS(mol, maxstep=0.2, trajectory="opt.traj")
    flag_broken = 0
    for i, _ in enumerate(opt.irun(fmax=0.10)):
        if (i+1) % 5 == 0:
            if test_broken(mol):
                print("molecule broken")
                flag_broken = i
                atoms.constraints = []
        if i - flag_broken >= 3:
            return True
        # if i >= 5:
        #     mol.constraints = []
        if i >= 100:
            break
    mol.constraints = []
    for i, _ in enumerate(opt.irun(fmax=0.10)):
        if i >= 100:
            break
    return False


def main(mol1, mol2, site1, site2, bond_length=None, charge=0,
         output="", run_opt=False, run_neb=False, nimages=5):
    from double_mol_reaction import run_2atoms_with_sites
    from utils import sopt_optimize_structure

    mol1 = read_mol(mol1)
    mol2 = read_mol(mol2)
    # if len(site1) == 1:
    #     site2 = site1[0]
    # if len(site2) == 1:
    #     site2 = site2[0]
    print("INPUT Bond length:", bond_length)
    init_start, init_end, _, _ = run_2atoms_with_sites(
        mol1, mol2,
        site1[0] if len(site1) == 1 else site1,
        site2[0] if len(site2) == 1 else site2, bond_length)
    init_start.write(output + "-" + "init_start.xyz")
    init_end.write(output + "-" + "init_end.xyz")

    #
    from ase.constraints import FixBondLength
    site2 = [_ +len(mol1) for _ in site2]
    constraints = [FixBondLength(a, b) for a, b in zip(site1, site2 + site2)]
    print(site2)
    print(constraints)
    init_end.constraints = constraints
    if run_opt:
        from ase.calculators.gaussian import Gaussian
        def get_calc(charge):
            return Gaussian(label='calc/gaussian',
                        xc='geom=nocrowd B3LYP',
                        basis='lanl2dz',
                        # basis='6-31+G(d,p)',
                        nprocshared=4,
                        charge=charge,
                        scf='maxcycle=100')

        optend = output+"-opted_end.xyz"
        optstart = output+"-opted_start.xyz"
        if not os.path.exists(optend):
            opted_end = init_end.copy()
            opted_end.calc = get_calc(charge)
            constrainted_opt_until_bond_break(opted_end)
            del opted_end.constraints
            opted_end.write(optend)
        else:
            opted_end   = ase.io.read(optend)

            # opt(init_end)

        if not os.path.exists(optstart):
            opted_start = init_start.copy()
            opted_start.calc = get_calc(charge)
            print("RUN SOPT start")
            opted_start, _ = sopt_optimize_structure(opted_start, opted_end, trajectory="start.traj")
            print("FINISH SOPT start")
            del opted_start.constraints
            opted_start.write(optstart)
        else:
            opted_start = ase.io.read(optstart)

        if run_neb:
            print("RUN NEB")
            opted_end   = ase.io.read(optend)
            opted_start = ase.io.read(optstart)
            try:
                opted_end.get_forces()
            except:
                opted_end.calc = get_calc(charge)
                opted_end.get_forces()
                opted_end.write(optend)
            try:
                opted_start.get_forces()
            except:
                opted_start.calc = get_calc(charge)
                opted_start.get_forces()
                opted_start.write(optstart)

            from ase.mep.neb import NEB
            # nimages = 5
            images = [opted_start] + [opted_start.copy() for _ in range(nimages-2)] + [opted_end]
            for img in images[1:-1]:
                img.calc = get_calc(charge)
            # import pdb; pdb.set_trace()
            neb = NEB(images, climb=True, remove_rotation_and_translation=True)
            # neb.interpolate()
            neb.interpolate(method='idpp')
            ase.io.write(f"{output}-imgstart.xyz", neb.images)
            nebfile = output+".traj"
            opt = BFGS(neb, trajectory=nebfile)
            opt.run(fmax=0.1, steps=100)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("mol1", type=str)
    parser.add_argument("mol2", type=str)
    parser.add_argument("--site1", nargs="*", type=int)
    parser.add_argument("--site2", nargs="*", type=int)
    parser.add_argument("--charge", nargs="?", default=0, type=int)
    parser.add_argument("--output", default="output", type=str)
    parser.add_argument("--run_opt", action="store_true")
    parser.add_argument("--run_neb", action="store_true")
    parser.add_argument("--nimages", nargs="?", default=5, type=int)
    parser.add_argument("--bond-length", nargs="?", default=None, type=float)
    args = parser.parse_args()

    print(args)
    main(args.mol1, args.mol2, args.site1, args.site2, args.bond_length,
         args.charge, args.output, args.run_opt, args.run_neb, args.nimages)
