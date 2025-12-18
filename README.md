# Orienta


This project is used for merging two atoms with minimal VSPER repulsion.


## Usage


```bash

export PYTHONPATH=$PWD/orienta

python orienta/orienta.py mol1 mol2 --site1 site1  --site2 site2 --output outputname --charge 0 --run_opt --run_neb --nimages nimages

```

Then orienta will find molecule minimum repulsion and merge two molecule with specific sites, running with the outputname, charge, and NEB.





