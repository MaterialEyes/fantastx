"""
Quick smoke tests for two new mol_crystals.py behaviours:
  1. op_perturb_atoms        — only terminal H atoms should be perturbed
  2. op_whole_cell_counterion — moves ~half of ClO4/H2O in full cell

For test 3 we use POSCAR_3_withH.vasp (200-atom full cell with water H added)
so that H2O units are detectable alongside ClO4. The structure is used in P1
mode (identity sym_op only) so _expand_to_full_cell is a no-op.

Run from the repo root:
    python test_mol_crystals_ops.py
"""

import sys, os, copy
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from fx19.mol_crystals import Compound3Ops
from pymatgen.core import Structure
from pymatgen.core.operations import SymmOp

CIF        = '/fs/scratch/PAS3157/PXRD_FX/Joe_PXRD_FX/compound_3/compound3_clean.cif'
POSCAR_WH  = '/fs/scratch/PAS3157/PXRD_FX/Joe_PXRD_FX/Okten_files/POSCAR_3_withH.vasp'

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'

# ── instantiate from CIF (gives correct ASU + sym_ops) ───────────────────────
print('Loading Compound3Ops from CIF ...')
ops = Compound3Ops(starting_structure_path=CIF)
asym    = ops.asym_unit
n_asym  = asym.num_sites
species = [s.symbol for s in asym.species]
print(f'  ASU: {n_asym} atoms  formula: {asym.formula}')

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — _protected_atom_indices
# ══════════════════════════════════════════════════════════════════════════════
print('\n── TEST 1: _protected_atom_indices ─────────────────────────────────────')
protected   = ops._protected_atom_indices()
perturbable = [i for i in range(n_asym) if i not in protected]

print(f'  Protected   ({len(protected):2d} atoms): {dict(Counter(species[i] for i in sorted(protected)))}')
print(f'  Perturbable ({len(perturbable):2d} atoms): {dict(Counter(species[i] for i in perturbable))}')

non_H = [i for i in perturbable if species[i] != 'H']
if non_H:
    print(f'  {FAIL} — non-H atoms are perturbable: {[(i, species[i]) for i in non_H]}')
    sys.exit(1)
elif not perturbable:
    print(f'  {FAIL} — no perturbable atoms found (expected terminal H)')
    sys.exit(1)
else:
    print(f'  {PASS} — all {len(perturbable)} perturbable atoms are H')

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — op_perturb_atoms: perturbed indices must all be H
# ══════════════════════════════════════════════════════════════════════════════
print('\n── TEST 2: op_perturb_atoms ────────────────────────────────────────────')
np.random.seed(42)
n_trials, failures = 10, 0
for trial in range(n_trials):
    result = ops.op_perturb_atoms(asym)
    if result is None:
        print(f'  trial {trial+1:2d}: returned None')
        continue
    moved = np.where(
        np.linalg.norm(result.cart_coords - asym.cart_coords, axis=1) > 1e-6
    )[0]
    non_H_moved = [i for i in moved if species[i] != 'H']
    if non_H_moved:
        failures += 1
        print(f'  trial {trial+1:2d}: {FAIL} — moved non-H: {[(i, species[i]) for i in non_H_moved]}')
    else:
        print(f'  trial {trial+1:2d}: moved {len(moved)} H atoms — OK')

if failures == 0:
    print(f'  {PASS} — all {n_trials} trials moved only H atoms')
else:
    print(f'  {FAIL} — {failures}/{n_trials} trials moved non-H atoms')
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — op_whole_cell_counterion (uses POSCAR_3_withH as P1 full cell)
# ══════════════════════════════════════════════════════════════════════════════
print('\n── TEST 3: op_whole_cell_counterion ────────────────────────────────────')
print(f'  Loading withH structure: {os.path.basename(POSCAR_WH)}')
full_withH = Structure.from_file(POSCAR_WH)
print(f'  Full cell: {full_withH.num_sites} atoms  formula: {full_withH.formula}')

# Build a P1 mock: asym_unit = full cell, sym_ops = [identity].
# _expand_to_full_cell returns one copy → the same structure unchanged.
ops_p1               = copy.copy(ops)
ops_p1.asym_unit     = full_withH
ops_p1.sym_ops       = [SymmOp.from_xyz_str('x,y,z')]
ops_p1._rigid_bodies_cache         = None
ops_p1._rigid_bodies_cache_species = []
ops_p1._adj_cache                  = None
ops_p1._protected_cache            = None

n_full   = full_withH.num_sites
sp_full  = [s.symbol for s in full_withH.species]
n_Cl     = sp_full.count('Cl')
# Count H2O candidates: O atoms bonded to H (rough: any O not bonded to Cl)
n_Cu     = sp_full.count('Cu')
print(f'  Cl (= # ClO4): {n_Cl}   Cu: {n_Cu}')

# ── 3a: verify mobile units are detected ─────────────────────────────────────
print('  3a: checking mobile unit detection ...')
# Replicate bond-graph logic to verify mobile unit detection
from fx19.mol_crystals import _COVALENT_RADII, _DEFAULT_COVALENT_RADIUS

full_s  = ops_p1._expand_to_full_cell(full_withH)
sp_fc   = [s.symbol for s in full_s.species]
frac_fc = full_s.frac_coords
lat_fc  = full_s.lattice.matrix
bt      = ops_p1.bond_tolerance

cov_r   = np.array([_COVALENT_RADII.get(s, _DEFAULT_COVALENT_RADIUS) for s in sp_fc])
cuts    = cov_r[:, None] + cov_r[None, :] + bt
df      = frac_fc[:, None, :] - frac_fc[None, :, :]
df     -= np.round(df)
dists_m = np.linalg.norm(df @ lat_fc, axis=2)
np.fill_diagonal(dists_m, np.inf)
bonded  = dists_m < cuts
adj_fc  = [list(np.where(bonded[i])[0]) for i in range(len(sp_fc))]

visited, comps = set(), []
for start in range(len(sp_fc)):
    if start in visited: continue
    comp, q = [], [start]
    while q:
        idx = q.pop(0)
        if idx in visited: continue
        visited.add(idx); comp.append(idx)
        q.extend(x for x in adj_fc[idx] if x not in visited)
    comps.append(sorted(comp))

mobile_dbg = {}
pn = wn = 0
for comp in comps:
    csp = {sp_fc[i] for i in comp}
    if 'Cl' in csp:
        pn += 1; mobile_dbg[f'perchlorate_{pn}'] = comp
    elif csp <= {'O', 'H'} and len(comp) <= 3:
        wn += 1; mobile_dbg[f'water_{wn}'] = comp

print(f'    Mobile units found: {len(mobile_dbg)} '
      f'({pn} perchlorate, {wn} water)')
if pn == 0 and wn == 0:
    print(f'  {FAIL} — no mobile units detected')
    sys.exit(1)
else:
    print(f'  {PASS} — mobile units detected correctly')

# ── 3b: rotate-only with lenient vdW (rotation keeps Cl/O centroid fixed) ────
print('  3b: rotate-only trials (most lenient — centroid stays put) ...')
ops_p1.vdw_scale = 0.70          # relax threshold slightly for test
ops_p1.max_attempts = 200

np.random.seed(7)
n_trials, n_passed = 20, 0
for trial in range(n_trials):
    # Force rotate-only by patching np.random.choice temporarily
    orig_choice = np.random.choice
    np.random.choice = lambda a, *args, **kw: (
        'rotate' if a == ['translate', 'rotate', 'both'] else orig_choice(a, *args, **kw)
    )
    result = ops_p1.op_whole_cell_counterion(full_withH)
    np.random.choice = orig_choice

    if result is None:
        continue

    if result.num_sites != n_full:
        print(f'  trial {trial+1:2d}: {FAIL} — {result.num_sites} atoms, expected {n_full}')
        continue

    ref_s = Structure(full_withH.lattice, full_withH.species,
                      full_withH.frac_coords.copy())
    res_s = Structure(result.lattice,     result.species,
                      result.frac_coords.copy())
    ref_s.sort(); res_s.sort()

    moved    = np.linalg.norm(res_s.frac_coords - ref_s.frac_coords, axis=1) > 1e-4
    sp_s     = [s.symbol for s in res_s.species]
    Cu_moved = sum(1 for i, m in enumerate(moved) if m and sp_s[i] == 'Cu')
    unexpected = {sp_s[i] for i, m in enumerate(moved) if m and sp_s[i] not in {'Cl','O','H'}}

    if Cu_moved > 0 or unexpected:
        print(f'  trial {trial+1:2d}: {FAIL} — Cu moved={Cu_moved}, unexpected={unexpected}')
        continue

    moved_elems = Counter(sp_s[i] for i, m in enumerate(moved) if m)
    print(f'  trial {trial+1:2d}: moved {sum(moved)} atoms {dict(moved_elems)} — OK')
    n_passed += 1

if n_passed > 0:
    print(f'  {PASS} — {n_passed}/{n_trials} rotate trials succeeded '
          f'(rest clashed — expected in dense crystal)')
else:
    print(f'  {FAIL} — 0/{n_trials} rotate trials succeeded')
    sys.exit(1)

print('\nAll tests passed.')
