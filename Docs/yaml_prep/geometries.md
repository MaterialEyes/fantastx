# Geometry-specific inputs

---

Currently, FANTASTX supports 5 different structural geometries. These are:

- Bulk ('bulk')

- Grain boundaries/ interfaces ('gb')

- Clusters ('cluster')

- Molecules ('molecule')

- Surfaces ('surface')

Each one of these structural geometries requires different inputs to the [FANTASTX input yaml file](index.md), and in the case of molecules, additional external files that need to be included in the FANTASTX job folder. These inputs govern the process by which FANTASTX randomly generates initial sets of candidate structures, as well as governing mating operations. Each geometry will be covered in turn.

## Bulk

FANTASTX supports bulk geometries with variable unit cells. When randomly assembling the initial set of bulk structures, FANTASTX utilizes an iterative approach. In detail, this approach is as follows:

1. FANTASTX creates a lattice from the user provided lattice constants and lattice angles. The user provided values are taken to be the center of normal distributions, with default standard deviations of 0.05, and the constants and angles of the generated lattice are randomly drawn from these distributions.
2. FANTASTX chooses a tiling pattern for the unit cell. By default, the tiling patterns are (2, 2, 1) and (1, 1, 1). If a tiling pattern other than (1, 1, 1) is used, then FANTASTX will configure the smaller unit cell instead of the supercell, ultimately tiling the supercell with the smaller unit cell.
3. FANTASTX chooses a target atomic composition. If a tiling pattern other than (1, 1, 1) is used, then the atomic composition of the smaller unit cell is the smallest possible integer composition which minimally satisifies the supercell composition.
4. FANTASTX iteratively adds atoms to the unit cell as bonded neighbors of pre-existing atoms, adhering to the max bonds constraints and (periodic) distance constraints provided elsewhere in the FANTASTX YAML file. Atoms are preferentially added as far away from currently bonded atoms as possible. Bonding atoms to atoms of the same species can be toggled on or off if multiple atomic species are present. 

### Input Keywords

The user currently has control over three parameters. These parameters, as well as their keywords in the corresponding section of the FANTASTX YAML file, are:

- Lattice constants (`box_abc`): the centers of the normal distributions (with $\sigma$=0.05) governing the a, b and c lattice constants.

- Lattice angles (`box_angles`): the centers of the normal distributions (with $\sigma$=0.05) governing the $\alpha$, $\beta$ and $\gamma$ lattice angles.

- Self-bonding during random assembly (`allow_random_model_self_bonding`): if `True`, then atoms of the same species can be attached during randomly assembly. If `False`, then this is prevented unless only one species is present.

### FANTASTX YAML Input

Bulk structure parameters are provided in the FANTASTX YAML file under the keyword `bulk`. An example of the YAML section is:

!!! example
    ```YAML
    bulk:
        box_abc: [5, 5, 5]
        box_angles: [90, 90, 90]
        allow_random_model_self_bonding: False
    ```

## Grain boundaries/ interfaces

FANTASTX supports the treatment of grain boundaries and interfaces with fixed grains, such that only the composition and atomic positions of the interface are allowed to vary. This can be viewed as an 'interface' region that FANTASTX manipulates implanted within a fixed atomic environment.

The FANTASTX methodology to randomly constructing grain boundaries/ interfaces for use in the initial population is quite simple. FANTASTX merges the two outer grains together, moving them each into the interface by a random distance. Atoms are appended to the new 'interface' structure one-by-one, eliminating atoms which would overlap with a previously added atom. The process is repeated as needed until the resulting interface composition lies within the user-specified bounds.

### Input Keywords

Input keywords, which must be added to the appropriate section of the FANTASTX YAML file, primarily describe the geometry and relaxability of the interface. These include:

- `init_gb_astr`: the path to the POSCAR file which describes the structure of the grains surrounding the boundary/interface. 
- `iface_thickness`: the thickness in Å of the grain boundary interface. Defaults to 10 Å.
- `iface_z_mid`: the mid-point of the interface in lattice units (fractional coordinates). Defaults to 0.5. 
- `sd_true_above`: (float) fractional z-coordinate above which selective dynamics will be True. Defaults to the fractional z-coordinate of the top of the bottom grain.
- `sd_true_below`: (float) fractional z-coordinate below which selective dynamics will be True. Defaults to the fractional z-coordinate of the bottom of the top grain.

Additionally, the keyword `num_slices` with an integer value can be added to the same section of the FANTASTX YAML file. This will determine how many slices are made when performing cut-and-splice mating of grain boundaries and interfaces. Defaults to 2.

### FANTASTX YAML Input

Grain boundary/ interface parameters are provided in the FANTASTX YAML file under the keyword `gb`. An example of the YAML section is:

!!! example
    ```YAML
    gb:
        init_gb_astr: # full path to the initial grain boundary structure
        iface_thickness: 2
        iface_z_mid: 0.475
        num_slices: 2
        sd_true_above: 0.3
        sd_true_below: 0.7
    ```

## Clusters

In FANTASTX, clusters are considered to be groups of atoms surrounded by vacuum. When FANTASTX creates random clusters for the initial population, the procedure is very similar to that of [bulk structures](#bulk), albeit keeping atoms within a fixed spherical region but not worrying about periodic distances. This procedure can be summarized as follows:

1. FANTASTX chooses a random target atomic composition within the user-specified bounds.
2. FANTASTX iteratively adds atoms to an empty unit cell as bonded neighbors of pre-existing atoms, adhering to the max bonds constraints and (periodic) distance constraints provided elsewhere in the FANTASTX YAML file. Atoms are preferentially added as far away from currently bonded atoms as possible, but are not allowed to be placed outside of a spherical region with fixed diameter. Bonding atoms to atoms of the same species can be toggled on or off if multiple atomic species are present. 
3. Distance constraints are checked, ensuring both that no atoms overlap and also that no atoms are isolated. If checks pass, then the cluster is moved to the cluster-center location within the user-specified unit cell.

### Input Keywords

Input keywords solely govern the nature of the randomly generated clusters, including the size of the unit cell and the diameter of the cluster. These keywords are:

- `box_abc`: lattice constants of the unit cell the cluster will ultimately be simulated in. Default value is [20, 20, 20] (a 20x20x20Å unit cell).
- `max_dia`: the maximum diameter of the cluster in Å. Default value is 8 Å. 
- `origin`: the center of the cluster in Å. Default is [a/2, b/2, c/2] where a, b and c are the lattice constants.
- `allow_random_model_self_bonding`: if True, allows atoms of the same species to bond during random assembly. If False, then this is prevented unless only a single atomic species is present. Defaults to False. 

### FANTASTX YAML Input

Cluster parameters are provided in the FANTASTX YAML file under the keyword `cluster`. An example of the YAML section is:

!!! example
    ```YAML
    cluster:
        box_abc: [20, 20, 20]
        max_dia: 8
        origin: [10, 10, 10]
        allow_random_model_self_bonding: True
    ```

## Molecules

FANTASTX treats molecules differently than other geometries. Rather than viewing the atomic configurations through the lens of single atoms, the atomic configurations are considered to be assembled from molecular fragments such as ligands or other small groups of atoms. This makes the random assembly procedure and genetic operations distinct from those of other structural geometries.

In particular, FANTASTX requires additional files to be provided which describe the atomic and electronic structure of the fragments which a molecule can be assembled from. Detailed [down below](#additional-molecule-files), these include POSCARS for each fragment, a YAML file detailing the nature of each fragment, and a bond length JSON file (this latter file is provided with FANTASTX). 

The FANTASTX random assembly procedure for molecules is quite similar to those of [bulk](#bulk) and [cluster](#clusters) geometries. It occurs iteratively, starting with a central fragment and adding additional fragments until the randomly chosen fragment composition is reached. This procedure can be summarized as follows:

1. Randomly draws the set of fragments that will comprise the molecule. The central fragment is always the first fragment (fragment 0) in the fragment YAML file, all other fragments are assigned probabilities of selection based on their frequency within the fragment "bath".
2. Creates a cubic unit cell that will house the molecule. This should include sufficient vacuum. The geometric center of the starting fragment is placed at the center.
3. Iteratively append fragments from the chosen set of fragments. Each fragment can be a single atom or set of atoms. Every fragment has a list of possible attachment sites, both where it can be attached and where subsequent fragments can be attached to it. Additionally, every attachment site has a maximum number of possible attachments which, once exhausted, means that no more fragments can be attached to that site.
4. When attaching fragments, first hydrogenate the fragment if hydrogenation probabilities are given for the molecule. These probabilities are given site-by-site.
5. Once the fragment has been hydrogenated, oxidize the fragment. Every fragment is assigned either a set oxidation state or a range of possible oxidation states. If a range of oxidation states, then the oxidation state is randomly assigned, otherwise the set oxidation state is assigned.
6. Choose the attachment site to the molecule, as well as the site on the fragment at which it will be attached, at random. The probability of choosing between the available attachment sites is given by the number of possible attachments still available for each site.
7. Draw the bond length of attachment from the provided JSON file if possible. 
8. Rotate the fragment around its axis of rotation and around the attachment site, until it is maximally distant from fragments already attached to the attachment site, while remaining within the expected angular distance given by the number of possible attachments for the site. For instance, if 6 bonds are expected, the fragment will attempt to be approximately 90° away from other fragments, within a set number of attachment attempts.
9. If the fragment satisfies distance constraints, attach it to the molecule. Otherwise keep attempting until a set number of attempts is exhausted. At that point, start over.
10. Attach counter ions if provided. It is possible for the user to provide a possible counter ion with corresponding oxidation state. Given the oxidation state of the counter ion, these will be appended to render the unit cell charge neutral. Currently only up to 4 counter ions can be appended.

### Additional Molecule Files

In addition to the inclusion of molecule-specific keywords in the FANTASTX YAML file, additional files must be provided. These are:

- POSCARs for each molecular fragment.
- A YAML file with all of the information FANTASTX needs for each fragment.

The POSCARs should all be placed in a single directory, the path to which is provided to FANTASTX in the molecule section of the FANTASTX YAML with the keyword `fragments_directory`.

The fragment YAML should be composed with each fragment taking a numbered entry increasing from 0. The central fragment should be assigned the key 0, with the other fragments (possibly including duplicates of the central fragment) taking the additional keys. The fragments YAML path should be provided to FANTASTX in the molecule section of the FANTASTX YAML with the keyword `fragments_yaml`. 

Each fragment entry can have the following keywords:

- `name`: the name of the fragment or atom. Make sure to use the correct atomic symbol for atoms.
- `type`: 'atom' or 'ligand'
- `poscar`: if the type is 'ligand', then the POSCAR is used to provide the atomic structure and composition of the ligand.
- `geometric_center_coords`: used to assign the geometric center of the fragment. Only relevant for ligands. Used to determine the axis for fragment attachment (drawn between the attachment site and the geometric center).
- `attachment_sites`: list of atomic sites where the fragment can attach to the molecule or be attached to the molecule. If a single atom, then `attachment_sites` should be [0].
- `available_attachments`: number of available attachments for each attachment site in `attachment_sites`. For instance, if `attachment_sites` is [0, 3], then `available_attachments`=[1,2] would indicate that atomic site 0 can be attached to 1 time, and atomic site 3 can be attached to two times.
- `H_site_addition_probs`: a list of the hydrogenation probabilities for each site. When hydrogenating a fragment, if a random number between 0 and 1 is drawn that is below this number, then a hydrogen atom will be attached to the site. This is helpful for systems where hydrogenation is necessary for stability and charge conservation. 
- `count`: integer frequency of this fragment in the bath. Should be set relative to the other fragment counts to tune the balance of the fragment distribution. For instance, in a two fragment system setting one fragment to have a count of 1 and the other to have a count of 2 means that fragment 1 will be selected 1/3 of the time, and fragment 2 will be selected 2/3 of the time. 
- `oxidation_states`: Oxidation state for the atom or atoms within the fragment. Should be in the form of a list. Entries themselves can be lists, in which case the oxidation state for that site is randomly chosen from the list values. 

An example of a fragment YAML file is below. In this example, two types of fragments can be seen: 'atom' and 'ligand'. Three total fragments are present in the system.

??? example "Example fragments.yaml file"
    ```YAML
    0:
        name: "Fe"
        type: "atom"
        poscar: null
        attachment_sites: [0]
        available_attachments: [6]
        geometric_center_coords: null
        H_site_addition_probs: [0]
        count: 1
        oxidation_states: [[2, 3]] # lower and upper bound
    1:
        name: "CN"
        type: "ligand"
        poscar: "CN_POSCAR"
        attachment_sites: [0]
        available_attachments: [1]
        geometric_center_coords: [0, 0, 0.58]
        H_site_addition_probs: [0, 0]
        count: 8 # number of fragments floating around in the "bath"
        oxidation_states: [2, -3]
    2:
        name: "H2O"
        type: "ligand"
        poscar: "H2O_POSCAR"
        attachment_sites: [0]
        available_attachments: [1]
        geometric_center_coords: [0, 0, 0.277]
        H_site_addition_probs: [0, 0, 0]
        count: 1
        oxidation_states: [-2, 1, 1]
    ```

### Input Keywords

Additional keywords govern the size of the molecule unit cell, the possible presence of counter ions, and the number of fragments which will be attached to the central fragment. Each keyword should be placed in the 'molecule' section of the FANTASTX YAML file. 

The size of the molecule unit cell is given by the `max_dia` keyword. Molecules are always placed in cubic unit cells, this keyword gives the lattice constant of the unit cell in units of Å. 

Counter ions are provided with the `counter_ions` keyword. They should be provided as a two item list, where the first item in the list is the atomic species of the counter ion, and the second item is the oxidation state of the counter ion. 

Finally, the number of fragments which will be attached should be given with the `number_of_fragments` keyword. This can be an integer value, or a two item integer list. In the case of a two item list, the list corresponds to [low, high] where high is exclusive. 

### FANTASTX YAML Input

Molecule parameters are provided in the FANTASTX YAML file under the keyword `molecule`. An example of the YAML section is:

!!! example
    ```YAML
    molecule:
        max_dia: 8
        counter_ions: ["K", 1]
        fragments_yaml: /PATH/TO/FRAGMENTS/YAML
        fragments_directory: /PATH/TO/FRAGMENTS/POSCAR/DIRECTORY
        number_of_fragments: [5, 7]
    ```

## Surfaces

### Input Keywords

### FANTASTX YAML Input

```YAML
surface:
    init_slabs_dir: # the path to the folder containing the initial surface slabs.
    surface_thickness: 1 # in Å
    substrate_thickness: 10 # in Å
    separation: 2 # separation in Å between the substrate top z and the surface bottom z
    constrain_z: True # boolean determining if the z coordinates of the surface should be kept fixed
    sd_cut_off: 10 # how far to freeze the substrate
    sd_no_z: False
    composition: ["Fe": 10, "Al": 2] # composition of the surface layer in a format parseable by pymatgen.
    num_slices: 2 # how many slices to make when mating structures by slicing
    hop_mate_frac: 0.3 # ratio between basinhopping and mating by slicing
```