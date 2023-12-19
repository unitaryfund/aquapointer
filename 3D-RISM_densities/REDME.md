# Testing 3D-RISM

3D-RISM results are compared with crystallographic water molecues, for selected protein-ligand systems where water is found inside binding pockets.
 

This is required, since the water placement algorithm capacity strongly dependents on the information carried by the 3D-RISM density map. In this repository we store the structures of those systems, with the relative 3D-RISM densities. The code used to 

For instance, we performed 3D-RISM simulations for the following proteins:

- The first human bromodomain of human BRD4 (bromoD)

- The HIV-1 protease (HIV1)

- Influenza virus neuraminidase subtype N9 (1NNC, 1.80 Ang resolution)

- The Scytalone Dehydratase (dehydratase, 1.65 Ang resolution)

- A protein taken from the WatSite paper examples.


For each system a 3D-RISM output in .dx format is provided. This file can be used as starting point to generate density slices, to be used for water sites prediction using the VQA or QAE algorithm on neutral atoms QPU. Such file stores a 3D-RISM oxygen density. The structure of the present folder is as follow:

```

|--SYSTEM_NAME
|----*.dx: 
|----*.pdb: 
|----with_ligand: additional folder, present if both with and without ligand molecule in a complex with the protein are available.

...

```
 
