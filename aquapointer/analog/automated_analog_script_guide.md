# Using the automated script with 3D-RISM input and settings text file
The script `./aquapointer/analog/automated_flow.py` enables execution of the analog workflow from the CLI.
The user must specify a 3D-RISM file and a settings file as input to `rism_to_locations()` in  `./aquapointer/analog/automated_flow.py`.
For example, (as in  `./aquapointer/analog/automated_flow.py`):

```python
    main_folder = "data"
    dna_folder = f"{main_folder}/DNA"

    def rism_file(path):
        return f"{path}/prot_3drism.O.1.dx"
    

    locations = rism_to_locations(rism_file(dna_folder), "aquapointer/analog/analog_settings_example")
```

## 3D-RISM
The 3D-RISM file must be in `dx` format to be read by the slicer.
There are some file path "shortcuts" already implemented in the script, but it is not required to use them. 

## Settings file
Specify the workflow parameters, i.e. pulse settings, lattice parameters, Gaussian parameters, density cropping and filtering settings, and density slicing planes.
Each line of the file contains settings for a particular category and must be specified in the order listed below. 
The parameters on each line are separated by spaces, as shown in `analog_settings_example.txt`.
For now, the script strictly requires the following parameters in the following order, unless indicated as "optional".
1. Pulse settings:
    - blockade radius
    - frequency `omega` 
    - pulse duration
    - max detuning
2. Gaussian parameters:
    - amplitude 
    - variance
3. Lattice parameters:
    - lattice type (poisson-disk, rectangular, triangular, hexagonal)
    - num_x (rectangular) or n_rows (triangular or hexagonal) or 0 (placeholder for poisson-disk)
    - num_y (rectangular) or n_columns (triangular or hexagonal) or 0 (placeholder for poisson-disk)
    - minimum exclusion radius
    - maximum exclusion radius
    - size (number of points in lattice after decimation, Optional)
4. Density filtering (Optional):
    - "filter": keyword indicating the settings on this line are for filtering, since filtering is optional
    - filter function: only Gaussian-Laplacian supported in current version of script
    - sigma standard deviation of the Gaussian filter
5. Cropping (Optional):
    - "crop": keyword indicating the settings on this line are for cropping, since croppping is optional
    - x_coordinate of center of cropped slice
    - y_coordinate of center of cropped slice
    - length of cropped slice in x
    - length of cropped slice in y
6. Density planes (minimum 1 slicing plane required):
    - Plane 1 (3 points): x1 y1 z1 x2 y2 z2 x3 y3 z3
    - Plane 2 (Optional)
    - Plane ... (Optional)
