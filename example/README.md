This examples shows the whole process from creating a TESLA cavity and setting up the simulation parameters, manually running the simulation in CST, running Monte Carlo simulation in python, and finally statistical analysis on the raw results.

STEP-BY-STEP EXAMPLE RUN

1. SETUP CAVITY
Open src/cavity_setup.py.
Geometry parameters (MID_CUP, END_CUP_1, PIPE_CONFIG, CAVITY) and solver settings (SIMULATION_SETUP, MESH_SETTINGS, SOLVER_SETTINGS) can be modified here.
Set:
CST_PROJECT_FILE = r"C:Your_Path\example_tesla.cst"
CST_MODE = "new"
Run:
python setup_cavity.py
You can see the end result of this step in Photos\Screenshot1

2. SOLVE IN CST (MANUAL)
Open example/example_tesla.cst in CST.
Click Start Simulation in the Simulation section in the Home ribbon.
You can see the end result of this step in Photos\Screenshot2
Close the CST project tab.

3. MONTE CARLO EXTRACTION
In example/config_example.yaml make sure to insert the path to example_tesla.cst.
Run:
python run.py example/config_example.yaml --verbose
You can see the end result of this step in Photos\Screenshot3

4. STATISTICAL ANALYSIS
Run:
python analyze.py outputs/example_run/<run_folder>
You can see the end result of this step in Photos\Screenshot4 and the generated plots under outputs/example_run/<run_folder>.
Here the main pipeline ends, below are two additional tools that give you an idea of what else you can do with the generated data.

5. PATCH SIZE TEST
This tool measures the extent (in mm) of which the current concentration (defined by contineus region where J^2 is between 0 and -3db) spans.
Run:
python test_patch_size.py outputs/example_run/<run_folder>/extracted_fields.npz
You can see the end result of this step in Photos\Screenshot5

6. 3D THICKNESS MAPS
This tools generates plots of the generated thickness maps.
Add your output folder to INPUT_DIRS in generate_thickness_maps.py.
Run:
python generate_thickness_maps.py
You can see the end result of this step in Photos\Screenshot6 and the generated thickness_maps folder