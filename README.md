## Water Heater MPC

Code to accompany the following paper:

Elizabeth Buechler, Aaron Goldin, and Ram Rajagopal. "Designing model predictive control strategies for grid-interactive water heaters for load shifting applications" Applied Energy. (2025) [Paper here](https://authors.elsevier.com/a/1kMMg15eifC9Fx)


#### Running the code:

* Run `run_mpc.py` to test the MPC controllers. This requires the CasADi package which utilizes the IPOPT solver. This work was developed using CasADi version 3.6.3 and python version 3.11.
* Run `run_thermostat.py` to test the baseline thermostatic controller with no load shifting.

#### Folders:

* `data`: Contains electricity price profiles, parameters of the MPC control models, and parameters of the mult-node model. We use water draw data from Ritichie et al. which can be downloaded [here](https://scholardata.sun.ac.za/articles/software/Water_heater_dataset_Grid_and_user-level_software_and_dataset_/16669651?file=30869992). Download the data and move the `Raw Water Profiles` folder to the `data` folder in this repository.
* `functions`: Contains helper functions for the MPC optimization, thermostatic control, and multi-node model simulation.
* `results`: Create this folder. Simulation results from `run_mpc.py` and `run_thermostat.py` get saved here.
