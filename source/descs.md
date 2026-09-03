# Simulator

**Notable Libraries: scipy.sparse, scqubits, Qutip (Qobj, mesovle), math**

*Contains the constructor for initalizing the fluxonium simulation object and execution of the simulation! Also contains bases definitions. Mostly untouched and polished, but future implementation may include basis as an argument and need revisitation*


## Functions and Objects

***Basis builder***: **build_sin_basis**(t: array, T, n_coeffs) -> builds a sine basis 
$sin(\frac{\pi k}{T} t)$ , may make a new basis! :3

**FluxoniumSimulator** object: takes params as an arg and builds a fluxonium simulator for this! :3 Has methods for execution. Supports updating of time without reinstantiation. :3





# Utils

**Notable Libraries: pandas, numpy, scipy.fft**

*Contains many helper utilities to encapsulate busywork and data transfer. Some libraries embedded in functions*

## Functions

**get_config**(model_name, root_dir) -> fetches the model configuration

**get_parser**() -> generates parser object to tweak parameters like this "python --arg1 val1 --arg2 val2" 

**update_config_with_args(config, args)** -> updates the input configuration dictionary with the args given by the parser

**get_initial_guess**(config, best_path = None) -> generates the intial guess for a new run by using the current best or mathematically generating a piecewise function or other

**log_data_to_global**(run_name, model_name, params, weights, fidelity, file_path) -> method to upload the new run data with appropriate tags to a global file! Helper function to do the job for the model. Nice abstraction. Tweakable later

# Pulse Resampler

*Converts weights between different times so that the information is kept and we don't start from 0!*



