# Fluxonium Project!! 

Hi welcome to my project!

***Project Structure***

- **`source`**: contains tools like the simulation object, datapipelining and some utilities!! :D
- **`ReadFromSmarterPeople`**: references and materials incoming to read and understand
- **`jobs_to_send`**: SLURM jobs to be sent to cluster
- **`data`**: contains the data outputs from simulations, potentialy model weights if needed
- **`model_config`**: contains the hyperparameters for the model
- **`params.json`**: contains the specific physics parameters of the situation (might need better wording :3)
- **`main`**: contains the actual production scripts to be used for analysis and answering the question
- **`playzone`**: contains learning, random silliness, and patchy code to be sent out :3


# Project Core


Goal: Optimize Fluxonium qubit initialization protocol!

In the protocol, a Fluxonium qubit exists in a mixed state! A current pulse and a variation of flux from $\pi$ away and back to $\pi$ achieves this and the question is on how to optimize these curves to achieve some fidelity goal (not locked for now) in minimal time! 

The relevant Hamiltonian equation is:

$H = 4E_c\hat{n}^2 + \frac{1}{2}E_L\hat{\phi}^2 - E_Jcos(\hat{\phi}
-\phi_t)$

Python project

Libraries: QuTip, Scqubits, scipy, more machine learning and number stuff


**Current vision** 
- Make a program which searches the space in some time $T_0$ to achieve some fidelity f, then have it update intelligently and use a smaller time $T_1$ iteratively until we've minimized T for the desired fidelity!

- Analyze key dimensions and patterns to develop physical intuition!


**Current Task**
- It's been a while since I've worked on the project
- Correcting physics assumptions currently!!





**Current Architectural Plan** 
- Conduct a search through $<{\phi}, {\Omega}>$ or what I'll call $X$ for some time $T_n$ until it hits target fidelity, then pipe the output data to an analyzer which will update to $T_{n+1}$, then retain info to get $X_0(T_{n+1})$ and iterate by feeding this in. Will try to use the global landscape and gradients too if it is not too expensive computationally though I do not know much :D



# Where to go? 

Indeed, see experiments_legit for a fully polished file to run. 
Particular files of note: source/simulator.py -> contains the code for instantiating all of the physics into code and actually making a simulator with given parameters! :D

