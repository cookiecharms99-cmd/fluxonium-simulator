# Fluxonium Qubit Project

Goal: Optimize fluxonium qubit initialization protocol!

In the protocol, a fluxonium qubit exists in a mixed state! A current pulse and a variation of flux from $\pi$ away and back to $\pi$ achieves this and the question is on how to optimize these curves to achieve some fidelity goal (not locked for now) in minimal time! 

The relevant Hamiltonian equation is:

$H = 4E_c\hat{n}^2 + \frac{1}{2}E_L\hat{\phi}^2 - E_Jcos(\hat{\phi}
-\phi_t)$

Python project

Libraries: Qutip and other machine learning libraries



Current vision - Make a program which searches the space in some time T_0 to achieve some fidelity f, then have it update intelligently and use a smaller time T_1 iteratively until we've minimized T for the desired fidelity!


Checklist!! 

1.  Modularized simulator initialization and execution away from the model script! -yay done!

2. Modularized many tools involved with setup and configuration loading into their own files.

3. Successfully ran a simulation using the new structure!! 


4. Ok next up! Read up on how to store the data. We want to ideally be able to use information gathered from longer data trial runs in some high dimensional space in order to set our x0 for a new run. Also we want to store some results in a global data training file with tags on the run and final fidelity and maybe time to properly filter and perhaps cut out bad data easily!

5. Execution step: for now incorporate a way to train along drive. In the meantime think about step 4!


YAYY

New checklist! 


" Ok I investigated PINN's and stuff and I think i have some sort of architecture for the entire project moving
   forward. We have a searcher (be it CMAES, RL, bayesian) that maps out the landscape by executing the simulation.
   Once it reaches some target fidelity 1 - epsilon or something, an algorithmic outerloop auto iterates T down! In
   this step, some adapter model or algorithm takes in the data from before and adapts it to lower T somehow. And then
   we just keep this running hehe. :3"


   Hmm apparently optimizing model parameters is important too!! Idk why this didn't cross my mind!

   

   Revisiting project after a long time!! :D