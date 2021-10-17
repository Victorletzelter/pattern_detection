# patterndetection

This project consists on a technical study on the Stellarator TJ-II, an experimental device allowing to carry out experiments of turbulent plasmas at high temperature confined by magnetic fields. More precisely, an algorithm for the automatic detection of patterns in the time series of the electrostatic potential measured in the plasma was constructed. For this purpose, a probabilistic model was designed to simulate additional data to those provided and to train a neural network. These patterns correspond to specific events within the reactor: the "Zonal Flows"; this detection is relevant insofar as the understanding and detection of these hazards allows the improvement of the magnetic confinement of the plasma.

![Stellarator TJ-II](https://upload.wikimedia.org/wikipedia/commons/d/d8/TJ-II_model_including_plasma%2C_coils_and_vacuum_vessel.jpg)
*Stellarator TJ-II*

To figure out the stakes of the internship, and the detail of the process to achieve the results of the internship, the report Ciemat__internship-VF.pdf from the Report folder can be read. 

These project has been divided in several parts, which correpond to different python files in the 'Python_project" folder. 

1.  At first, the corresponding patterns in the time series were labeled by hand, given the time series of two probes (See the report and the Hand_labeling.py file for the process).
2.  Then, the results of this hand-labeling allowed to perform statistics, and to tune an algorithm that perform the labeling of the events (given the data of the two probes again) in an automatic way (See Event_labeling.py). The python files Exp.py, Coherence.py, Stats.py and Non-coverage.py are directly associated to this algorithm. 
3.  The results of this new algorithm permit to tune the parameters of a probabilistic model, that will generate simulated labeled data as closely as possible to the real data (See Data_generator.py).
4.  After that, a neural network was designed. Its parameters were tuned in a random grid to map, the best as possible, the ZF events when the simulated data is given. The final architecture of the network, and the related functions that create inputs and outputs are written in the NN_Model.py file.   
5.  Finally, the scripts to reproduce the plots given in the report are given in the Report_plots.py file. 




