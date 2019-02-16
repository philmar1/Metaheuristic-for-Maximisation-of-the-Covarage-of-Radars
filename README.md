# Metaheuristic-for-Maximisation-of-the-Covarage-of-Radars

This project uses a metaheuristic approach in order to maximize the coverage of sensors displayed in an area. 
A metaheuristic is a powerful optimization method used when the function to optimize is unknown. The programming language I used is python.

I developped an enhanced method of the annealing algorithm. For more information on the annealing algorithm, please refer to this link https://en.wikipedia.org/wiki/Simulated_annealing .

The codes I've written are : 
- create_graphs.py
- snp.py (from ---start here to ---end here) 
- codes/algo.y
- codes/bit.py

To run the algorithm, run the command : python3 snp.py -arguments  (The arguments may be found in the snp.py file. )
To compare the ERT, run the command : python3 create_graphs.py

ERT is a method to compare different metaheuristic algorithms. It gives information about the ability to reach a specific value (over a %of max or below %of min) over the itterations 

