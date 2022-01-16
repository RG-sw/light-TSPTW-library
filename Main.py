from ORalgos import TabuSearch, SimulatedAnnealing, GeneralVNS
from ORtools import instance_generator

if __name__ == "__main__":
    #instance_generator creates a file named "ist_rand.txt"
    # Algorithms have to know this file path  
    instance_generator(10)
    ts = TabuSearch()
    result_ts = ts.runAlgo("ist_rand.txt")

    instance_generator(20)
    sa = SimulatedAnnealing()
    result_sa = sa.runAlgo("ist_rand.txt")

    instance_generator(30)
    gvns = GeneralVNS()
    result_gvns = gvns.runAlgo("ist_rand.txt")
