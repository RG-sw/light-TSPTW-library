from abc import ABC,abstractmethod
import numpy as np
from copy import deepcopy
import ORtools

class templateAlgo:

    def __init__(self):
        self.distances = []
        self.openings = []
        self.results = []

    def getData(self, path):
        #path = "ist_rand.txt"
        lines = np.array(open(path,'r').read().splitlines()) #lines = f.splitlines() # è una lista
        num_nodes = int(lines[0])

        distances_temp = lines[1:num_nodes+1]
        openings_temp = lines[num_nodes+1:]

        distances = []
        openings = []

        for i in range(num_nodes):
            distances.append( list(map( float, distances_temp[i].split())) )
            openings.append( float(openings_temp[i] ) )     
        
        self.distances = deepcopy(distances)
        self.openings = deepcopy(openings)
        self.num_nodes = num_nodes

    @abstractmethod
    def Greedy(self):
        pass

    @abstractmethod
    def LSorMeta(self):
        pass
    
    def runAlgo(self, data_path):
        self.getData(data_path)
        self.Greedy()
        self.LSorMeta()
        return self.results

class SimulatedAnnealing(templateAlgo):
    def Greedy(self):
        self.results = deepcopy(ORtools.Nearest_Neighbor(self.distances, self.openings, self.num_nodes))
    def LSorMeta(self):
        self.results = deepcopy(ORtools.Simulated_Annealing(self.results,self.distances,self.openings,self.num_nodes))

class TabuSearch(templateAlgo):
    def Greedy(self):
        self.results = deepcopy(ORtools.Nearest_Neighbor(self.distances, self.openings, self.num_nodes))
    def LSorMeta(self):
        self.results = deepcopy(ORtools.Tabu_Search(self.results,self.distances,self.openings,self.num_nodes))

class VND(templateAlgo):
    def Greedy(self):
        self.results = deepcopy(ORtools.Nearest_Neighbor(self.distances, self.openings, self.num_nodes))
    def LSorMeta(self):
        self.results = deepcopy(ORtools.VND(self.results,self.distances,self.openings,self.num_nodes))   

class GeneralVNS(templateAlgo):
    def Greedy(self):
        self.results = deepcopy(ORtools.Nearest_Neighbor(self.distances, self.openings, self.num_nodes))
    def LSorMeta(self):
        self.results = deepcopy(ORtools.General_VNS(self.results,self.distances,self.openings,self.num_nodes))     
        
class StokVNS(templateAlgo):
    def Greedy(self):
        self.results = deepcopy(ORtools.cheapest_insertion(self.distances, self.openings, self.num_nodes))
    def LSorMeta(self):
        self.results = deepcopy(ORtools.Stok_VNS(self.results,self.distances,self.openings,self.num_nodes))      

