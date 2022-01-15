# light-TSPTW-library
OR Library for relaxed version of TSPTW problem

## Problem Description :
Given 
- a set of N stores (nodes)
- the opening time of each store
- the distance beetween them

Temporal Constraints
- if you arrive to a node before the opening -> you have to wait until the opening
- if you arrive after the opening -> you have to pay a penalty equal to (Arrival_Time - Opening_Time) 

Goal : find the shortest route for visiting every store and return to the starting point, considering the penalties.

## Heuristic Algorithms
- Greedy Nearest Neighbor
- Cheapest Insertion

## Local Searches
- 2 OPT
- 2 Node Swap
- VND (Variable Neighborhood Descent)

## Meta Heuristics
- Tabu Search
- Simulated Annealing
- VNS + Stock VNS (Variable Neighborhood Search)

## Map Visualitazion with MapBox
![image](https://user-images.githubusercontent.com/94836571/149630598-97ebc804-f3b9-45d4-a61f-cbeb17e50153.png)

