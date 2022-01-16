import numpy as np
import math
import random
from numpy import cos, pi, sin
import networkx as nx
import matplotlib.pyplot as plt
import itertools

def instance_generator(num_nodi):
    distanze = np.zeros(shape=(num_nodi,num_nodi))
    for i in range(num_nodi):
        for j in range(num_nodi):
            if i == j:
                distanze[i][j] = 0
            elif i < j:
                distanze[i][j] = random.randint(1,1000)
            else:
                distanze[i][j] = distanze[j][i]

    istanza = open("ist_rand.txt", 'w')
    istanza.close()
    istanza = open("ist_rand.txt", 'a')
    istanza.write(str(num_nodi))
    for i in range(num_nodi):
        istanza.write("\n")
        for j in range(num_nodi):
                istanza.write(f"{distanze[i][j]} ")
    istanza.write("\n0\n")
    for i in range(1,num_nodi):
        apertura = random.randint(1,2000)
        istanza.write(f"{apertura}\n")
    istanza.close()

def generatore_istanza_mappe(num_nodi, dist_matrix):
    distanze = dist_matrix

    istanza = open("ist_rand.txt", 'w')
    istanza.close()
    istanza = open("ist_rand.txt", 'a')
    istanza.write(str(num_nodi))
    for i in range(num_nodi):
        istanza.write("\n")
        for j in range(num_nodi):
                istanza.write(f"{distanze[i][j]} ")
    istanza.write("\n0\n")
    for i in range(1,num_nodi):
        apertura = random.randint(1,2000)
        istanza.write(f"{apertura}\n")
    istanza.close()

def get_instance():

    path = "ist_rand.txt"
    lines = np.array(open(path,'r').read().splitlines()) #lines = f.splitlines() # è una lista
    num_nodes = int(lines[0])

    distances_temp = lines[1:num_nodes+1]
    openings_temp = lines[num_nodes+1:]

    distances = []
    openings = []

    for i in range(num_nodes):
        distances.append( list(map( float, distances_temp[i].split())) )
        openings.append( float(openings_temp[i] ) )     
    return distances, openings, num_nodes

def generate_rand_sol(sol):
    next_sol = sol[1:-1]
    a, b= random.sample(next_sol,2)
    indexA = next_sol.index(a)
    indexB = next_sol.index(b)

    next_sol[indexA] ,next_sol[indexB] = b, a

    next_sol.insert(0,0) #Con queste due istruzioni attacco nuovamente il nodo inziale e quello finale alla soluzione
    next_sol.append(0)
    
    return next_sol

def costo(sol,distanze,apertura,num_nodi):
    costo = 0
    for i in range(num_nodi):
        costo += distanze[sol[i]][sol[i+1]] #Aumento il costo in base alla distanza fra i nodi
        if (costo != apertura[sol[i+1]]) and (sol[i+1]!=sol[0]): #Se arrivo in anticipo o in ritardo pago una penalità
            costo += abs(costo - apertura[sol[i+1]])
    return costo

def costo_Cheapest_Insertion(sol):
    distanze, apertura, num_nodi = get_instance() #Recupero dati da file
    costo = 0
    for i in range(len(sol)-1):
        costo += distanze[sol[i]][sol[i+1]] #Aumento il costo in base alla distanza fra i nodi
        if (costo != apertura[sol[i+1]]) and (sol[i+1]!=sol[0]): #Se arrivo in anticipo o in ritardo pago una penalità
            costo += abs(costo - apertura[sol[i+1]])
    return costo  


def cheapest_insertion(dist_matrix, openings, num_nodes):
    
    visited = [0]

    # inizio prendendo il nodo + vicino alla base e ci attacco lo zero alla fine
    best_dist, best_index = best_neighbor(0, dist_matrix, num_nodes, openings, 0, visited)
    visited.append(best_index)
    visited.append(0)

    # nodi ancora da aggiungere
    nodes = list(set(range(num_nodes)) - set(visited))
    
    for i in range(len(nodes)):
        min_cost = float('inf')
        best_tour = visited[:] 
        # Ad ogni iterazione prendo il nodo nella posizione che minimizza incremento di F.O.
        for node in nodes:
            min_costTMP = float('inf')
            best_tourTMP = best_tour[:]#visited[:] 
            
            #Valuto dove posizionare il node
            for index in range(1,len(visited)):
                solTMP = visited[:]
                solTMP.insert(index,node)
                costoTMP = costo_Cheapest_Insertion(solTMP)
                if(costoTMP < min_costTMP):
                    min_costTMP = costoTMP
                    best_tourTMP = solTMP[:] 
            
            if( min_cost > costo_Cheapest_Insertion(best_tourTMP) ):
                min_cost = costo_Cheapest_Insertion(best_tourTMP)
                best_tour = best_tourTMP[:]
                best_node = node
            
        visited = best_tour[:]
        nodes.pop( nodes.index(best_node)) 
        #print("\nStep : ",visited, " Costo : ", costo_Cheapest_Insertion(visited))
        
    return visited

# FUNZIONE BEST+IND della Greedy
def best_neighbor(start_node, dist_matrix, num_nodes, times, current_time, visited):

    #al posto di num nodes posso calcolare la lunghezza della matrice così ho meno parametri
    best_distance = float('inf')
    dist_from_node = dist_matrix[start_node]

    # Aggiorno le distanze dal Nodo Corrente, considerando i tempi di apertura
    for i in range(num_nodes):
        if i!=start_node :
            # Se sono in anticipo quando arrivo...
            if current_time + dist_from_node[i] <= times[i] :
                dist_from_node[i] +=  times[i] - (current_time + dist_from_node[i])
            #Se sono in ritardo quando arrivo...
            elif current_time + dist_from_node[i]> times[i] :
                dist_from_node[i] += current_time + dist_from_node[i]- times[i] 

    # Calcolo il nodo migliore, che non sia già stato visitato
    for i in range(num_nodes):
        if i!=start_node and dist_from_node[i] < best_distance and not(i in visited):
            best_distance = dist_from_node[i]
            best_index = i

    return best_distance, best_index

def Nearest_Neighbor(dist_matrix,openings,num_nodes):
    visited = [0]
    current_time = 0
    current_node = 0


    # Calcolo con una Greedy il tour partendo dalla Base
    for i in range(num_nodes-1):
        best_dist, best_index = best_neighbor(current_node, dist_matrix, num_nodes, openings, current_time, visited)

        current_node = best_index
        visited.append(best_index)
        current_time += best_dist

    # Alla fine del Tour torno alla Base
    visited.append(0)
    current_time += dist_matrix[current_node][0]

    print(f"Greedy_NN    : {visited} -> Durata: {current_time}")
    return visited

def Simulated_Annealing(sol,distanze,apertura,num_nodi):
    TF = 2 #Temperatura frozen
    cooling_ratio = 0.9 #Fattore di diminuzione temperatura
    current_sol = sol #Soluzione corrente
    best = current_sol #Best è la miglior soluzione trovata
    durata = costo(best,distanze,apertura,num_nodi)
    T = 0.8*durata #Temperatura corrente
    while T>TF:
        count = 0
        while count < 10: #Finchè per tot tentativi di seguito non trovo sol migliori non abbasso la temperatura
            next = generate_rand_sol(current_sol)
            delta_E = costo(next,distanze,apertura,num_nodi) - costo(current_sol,distanze,apertura,num_nodi)
            if delta_E < 0:
                count = 0
                current_sol = next
                if costo(next,distanze,apertura,num_nodi) < costo(best,distanze,apertura,num_nodi):
                    best = next
                    durata = costo(best,distanze,apertura,num_nodi)
            else:
                r = random.random()
                count += 1
                if r < math.e ** (-delta_E/T):
                    current_sol = next
        T *= cooling_ratio
    print(f"Sim_Annealing: {best} -> Durata: {durata}")
    return best

def best_sol(sol,tabu_list,distanze,apertura,num_nodi):
    next_sol = sol[1:-1]
    best = np.inf
    best_sol = sol
    best_ind_a, best_ind_b = -1, -1
    for i in range(len(next_sol)):
        for j in range(i+1,len(next_sol)):
            temp_sol = sol[1:-1]
            temp_sol[i], temp_sol[j] = temp_sol[j], temp_sol[i]
            temp_sol.insert(0,0) #Con queste due istruzioni attacco nuovamente il nodo inziale e quello finale alla soluzione
            temp_sol.append(0)
            index_a, index_b = i, j
            if (costo(temp_sol,distanze,apertura,num_nodi) < best) and ([index_a, index_b] not in tabu_list):
                best = costo(temp_sol,distanze,apertura,num_nodi)
                best_sol = temp_sol
                best_ind_a, best_ind_b = index_a, index_b 
    return best_sol, best_ind_a, best_ind_b
            
def Tabu_Search(cur_sol,distanze,apertura,num_nodi):
    tabu_list = []
    count = 0
    contatore = 0
    best_solution = cur_sol
    while(contatore<10):
        next_sol,i,j = best_sol(cur_sol,tabu_list,distanze,apertura,num_nodi)
        if costo(next_sol,distanze,apertura,num_nodi) >= costo(best_solution,distanze,apertura,num_nodi):
            contatore += 1
        else:
            contatore = 0
            best_solution = next_sol
        cur_sol = next_sol
        tabu_list.append([i,j])
        if(count>1):
            tabu_list.pop(0)
        count += 1
    durata = costo(best_solution,distanze,apertura,num_nodi)
    print(f"Tabu_Search  : {best_solution} -> Durata: {durata}")
    return best_solution

def two_OPT(route):

    num_archi = len(route)-1
    # Enumero gli archi da [1...N] compresi
    while True:
        a, b= random.sample(range(1,num_archi+1),2)

        if( abs(a - b)%(num_archi-1) > 1 ):
            break

    new_route = route[:]
        
    if a>b:
        new_route[b:a] = route[a - 1:b - 1:-1]
    else:
        new_route[a:b] = route[b - 1:a - 1:-1]

    #print(a,b)
    return new_route


def best_in_two_OPT(route,distanze,apertura,num_nodi):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)):
                    if j-i == 1: continue # changes nothing, skip then
                    new_route = route[:]
                    new_route[i:j] = route[j-1:i-1:-1] # this is the 2woptSwap
                    if costo(new_route,distanze,apertura,num_nodi) < costo(best,distanze,apertura,num_nodi):
                        best = new_route
                        improved = True
        route = best
    return best


def two_node_swap(sol):

    next_sol = sol[:]
    a, b= random.sample(range(1,len(sol)-1),2)
    next_sol[a] ,next_sol[b] = next_sol[b], next_sol[a]
    
    return next_sol  

def best_in_two_node_swap(solution,distanze,apertura,num_nodi):
    
    tmp_disposition_list = list( itertools.permutations(range(1,len(solution)-1),2) )

    sol_disposition_list = []
    cost_list = []
    
    for e in tmp_disposition_list:
        sol_tmp = solution[:]
        sol_tmp[e[0]] , sol_tmp[e[1]]= sol_tmp[e[1]] , sol_tmp[e[0]]
        sol_disposition_list.append( sol_tmp )
        cost_list.append( costo(sol_tmp,distanze,apertura,num_nodi) )

    shortest_tour = min(cost_list)
    min_index = cost_list.index(shortest_tour)

    return sol_disposition_list[min_index]#, shortest_tour

# E' UNA LOCAL SEARCH
def VND(sol,distanze,apertura,num_nodi):

    x = sol[:]
    level = 1
    level_max = 2

    while level <= level_max:
    # 1) LS -> BEST SOLUTION IN NEIGHBORHOOD
        if level == 1:
            x_new = best_in_two_OPT(x,distanze,apertura,num_nodi) 
        if level == 2:
            x_new = best_in_two_node_swap(x,distanze,apertura,num_nodi)
    # 2) MOVE OR NOT
        if costo(x_new,distanze,apertura,num_nodi) < costo(x,distanze,apertura,num_nodi):
            x = x_new[:]
            level = 1
        else:
            level += 1
    return x

def General_VNS(sol,distanze,apertura,num_nodi):

    x = sol[:]
    level = 1
    level_max = 2

    while level <= level_max:
    # 1) SHAKING    
        if level == 1:
            x_new = two_OPT(x) 
        if level == 2:
            x_new = two_node_swap(x)
    # 2) LOCAL SEARCH by VND
        x_new = VND(x_new,distanze,apertura,num_nodi)
    # 3) MOVE OR NOT
        if costo(x_new,distanze,apertura,num_nodi) < costo(x,distanze,apertura,num_nodi):
            x = x_new[:]
            level = 1
        else:
            level += 1
    print(f"General_VNS  : {x} -> Durata: {costo(x,distanze,apertura,num_nodi)}")
    return x

def Stok_VNS(sol,distanze,apertura,num_nodi):

    x = sol[:]
    level = 1
    level_max = 2

    while level <= level_max:
    # 1) SHAKING
        if level == 1:
            x_new = two_OPT(x) 
        if level == 2:
            x_new = two_node_swap(x)
    # 2) MOVE OR NOT       
        if costo(x_new,distanze,apertura,num_nodi) < costo(x,distanze,apertura,num_nodi):
            x = x_new[:]
            level = 1
        else:
            level += 1
    print(f"Stok_VNS     : {x} -> Durata: {costo(x,distanze,apertura,num_nodi)}")
    return x

def grafico(sol,titolo,distanze,apertura,num_nodi):
    lista = sol[:] #Prendo la sol best trovata
    lista.pop()
    G = nx.DiGraph() #Creo il grafo G
    for i in range(num_nodi-1): #Inserisco i nodi all'interno del grafo
        for j in range(i+1,num_nodi):
            G.add_edges_from([(lista[i], lista[j])])

    # Creo gli archi rossi per il percorso soluzione e i restanti sono archi neri
    archi_rossi = []
    for i in range(num_nodi-1):
        archi_rossi.append((lista[i], lista[i+1]))
    archi_rossi.append((lista[0], lista[num_nodi-1]))
    edge_colours = ['black' if not edge in archi_rossi else 'red'
                    for edge in G.edges()]
    archi_neri = [edge for edge in G.edges() if edge not in archi_rossi]

    
    #Faccio il plot di nodi e archi
    pos = {}
    for i in range(num_nodi):
        ang = random.uniform(0,2*pi)
        x = cos(ang)
        y = sin(ang)
        dist = {i : [distanze[0][i]*x, distanze[0][i]*y]}
        pos.update(dist)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_color = 'b', node_size = 500) #Disegna i nodi
    nx.draw_networkx_labels(G, pos) #Disegna le etichette dei nodi
    nx.draw_networkx_edges(G, pos, edgelist=archi_rossi, edge_color='r', arrows=False) #Disegna gli archi rossi
    if num_nodi <= 10:
        nx.draw_networkx_edges(G, pos, edgelist=archi_neri, arrows=False) #Disegna gli archi neri
    plt.title(titolo)
    plt.show()
