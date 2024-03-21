import pickle
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import torch


def generate_complete_graph(n=8):
    graph = nx.complete_graph(n)
    adjacency_matrix = np.array(nx.adjacency_matrix(graph).todense())
    adjacency_matrix = torch.tensor(adjacency_matrix,
                                    dtype=torch.float
                                    )
    return adjacency_matrix


def generate_erdos_renyi(n=225, p=0.3, seed=1):
    graph = nx.erdos_renyi_graph(n, p, seed=seed)
    adjacency_matrix = np.array(nx.adjacency_matrix(graph).todense())
    adjacency_matrix = torch.tensor(adjacency_matrix,
                                    dtype=torch.float
                                    )
    return adjacency_matrix


def generate_square_lattice(side_size=15, seed=1):
    graph = nx.grid_graph([side_size, side_size])
    adjacency_matrix = np.array(nx.adjacency_matrix(graph).todense())
    adjacency_matrix = torch.tensor(adjacency_matrix,
                                    dtype=torch.float
                                    )
    return adjacency_matrix


def generate_watts_strogatz(n=225, p=0.3, k=5, seed=1):
    graph = nx.watts_strogatz_graph(n, p=p, k=k, seed=seed)
    adjacency_matrix = np.array(nx.adjacency_matrix(graph).todense())
    adjacency_matrix = torch.tensor(adjacency_matrix,
                                    dtype=torch.float
                                    )
    return adjacency_matrix


Adjacecy_complete = generate_complete_graph(n=64) 
Adjacecy_erdos = generate_erdos_renyi(n=64)
Adjacecy_watts = generate_watts_strogatz(n =64)
Adjacecy_square = generate_square_lattice(side_size = 8, seed=1) 

with open('/home/mzakwan/TAC2023/Karutomo_Oscilators/distributed_square_latttice.pkl', 'rb') as file:
      
    # Call load method to deserialze
    myvar_square = pickle.load(file)
  

with open('/home/mzakwan/TAC2023/Karutomo_Oscilators/distributed_complete.pkl', 'rb') as file:
      
    # Call load method to deserialze
    myvar_complete = pickle.load(file)


with open('/home/mzakwan/TAC2023/Karutomo_Oscilators/distributed_erdos_renyi.pkl', 'rb') as file:
      
    # Call load method to deserialze
    myvar_erdos_renyi = pickle.load(file)


with open('/home/mzakwan/TAC2023/Karutomo_Oscilators/distributed_watts_strogatz.pkl', 'rb') as file:
      
    # Call load method to deserialze
    myvar_watts_strogatz = pickle.load(file)


plt.figure()
t_closed_loop_square = myvar_square[2]
r_closed_loop_square = myvar_square[3]
u_square = myvar_square[4]
plt.plot(t_closed_loop_square.detach().numpy(),r_closed_loop_square.detach().numpy())
plt.xlabel('Time(secs)')
plt.ylabel('r(t)')
# plt.legend(['Hamiltonian-distributed'], loc= 1)


G = nx.Graph(np.array(Adjacecy_square))
pos = nx.kamada_kawai_layout(G)
a = plt.axes([0.55, 0.2, .3, .3])
nx.draw(G,pos, node_size=10, node_color = 'darkgreen', edge_color = 'green', alpha = 0.4)

plt.savefig('/home/mzakwan/TAC2023/Karutomo_Oscilators/r_closed_loop_all_square.pdf')


plt.figure()
t_closed_loop_complete = myvar_complete[2]
r_closed_loop_complete = myvar_complete[3]
u_complete = myvar_complete[4]
plt.plot(t_closed_loop_complete.detach().numpy(),r_closed_loop_complete.detach().numpy())
plt.xlabel('Time(secs)')
plt.ylabel('r(t)')
# plt.legend(['Hamiltonian-distributed'], loc=1)

G = nx.Graph(np.array(Adjacecy_complete))
print(G)
pos = nx.random_layout(G)
a = plt.axes([0.55, 0.2, .3, .3])
nx.draw(G,pos, node_size=4, node_color = 'black', edge_color = 'grey', width = 0.005, style = '--',  alpha = 0.4)

plt.savefig('/home/mzakwan/TAC2023/Karutomo_Oscilators/r_closed_loop_all_complete.pdf')


plt.figure()
t_closed_loop_erdos_renyi = myvar_erdos_renyi[2]
r_closed_loop_erdos_renyi = myvar_erdos_renyi[3]
u_erdos_renyi = myvar_erdos_renyi[4]
plt.plot(t_closed_loop_erdos_renyi.detach().numpy(),r_closed_loop_erdos_renyi.detach().numpy())
plt.xlabel('Time(secs)')
plt.ylabel('r(t)')
# plt.legend(['Hamiltonian-distributed'], loc = 1)


G = nx.Graph(np.array(Adjacecy_erdos))
pos = nx.kamada_kawai_layout(G)
a = plt.axes([0.55, 0.2, .3, .3])
nx.draw(G,pos, node_size=4, node_color = 'blue', edge_color = 'skyblue', alpha = 0.4, width = 0.5)
plt.xticks([])
plt.yticks([])



plt.savefig('/home/mzakwan/TAC2023/Karutomo_Oscilators/r_closed_loop_all_erdos_renyi.pdf')


plt.figure()
t_closed_loop_watts_strogatz = myvar_watts_strogatz[2]
r_closed_loop_watts_strogatz = myvar_watts_strogatz[3]
u_watts_strogatz = myvar_watts_strogatz[4]
plt.plot(t_closed_loop_watts_strogatz.detach().numpy(),r_closed_loop_watts_strogatz.detach().numpy())
plt.xlabel('Time(secs)')
plt.ylabel('r(t)')
# plt.legend(['Hamiltonian-distributed'], loc = 1)

G = nx.Graph(np.array(Adjacecy_watts))
pos = nx.kamada_kawai_layout(G)
a = plt.axes([0.55, 0.2, .3, .3])
nx.draw(G,pos, node_size=4, node_color = 'darkred', edge_color = 'red', alpha = 0.4, width = 0.5)

plt.savefig('/home/mzakwan/TAC2023/Karutomo_Oscilators/r_closed_loop_all_watts_strogatz.pdf')




