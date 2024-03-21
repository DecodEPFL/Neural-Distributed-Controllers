
import numpy as np 
import torch
import networkx as nx


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


def calculate_critical_coupling_constant(adjacency_matrix, natural_frequencies):
    G = nx.from_numpy_matrix(adjacency_matrix.cpu().detach().numpy())
    laplacian_matrix = np.array(nx.laplacian_matrix(G).todense())
    laplacian_p_inverse = np.linalg.pinv(laplacian_matrix)
    inner_prod_lapl_nat = laplacian_p_inverse @ natural_frequencies.detach().numpy()
    coupling_constant = torch.tensor([np.abs(inner_prod_lapl_nat[np.newaxis, :] -\
                       inner_prod_lapl_nat[:, np.newaxis]).max()*G.number_of_nodes()]).float()
    coupling_constant =  coupling_constant
    return coupling_constant