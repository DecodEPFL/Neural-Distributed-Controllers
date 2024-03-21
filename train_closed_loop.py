import torch 
from torch import nn
import numpy as np
from matplotlib import pyplot as plt

from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torchdiffeq import odeint
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = 'cuda' 
class TrainDataset(Dataset):
    def __init__(self, q_range, p_range):

        grid_q, grid_p = torch.meshgrid(
            q_range, p_range)

        self.q_data = grid_q.reshape(-1,)
        self.p_data = grid_p.reshape(-1,)

    def __len__(self):
        return len(self.q_data)

    def __getitem__(self, idx):

        return self.q_data[idx], self.p_data[idx]
    

class kuramoto_angular(nn.Module):  

    def __init__(self,  adj_mat, coupling, n_nodes=4, natfreqs=None):
        super(kuramoto_angular, self).__init__()
        self.adj_mat = adj_mat
        self.coupling = coupling
        self.K = nn.Parameter(torch.rand(4,4))
        if natfreqs is not None:
            self.natfreqs = natfreqs
            self.n_nodes = len(natfreqs)
        else:
            self.n_nodes = n_nodes
            self.natfreqs = torch.randn((self.n_nodes,1))

        
        self.fc1 = nn.Linear(4,32)
        self.fc2 = nn.ELU()
        self.fc3 = nn.Linear(32,32)
        self.fc4 = nn.ELU()
        self.fc5 = nn.Linear(32,4)
        self.fc6 = nn.ELU()
        self.controller = nn.Sequential(self.fc1,self.fc2,self.fc3,self.fc4,self.fc5,self.fc6)
        

    def forward(self, t, x):

        x_1, x_2 = torch.split(x, len(x) // 2)
   
        assert len(x) // 2 == len(self.natfreqs) == len(self.adj_mat), \
            'Input dimensions do not match, check lengths'
       
        angles_i, angles_j = torch.meshgrid(x_1.view(4), x_1.view(4), indexing=None)
        x_i, x_j = torch.meshgrid(x_2.view(4), x_2.view(4), indexing=None)
        g_ij = torch.cos(angles_j - angles_i)
        x_ij = x_j - x_i
        interactions = self.adj_mat * g_ij *  x_ij                  # Aij * sin(j-i)
        u = self.controller(x_2.T).T                                #controller
        dxdt = 5 * (u/4.0) * interactions.sum(axis=0).reshape(4,1) # sum over incoming interactions (5/4)* 

        stacked_dxdt = torch.cat((x_2, dxdt), dim=0)
        return stacked_dxdt


class kuramoto_angular_closed_loop(nn.Module):  

    def __init__(self,  adj_mat, coupling, n_nodes=4, natfreqs=None):
        super(kuramoto_angular_closed_loop, self).__init__()
        self.adj_mat = adj_mat
        self.coupling = coupling
        self.K = nn.Parameter(torch.rand(4,4))
        if natfreqs is not None:
            self.natfreqs = natfreqs
            self.n_nodes = len(natfreqs)
        else:
            self.n_nodes = n_nodes
            self.natfreqs = torch.randn((self.n_nodes,1))

        # Define controller parameters 
        self.act = nn.Tanh()
        self.K = nn.Parameter(torch.rand(4,4))
        self.b = nn.Parameter(torch.rand(4,1))
        self.G = nn.Parameter(torch.rand(4,4))
        self.J = torch.tensor([[0, -1, 0 ,0],
                               [1, 0 , 0 ,0],
                               [0 ,0 , 0 ,-1],
                               [0, 0, 1, 0]])
        self.gamma =  5.0 # torch.exp(nn.Parameter(torch.rand(1))) +
       
   
    def forward(self, t, x):

        x_1, x_2, xi = torch.split(x, len(x) // 3)
        # print("x_1", x_1)
        # print("x_2",x_2)
        # print("xi",xi)
   
        assert len(x) // 3 == len(self.natfreqs) == len(self.adj_mat), \
            'Input dimensions do not match, check lengths'
       

        angles_i, angles_j = torch.meshgrid(x_1.view(4), x_1.view(4), indexing=None)
        x_i, x_j = torch.meshgrid(x_2.view(4), x_2.view(4), indexing=None)
        g_ij = torch.cos(angles_j - angles_i)
        x_ij = x_j - x_i
        interactions = self.adj_mat * g_ij *  x_ij                  # Aij * sin(j-i)
                                       #controller
        u = -(self.adj_mat * self.G).T @ self.dH_dxi(xi) 
        dxdt = (5/4.0) * u * interactions.sum(axis=0).reshape(4,1) # sum over incoming interactions (5/4)* 
        
        self.gamma = 0.1 * torch.max(torch.real(torch.linalg.eigvals(self.G @ self.G.T))) 
        R = self.gamma * torch.eye(4)
        F = self.J - R
        dxi_dt = F @ self.dH_dxi(xi) + (self.adj_mat * self.G) @ x_2
        stacked_dxdt = torch.cat((x_2, dxdt), dim=0)
        stacked_cl_loop = torch.cat((stacked_dxdt, dxi_dt), dim=0)
        return stacked_cl_loop
    
    def dH_dxi(self,xi):

        M = self.act(self.K @ xi + self.b)
         
        return self.K.T @ M



Adjacecy = torch.tensor([[0 ,1, 0 ,1],
                        [1, 0, 1 ,0],
                        [0 ,1, 0 ,1],
                        [1, 0, 1, 0]])


endtime = 3.0
tol = 1e-7
class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, endtime]).float()
        self.time = torch.linspace(0,endtime,int(20*endtime))

    def node_propagation(self,x): 
        out = odeint(self.odefunc, x, self.time, rtol=tol, atol=tol, method="euler")
        return out

    def forward(self,x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=tol, atol=tol, method="euler")
        return out[1]
        # return self.node_propagation(x)[-1]
    

N_nodes = 4
odefunc = kuramoto_angular_closed_loop(adj_mat= Adjacecy, coupling = 2.0 , n_nodes=N_nodes, natfreqs=torch.randn(N_nodes,1)*0.2)

vec_angles = torch.randn(12,1)*0.2
print("initial conditions",vec_angles)
# vec_angles = torch.tensor([0.2,0.1,0.4,0.5]).reshape(4,1) 
# print(odefunc(0.1,vec_angles))

tol = 1e-7
endtime = 3
t = torch.linspace(0., endtime, 200*endtime)
out = odeint(odefunc, vec_angles, t, method="rk4")


out_r = out[:,N_nodes:-1].squeeze(-1)
n = out_r.shape[-1]
diff = torch.cos(out_r.unsqueeze(-1) - out_r.unsqueeze(-2))
sum_diff = (diff).sum(-1).sum(-1)
r = (1 / n) * (sum_diff ** (1 / 2))

plt.figure()
plt.plot(t, r.detach().numpy())
plt.ylabel('r(t)')
plt.xlabel('t(secs)')
plt.savefig("/home/mzakwan/TAC2023/Karutomo_Oscilators/r_test_closed_loop.pdf")

# plt.figure()
# plt.plot(t, out[:,4].detach().numpy())
# plt.plot(t, out[:,5].detach().numpy())
# plt.plot(t, out[:,6].detach().numpy())
# plt.plot(t, out[:,7].detach().numpy())
# plt.savefig("/home/mzakwan/TAC2023/Karutomo_Oscilators/graph_before.pdf")

model = ODEBlock(odefunc)

class ImageClassifier_global(LightningModule):
        def __init__(self):
            super().__init__()
            self.save_hyperparameters()
            self.net = model
            self.adj_mat = Adjacecy
            # self.automatic_optimization = False

        def forward(self,x):
            return self.net(x)
        
        def calculate_loss(self, x):
            x_i, x_j = torch.meshgrid(x.view(4), x.view(4), indexing=None)
            diff = x_j-x_i # create the difference of x_j - x_i in each row element
            sin_sqr_diff   =  torch.sin(diff)**2
            adjacency_mult = self.adj_mat*sin_sqr_diff
            loss = adjacency_mult.sum() # sum over i and j
            return loss
    

        def training_step(self, batch, batch_idx):
            
            x0 = torch.randn(12,1)*0.1
            # x0 = torch.tensor([0.1,0.1,0.1,0.1,0.2,0.5,0.1,0.4,0.0,0.0,0.0,0.0]).reshape(12,1)
            x = self.forward(x0)
            # x_ext =self.net.node_propagation(x0).squeeze(-1)
            _ , x_2, xi = torch.split(x, len(x) // 3)
            # x_2_ext = x_ext[:,4:8]
            # print(x_2_ext.shape)
            # loss = self.calculate_loss(x_2_ext)
            x_long = self.net.node_propagation(x0).squeeze(-1)
            cumm_loss = 0.0
            for idx in range(0,x_long.shape[0]):
                xloss = x_long[idx,:]
                x_1, x_2, xi = torch.split(xloss, len(xloss) // 3)
                loss_int = self.calculate_loss(x_2)
                cumm_loss += loss_int
            
            loss = cumm_loss.mean()
            self.log("total_loss", loss, prog_bar=True)
            # loss.backward(retain_graph=True)
            # self.manual_backward(loss,retain_graph = True)
            return loss
            

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)
            return optimizer
        
    

if __name__ == "__main__":


    training_data = TrainDataset(
        torch.linspace(1/4-2, 1/4+2, 1), torch.linspace(-2, 2, 1))

    train_dataloader = DataLoader(
        training_data, batch_size=1, num_workers=30, persistent_workers=True)


    torch.set_float32_matmul_precision('medium')
        

    #Defining the logger 

    model_ode = ImageClassifier_global()

    trainer = pl.Trainer(
        max_epochs=2000,
        accelerator='cpu',
        num_nodes=1,
    )
    
    trainer.fit(model_ode, train_dataloader)
    trainer.save_checkpoint(
            "/home/mzakwan/TAC2023/Karutomo_Oscilators/test_lightning_model.ckpt")
    time.sleep(5)


    # vec_angles = torch.tensor([0.1,0.1,0.1,0.1,0.2,0.5,0.1,0.4,0.0,0.0,0.0,0.0]).reshape(12,1)
    vec_angles = torch.randn(12,1)*0.1
    tol = 1e-7
    endtime = endtime + 0.0
    t = torch.linspace(0., endtime, int(200*endtime))
    out = odeint(odefunc, vec_angles, t, method="euler")

# out = out.cpu().detach()

# plt.figure()
# plt.plot(t.cpu(), out[:,4])
# plt.plot(t.cpu(), out[:,5])
# plt.plot(t.cpu(), out[:,6])
# plt.plot(t.cpu(), out[:,7])
# plt.savefig("/home/mzakwan/TAC2023/Karutomo_Oscilators/graph.pdf")
    # out = out.squeeze(-1)
    # _, out_r, _ = torch.split(out, len(out) // 3)
    out_r = out[:,N_nodes:N_nodes+3].squeeze(-1)
    n = out_r.shape[-1]
    diff = torch.cos(out_r.unsqueeze(-1) - out_r.unsqueeze(-2))
    sum_diff = (diff).sum(-1).sum(-1)
    r = (1 / n) * (sum_diff ** (1 / 2))

    plt.figure()
    plt.plot(t, r.detach().numpy())
    plt.ylabel('r(t)')
    plt.xlabel('t(secs)')
    plt.savefig("/home/mzakwan/TAC2023/Karutomo_Oscilators/r_closed_loop.pdf")


    plt.figure()
    plt.plot(t, out[:,4].detach().numpy())
    plt.plot(t, out[:,5].detach().numpy())
    plt.plot(t, out[:,6].detach().numpy())
    plt.plot(t, out[:,7].detach().numpy())
    plt.savefig("/home/mzakwan/TAC2023/Karutomo_Oscilators/graph_trained_closed_loop.pdf")

# # out = out.squeeze(-1).T.tolist()
# # out = [torch.tensor(sublist) for sublist in out]
# # trajectories = torch.stack(out)
# # num_trajectories = len(trajectories)
# # # Expand dimensions for broadcasting
# # trajectories_expanded = trajectories.unsqueeze(1)
# # trajectories_expanded = trajectories_expanded.expand(-1, num_trajectories, -1)
# # # Compute differences while excluding self-differences
# # differences = trajectories_expanded - trajectories
# # mask = torch.eye(num_trajectories).bool()
# # differences = (1/4)*torch.sqrt(torch.cos(differences[~mask].view(-1, trajectories.size(1))).sum(dim=0).reshape(trajectories.size(1),1))


# angles_i, angles_j = np.meshgrid(x.float(), x.float())
#         # print(angles_i)
#         # print(angles_j)
# interactions = Adjacecy * torch.cos(torch.tensor(angles_j - angles_i))  # Aij * sin(j-i)
#         # print("size", interactions.sum(axis=0).shape)
# dxdt = self.natfreqs + self.coupling * interactions.sum(axis=0).reshape(4,1)  # sum over incoming interactions