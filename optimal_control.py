import torch 
from torch import nn 
from abc import abstractmethod, ABC
from torchdiffeq import odeint
import numpy as np
import logging
import networkx as nx
from matplotlib import pyplot as plt

class ControlledDynamics(torch.nn.Module):
    def __init__(self, state_var_list: list):
        """
        The template for control dynamics, which showcases how the controlled dynamics code is
        structured to interface seamlessly with this package.
        :param state_var_list: A list of the state variable labels/names for utility purposes.

        E.g. in SIS dynamics model our state is a matrix of dimensions:
        `m x n_nodes`, for `n_nodes` nodes and `m` = 2 state variables:
        `state_var_list = [susceptible, infected]`.
        This indicates that in our state vector, `x[:,0,:]` contains the **susceptible** values
        across all batch samples for all nodes, whereas `x[:,1:]` contains the **infected**
        values across all batch samples for all nodes.
        """
        super().__init__()
        self.state_var_list = state_var_list

    @abstractmethod
    def forward(self, t, x, u=None) -> torch.Tensor:
        """
        The abstract controlled dynamics forward method, used to calculate the derivative under
        control.
        :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
        across batch.
        :param x: A tensor containing state values, shape: `[b, m, n_nodes ]`.
        :param u: A tensor containing the control values, can be of arbitrary shape.
        E.g. for a scalar control signal per node, shape: `[b, m, n_nodes]`.
        :return: `dx` A tensor containing the derivative (**amount of change**) of `x`,
        shape: `[b, m, n_nodes]`
        """
        # TODO: memory efficient implementation for heterogenous nodes:
        # i.e. state variables that apply to or are shared by a subset of nodes.
        pass

class BackwardKuramotoDynamics(ControlledDynamics):
    def __init__(self,
                 adjacency_matrix,  # oscillator coupling
                 driver_matrix,  # identity
                 coupling_constant,  # K
                 natural_frequencies,  # omega
                 dtype=torch.float32,
                 device=None
                 ):
        """
        Kuramoto primal system
        :param adjacency_matrix: The matrix A
        :param driver_matrix: The matrix B
        :param dtype: Datatype of the calculations
        :param device: Device of the calculations, usually "cpu" or "cuda:0"
        """
        super().__init__(['theta', 'p'])
        self.device = device
        self.dtype = dtype
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda:" + str(torch.cuda.current_device())
            else:
                self.device = torch.device("cpu")
        # unsqueeze matrices in the adjacency_matrix dimension so that operation broadcasts across
        self.interaction_matrix = adjacency_matrix.unsqueeze(0)
        self.n_nodes = adjacency_matrix.shape[-1]
        self.driver_matrix = driver_matrix.unsqueeze(0)
        self.degree_matrix = torch.diag(self.interaction_matrix.sum(-1)[0]).unsqueeze(0)
        self.coupling_constant = coupling_constant
        self.natural_frequencies = natural_frequencies.unsqueeze(0)
        self.label = 'kuramoto-multiplicative-backward'
        self.k_by_N = self.coupling_constant / self.n_nodes

    def forward(self, t, x, u):
        """
        Evaluation of dx/dts
        :param x: current state values for nodes. Please ensure the input is not permuted, unless you know
        what you doing.
        :param t: time scalar, which is not used as the model is time invariant.s
        :param u: control vectors. In this case make it has proper dimensionality such that
        torch.matmul(driver_matrix, u) is possible.
        :return: dx/dt
        """
        p = x[:, 0, :]
        theta = x[:, 1, :]
        dp = self._adjoint_backward(p, theta, t, u)
        dx = dp
        return dx

    def _adjoint_backward(self, p, theta, t, u):
        """
        Evaluation of the adjoint system dp/dt
        :param p: current adjoint vector.
        :param theta: current theta vector.
        :param t: time scalar (note that the current dynamics has no explicit
                  time dependence)
        :param u: control vector
        :return: dp/dt, du/dt
        """
        k_by_n_u = self.k_by_N * u

        sin_theta = torch.sin(theta).unsqueeze(-1)
        cos_theta = torch.cos(theta).unsqueeze(-1)

        interaction_term_1_part_a = (cos_theta * torch.matmul(self.interaction_matrix,
                                                              cos_theta)).squeeze(-1)
        interaction_term_1_part_b = (sin_theta * torch.matmul(self.interaction_matrix,
                                                              sin_theta)).squeeze(-1)
        interaction_term_1 = interaction_term_1_part_a + interaction_term_1_part_b

        control_vector_1 = k_by_n_u * p
        sum_term_1 = control_vector_1 * interaction_term_1

        # sum, term 2
        control_vector_2 = self.k_by_N * u  # optimize here
        interaction_term_2_part_a = (cos_theta * torch.matmul(self.interaction_matrix,
                                                              p.unsqueeze(-1) * cos_theta)
                                     ).squeeze(-1)
        interaction_term_2_part_b = (sin_theta * torch.matmul(self.interaction_matrix,
                                                              p.unsqueeze(-1) * sin_theta)
                                     ).squeeze(-1)
        interaction_term_2 = interaction_term_2_part_a + interaction_term_2_part_b

        sum_term_2 = control_vector_2 * interaction_term_2

        dp = sum_term_1 - sum_term_2

        return dp
    

class BaseController(ABC, nn.Module):
    def __init__(self):
        """
        The template for a controller class.
        """
        super().__init__()

    @abstractmethod
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        A method implementing the control calculation based on either or both time and state
        values.
        The following notation is used for shapes:
        `b` dimensions for batch size, `m` dimensions for state variables and `n_nodes` dimensions for
        number of nodes.
        Explicit dimension assignment of inputs and outputs for specific functionality are
        defined in the corresponding implementation.
        :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
        across batch.
        :param x: A tensor containing state values, shape: `[b, m, n_nodes ]`
        :return: A tensor containing control values, shape: `[b, ?, ?]`

        A controller  that takes as input a `b x 1` dimensional time `t` tensor and an `b x m x
        n_nodes`-dimensional state `x` to calculate control signals `u` of arbitrary dimensionality.
        """
        pass


class ForwardKuramotoDynamics(ControlledDynamics):
    def __init__(self,
                 adjacency_matrix,
                 driver_matrix,
                 coupling_constant,
                 natural_frequencies,
                 dtype=torch.float32,
                 device=None
                 ):
        """
        Kuramoto primal system
        :param adjacency_matrix: The matrix A
        :param driver_matrix: The matrix B
        :param dtype: Datatype of the calculations
        :param device: Device of the calculations, usuallu "cpu" or "cuda:0"
        """
        super().__init__(['theta'])
        self.device = device
        self.dtype = dtype
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda:" + str(torch.cuda.current_device())
            else:
                self.device = torch.device("cpu")
        # unsqueeze matrices in the adjacency_matrix dimension so that operation broadcasts across
        # batch dimension
        self.adjacency_matrix = adjacency_matrix.unsqueeze(0)
        self.n_nodes = adjacency_matrix.shape[-1]
        self.driver_matrix = driver_matrix.unsqueeze(0)
        self.driver_index = self.driver_matrix.squeeze(0).nonzero()
        self.degree_matrix = torch.diag(self.adjacency_matrix.sum(-1)[0]).unsqueeze(0)
        self.laplacian_matrix = self.degree_matrix - self.adjacency_matrix
        self.coupling_constant = coupling_constant
        self.natural_frequencies = natural_frequencies.unsqueeze(0)
        self.label = 'kuramoto-multiplicative-forward'

    def forward(self, t, x, u):
        """
        Evaluation of dx/dts
        :param x: current state values for nodes. In this case the state is the angle of the
        system and the angular velocity.
        :param t: time scalar, which is not used as the model is time invariant.
        :param u: control vector. In this case make it has proper dimensionality such that
        torch.matmul(driver_matrix, u) is possible.
        :return: dx/dt
        """

        if len(x.shape) > 2:
            theta = x[:, 0, :]
        else:
            theta = x

        dx = self.natural_frequencies
        # print("shape of x", x.shape)
        # calculation of the interaction term: F
        sin_theta = torch.sin(theta).unsqueeze(-1)
        cos_theta = torch.cos(theta).unsqueeze(-1)

        interaction_term = (cos_theta * torch.matmul(self.adjacency_matrix, sin_theta)) \
            .squeeze(-1)
        interaction_term = interaction_term - (sin_theta * torch.matmul(self.adjacency_matrix,
                                                                        cos_theta)).squeeze(-1)
        coupling_term = (self.coupling_constant / self.n_nodes) * interaction_term
        if u is not None:
            control_term = torch.matmul(self.driver_matrix, u.unsqueeze(-1)).squeeze(-1)
            dx = dx + control_term * coupling_term
        else:
            dx = dx + coupling_term

        return dx
    
class AdjointGD(BaseController):

    def __init__(self,
                 forward_dynamics,
                 backward_dynamics,
                 theta_0,
                 n_timesteps,
                 total_time,
                 learning_rate=1,
                 beta=10 ** -7,
                 iterations=10,
                 control_change_tolerance=10 ** -5,
                 progress_bar=None,
                 ode_int_kwargs=None
                 ):
        super().__init__()
        self.forward_dynamics = forward_dynamics
        self.backward_dynamics = backward_dynamics
        self.n_nodes = self.forward_dynamics.adjacency_matrix.shape[-1]
        self.device = self.forward_dynamics.adjacency_matrix.device
        self.n_timesteps = n_timesteps
        self.timesteps = torch.linspace(0,
                                        total_time,
                                        n_timesteps,
                                        device=self.device
                                        )
        self.total_time = total_time
        self.learning_rate = learning_rate
        self.beta = beta
        self.iterations = iterations
        self.control_change_tolerance = control_change_tolerance
        self.progress_bar = progress_bar
        self.ode_int_kwargs = ode_int_kwargs
        self.u_baseline = 1+ torch.rand([theta_0.shape[0],
                                       self.n_timesteps, 1],
                                      device=self.device,
                                      requires_grad=False
                                      )
        self.theta_0 = theta_0
        self._learn(theta_0)

    def _dtheta(self, t, theta):
        u = self.forward(t)
        return self.forward_dynamics(
            t,
            u=u,
            x=theta
        )

    def _dp(self, t, p, thetas):
        u = self.forward(t).detach()
        theta = thetas[torch.argmin(torch.abs(self.timesteps - t))]
        return - self.backward_dynamics(
            t=t,
            x=torch.stack([p, theta], 1),
            u=u
        )

    def _learn(self, theta_0):
        control_value_change = np.infty
        iteration = 0

        adjacency_matrix = self.forward_dynamics.adjacency_matrix
        coupling_constant = self.forward_dynamics.coupling_constant
        while control_value_change > self.control_change_tolerance and iteration < self.iterations:
            iteration = iteration + 1
            try:
                #forward integration
                thetas = odeint(self._dtheta,
                                theta_0.detach(),
                                self.timesteps.detach(),
                                **self.ode_int_kwargs
                                )
                theta_total_time = thetas[-1]

                # adjoint state at T: total_time
                sin_theta_total_time = torch.sin(2 * theta_total_time)
                cos_theta_total_time = torch.cos(2 * theta_total_time)
                sin_product_p = sin_theta_total_time * (
                    (adjacency_matrix @ cos_theta_total_time.unsqueeze(-1)).squeeze(-1)) -\
                                cos_theta_total_time * (
                     (adjacency_matrix @ sin_theta_total_time.unsqueeze(-1)).squeeze(-1)
                    )
                p_total_time = 1 / 2 * sin_product_p
                
                # backward integration
                all_p = odeint(
                    lambda t, y: self._dp(t, y, thetas),
                    p_total_time,
                    self.timesteps,
                    **self.ode_int_kwargs
                ) 
                sin_thetas = torch.sin(thetas).unsqueeze(-1)
                cos_thetas = torch.cos(thetas).unsqueeze(-1)
                sin_product = cos_thetas * (
                              adjacency_matrix.unsqueeze(0).unsqueeze(0) @ sin_thetas) - \
                              sin_thetas * (adjacency_matrix.unsqueeze(0).unsqueeze(0) @ cos_thetas)
                u_update_term = (all_p.unsqueeze(-2) @ sin_product).squeeze(-1).squeeze(-1)
                control_value_change = self.u_baseline.detach()
                self.u_baseline = self.u_baseline - \
                                  self.learning_rate * (
                                          self.beta * self.u_baseline + 
                                          (coupling_constant / self.n_nodes) *
                                          u_update_term)
                control_value_change = torch.mean((control_value_change - self.u_baseline) ** 2)
                # loss = (sinTprod**2).mean()
                loss = ((self.forward_dynamics.laplacian_matrix
                          @ theta_total_time.unsqueeze(-1)) ** 2).sum()
               
                logging.info('GD step loss: ' + str(loss.item()))

            except AssertionError:
                logging.info("ODESolver encountered numerical instability, gradient descent stops!")
                return

    def forward(self, t: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        # we perform a midpoint interpolation between controls
        t_diff = t - self.timesteps
        t_prev_diff = t_diff.clone()
        t_prev_diff[t_prev_diff < 0] = np.infty
        t_prev_idx = torch.argmin(t_prev_diff)
        u_prev = self.u_baseline[:, t_prev_idx]
        t_next_diff = t_diff
        t_next_diff[t_next_diff >= 0] = -np.infty
        t_next_idx = torch.argmax(t_next_diff)

        t_prev = self.timesteps[t_prev_idx]
        t_next = self.timesteps[t_next_idx]
        u_next = self.u_baseline[:, t_next_idx]
        time_weight = (t-t_prev)/(t_next-t_prev)
        u = u_prev*(1-time_weight)*u_prev + u_next*(time_weight)
        return u_prev
    



def generate_complete_graph(n=8):
    graph = nx.complete_graph(n)
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

n = 64
adjacency_matrix = generate_complete_graph(n) 
driver_vector = torch.ones([adjacency_matrix.shape[0],1])


total_time = 3.0
natural_frequencies = torch.empty([n])\
                               .normal_(mean=0, std=0.2)

critical_coupling_constant = calculate_critical_coupling_constant(adjacency_matrix, natural_frequencies)
coupling_constant = 0.1*critical_coupling_constant
theta_0 = torch.empty([1, n]).normal_(mean=0, std=0.2)


forward_dynamics = ForwardKuramotoDynamics(adjacency_matrix, 
                                           driver_vector, 
                                           coupling_constant, 
                                           natural_frequencies
                                          )
backward_dynamics = BackwardKuramotoDynamics(adjacency_matrix, 
                                             driver_vector, 
                                             coupling_constant, 
                                             natural_frequencies
                                            )



logging.getLogger().setLevel(logging.INFO)
torch.manual_seed(1)
baseline_control = AdjointGD(
    forward_dynamics,
    backward_dynamics,
    theta_0,
    n_timesteps=100,
    total_time=total_time,
    learning_rate=5,
    beta=1e-07,
    iterations=2,
    control_change_tolerance=1e-07,
    progress_bar=None,
    ode_int_kwargs={'method':'dopri5', 
                    # 'options' : {'step_size' : 0.01}, For RK4
                   },
)
# we save the trained model.
torch.save(baseline_control.u_baseline, '/home/mzakwan/TAC2023/Karutomo_Oscilators/baseline_signal.pt')


def evaluate(dynamics, theta_0, controller, total_time, n_interactions):
    timesteps = torch.linspace(0, total_time, n_interactions)
    theta = theta_0
    control_trajectory = [torch.zeros([1, 1])]
    state_trajectory = [theta_0]
    for i in range(timesteps.shape[0] - 1):
        time_start = timesteps[i]
        time_end = timesteps[i + 1]
        current_interval = torch.linspace(time_start, time_end, 2)
        u = controller(time_start, theta)
        controlled_dynamics = lambda t, y: dynamics(t=t,
                                                    x=y,
                                                    u=u)

        theta = odeint(controlled_dynamics,
                       theta,
                       current_interval,
                       method='rk4',
                       options={'step_size': 0.01}
                       )[-1]
        control_trajectory.append(u)
        state_trajectory.append(theta)
    return control_trajectory, state_trajectory

def order_parameter_cos(x):
    """
    The order parameter calculated based on the cosine formula in the paper.
    :param x: the state of the oscillators
    :return: The r value
    """
    n = x.shape[-1]
    diff = torch.cos(x.unsqueeze(-1) - x.unsqueeze(-2))
    sum_diff = (diff).sum(-1).sum(-1)
    r = (1 / n) * (sum_diff ** (1 / 2))
    return r

control_trajectory, state_trajectory =\
evaluate(forward_dynamics, theta_0, baseline_control, total_time, 100)
adj_control = torch.cat(control_trajectory).squeeze().cpu().detach().numpy()
adj_states = torch.cat(state_trajectory).cpu().detach().numpy()
adj_r = order_parameter_cos(torch.tensor(adj_states)).cpu().numpy()
adj_e = (adj_control**2).cumsum(-1)

# print(adj_states.shape)
np.save('/home/mzakwan/TAC2023/Karutomo_Oscilators' + 'agd_control_signal', adj_control)
np.save('/home/mzakwan/TAC2023/Karutomo_Oscilators' + 'agd_states', adj_states)
np.save('/home/mzakwan/TAC2023/Karutomo_Oscilators' + 'agd_syn_param', adj_r)



# print(adj_r.shape)
plt.figure(1)
plt.plot(np.linspace(0, total_time, adj_r.shape[0]), adj_r)
plt.savefig("/home/mzakwan/TAC2023/Karutomo_Oscilators/r_optimal.pdf")


plt.figure(2)
plt.plot(np.linspace(0, total_time, adj_r.shape[0]), adj_control)
plt.savefig("/home/mzakwan/TAC2023/Karutomo_Oscilators/r_optimal_u.pdf")



# # fig = comparison_plot(nn_r, adj_r, np.linspace(0, total_time, adj_r.shape[0]),

# optimal_u = baseline_control.u_baseline.squeeze()

    
class ForwardKuramotoDynamics_passive(ControlledDynamics):
    def __init__(self,
                 adjacency_matrix,
                 driver_matrix,
                 coupling_constant,
                 natural_frequencies,
                 dtype=torch.float32,
                 device=None
                 ):
        """
        Kuramoto primal system
        :param adjacency_matrix: The matrix A
        :param driver_matrix: The matrix B
        :param dtype: Datatype of the calculations
        :param device: Device of the calculations, usuallu "cpu" or "cuda:0"
        """
        super().__init__(['theta'])
        self.device = device
        self.dtype = dtype
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda:" + str(torch.cuda.current_device())
            else:
                self.device = torch.device("cpu")
        # unsqueeze matrices in the adjacency_matrix dimension so that operation broadcasts across
        # batch dimension
        self.adjacency_matrix = adjacency_matrix.unsqueeze(0)
        self.n_nodes = n
        self.driver_matrix = driver_matrix.unsqueeze(0)
        self.driver_index = self.driver_matrix.squeeze(0).nonzero()
        self.degree_matrix = torch.diag(self.adjacency_matrix.sum(-1)[0]).unsqueeze(0)
        self.laplacian_matrix = self.degree_matrix - self.adjacency_matrix
        self.coupling_constant = coupling_constant
        self.natural_frequencies = natural_frequencies.unsqueeze(0)
        self.label = 'kuramoto-multiplicative-forward'

    def forward(self, t, x, u):
        """
        Evaluation of dx/dts
        :param x: current state values for nodes. In this case the state is the angle of the
        system and the angular velocity.
        :param t: time scalar, which is not used as the model is time invariant.
        :param u: control vector. In this case make it has proper dimensionality such that
        torch.matmul(driver_matrix, u) is possible.
        :return: dx/dt
        """
        # print("shape of x",x.shape)
        x_1, x_2 = x[0:self.n_nodes,:],  x[self.n_nodes:,:]
        angles_i, angles_j = torch.meshgrid(x_1.view(self.n_nodes), x_1.view(self.n_nodes), indexing=None)
        x_i, x_j = torch.meshgrid(x_2.view(self.n_nodes), x_2.view(self.n_nodes), indexing=None)
        g_ij = torch.cos(angles_j - angles_i)
        x_ij = x_j - x_i
        interactions = self.adjacency_matrix * g_ij *  x_ij  
        interactions = interactions.reshape(self.n_nodes,self.n_nodes)
        u1 = u.repeat(self.n_nodes,1)
        dxdt = (self.coupling_constant/self.n_nodes) * u * interactions.sum(axis=0).reshape(self.n_nodes,1) # sum over incoming interactions
        stacked_dxdt = torch.cat((x_2, dxdt), dim=0)
    
        return stacked_dxdt
    
forward_dynamics_passive = ForwardKuramotoDynamics_passive(adjacency_matrix, 
                                           driver_vector, 
                                           coupling_constant, 
                                           natural_frequencies
                                          )



natural_frequencies_2 = torch.empty([n, 1])\
                               .normal_(mean=0, std=0.2)
theta_0_passive = torch.cat((torch.empty([n, 1]).normal_(mean=0, std=0.2),natural_frequencies_2), dim = 0)
control_trajectory_passive, state_trajectory_passive =evaluate(forward_dynamics_passive, theta_0_passive, baseline_control, total_time, 100)
adj_control = torch.cat(control_trajectory_passive).squeeze().cpu().detach().numpy()
adj_states_passive = torch.cat(state_trajectory_passive).cpu().detach().numpy()
adj_r_passive = order_parameter_cos(torch.tensor(adj_states_passive)).cpu().numpy()
adj_e = (adj_control**2).cumsum(-1)

# print(adj_states.shape)
np.save('/home/mzakwan/TAC2023/Karutomo_Oscilators' + 'agd_control_signal_passive', adj_control)
np.save('/home/mzakwan/TAC2023/Karutomo_Oscilators' + 'agd_states_passive', adj_states_passive)
np.save('/home/mzakwan/TAC2023/Karutomo_Oscilators' + 'agd_syn_param_passive', adj_r_passive)


# print(state_trajectory_passive)
plt.figure(3)
plt.plot(np.linspace(0, total_time, adj_r_passive.shape[0]), adj_r_passive)
plt.show()
plt.savefig("/home/mzakwan/TAC2023/Karutomo_Oscilators/r_optimal_passive.pdf")
# fig = comparison_plot(nn_r, adj_r, np.linspace(0, total_time, adj_r.shape[0]),