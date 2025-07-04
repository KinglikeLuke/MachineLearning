import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

def loss_fct(y, y_hat):
    return np.sum((y-y_hat)**2)/np.shape(y)[0]

def training_fct(x, p, q):
    return np.array([np.cos(p/q*x[:,0])*np.cos(x[:,0]), np.cos(p/q*x[:,0])*np.sin(x[:,0])]).T

def norm_input(x):
    return (x - np.pi)/np.pi

def softmax(y, beta=10):
    return np.exp(beta*y)/np.sum(np.exp(beta*y), axis=1)

class NeuralNetwork:
    def __init__(self, N):
        # hidden layer
        self.A_2 = np.random.normal(0, np.sqrt(2/3), (N,2))
        self.b_2 = np.random.normal(0, np.sqrt(2/3), N)
        # surface layer
        self.A_1 = np.random.normal(0, np.sqrt(2/3), (2,N))
        self.b_1 = np.random.normal(0, np.sqrt(2/3), 2)
    
    def forward_propagation(self, x):
        hidden_result = np.tanh((self.A_2@x.T).T + self.b_2[np.newaxis])
        result = (self.A_1@hidden_result.T).T + self.b_1[np.newaxis]
        return result, hidden_result
    
    def backward_propagation(self, learning_rate, x, h, y, y_hat):
        loss_grad = -2*(y-y_hat)
        activation_grad = 1-h**2
        
        self.A_2 -= learning_rate*(np.mean((loss_grad@self.A_1), axis=0)*(activation_grad.T@x).T).T
        self.b_2 -= learning_rate*np.mean((loss_grad@self.A_1)*activation_grad, axis=0)
        
        self.A_1 -= learning_rate*loss_grad.T@h/len(x)
        #learning_rate*(loss_grad@h)
        self.b_1 -= learning_rate*np.mean(loss_grad, axis=0)

def training(network:NeuralNetwork, learning_rate):
    x_batches = np.random.uniform(0, np.pi*2, (100,2))
    y = training_fct(x_batches, P, Q)
    y_hat, h = network.forward_propagation(norm_input(x_batches))
    network.backward_propagation(learning_rate, norm_input(x_batches), h, y, y_hat)
    
def print_network_params(network:NeuralNetwork):
    print(f"A_1: {network.A_1}")
    print(f"b_1: {network.b_1}")
    print(f"A_2: {network.A_2}")
    print(f"b_2: {network.b_2}")
    
def plot_network(network:NeuralNetwork):
    """Plotting as required by task. Layout taken from chatGPT"""
    x_plot = np.linspace(0, 2*np.pi, 200)
    x_network = np.array([x_plot,np.zeros_like(x_plot)]).T
    trainingdata = training_fct(x_network, P, Q).T
    networkdata = network.forward_propagation(norm_input(x_network))[0].T
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.15, hspace=0)
    # Big left plot spanning two rows
    ax_big = fig.add_subplot(gs[:, 0])  # All rows, first column
    # Top-right subplot
    ax_top = fig.add_subplot(gs[0, 1])  # First row, second column
    # Bottom-right subplot
    ax_bot = fig.add_subplot(gs[1, 1], sharex=ax_top)  # Second row, second column
    ax_big.plot(*networkdata, label=r"$\hat{y}(x)$")
    ax_big.plot(*trainingdata, label=r"$f(x)$")
    ax_top.plot(x_plot, trainingdata[0], label=r"$f(x)_1$")
    ax_top.plot(x_plot, networkdata[0], label=r"$\hat{y}(x)_1$")
    ax_bot.plot(x_plot, trainingdata[1], label=r"$f(x)_2$")
    ax_bot.plot(x_plot, networkdata[1], label=r"$\hat{y}(x)_2$")
    ax_bot.set_xlabel("x")
    ax_big.set_title("(a)")
    ax_top.set_title("(b)")
    ax_top.legend(loc="upper center")
    ax_bot.legend()
    ax_big.legend()
    
P = 4
Q = 1
WIDTH = 15

# Displaying pickled models 
# with open("network.pkl", "rb") as f:
#     trained_network:NeuralNetwork = pickle.load(f)
# print_network_params(trained_network)
# plot_network(trained_network)
# x_batches = np.random.uniform(0, np.pi*2, 2000)
# y_hat, _ = trained_network.forward_propagation(norm_input(x_batches))
# print(loss_fct(training_fct(x_batches, P, Q), y_hat))
# plt.show()
# quit()

new_network = NeuralNetwork(WIDTH)

epochs = 2000000
loss_history = []
learning_rate = 0.1
for t in range(epochs):
    training(new_network, learning_rate)
    if t % 100 == 0:
        x_batches = np.random.uniform(0, np.pi*2, (200,2))
        y_hat, _ = new_network.forward_propagation(norm_input(x_batches))
        loss = loss_fct(training_fct(x_batches, P, Q), y_hat)
        loss_history.append(loss)
plot_network(new_network)
print(loss_fct(training_fct(x_batches, P, Q), y_hat))
plt.show()
# Pickling models
# with open("network.pkl", "wb") as f:
#     pickle.dump(new_network, f)