import numpy as np
from matplotlib import pyplot as plt

class TwoLayerPerceptron:
    def __init__(self, input_size=1, hidden_size=10, output_size=2):
        """Initialize network with He initialization"""
        self.hidden_size = hidden_size
        
        # He initialization for tanh
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2./input_size)
        self.b1 = np.zeros(hidden_size)
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2./hidden_size)
        self.b2 = np.zeros(output_size)
        
        # Activation functions
        self.tanh = lambda x: np.tanh(x)
        self.tanh_deriv = lambda x: 1 - np.tanh(x)**2
        
    def forward(self, X):
        """Forward pass through the network"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2
    
    def backward(self, X, y, y_pred, learning_rate):
        """Backward pass and parameter updates"""
        m = X.shape[0]  # batch size
        
        # Output layer gradient
        dL_dz2 = 2 * (y_pred - y) / m
        dL_dW2 = self.a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0)
        
        # Hidden layer gradient
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * self.tanh_deriv(self.z1)
        dL_dW1 = X.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0)
        
        # Update parameters
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
    
    def train(self, X, y, epochs=10000, learning_rate=0.1, verbose=True):
        """Train the network"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss (MSE)
            loss = np.mean((y_pred - y)**2)
            losses.append(loss)
            
            # Backward pass and update
            self.backward(X, y, y_pred, learning_rate)
            
            # Print progress
            if verbose and (epoch % (epochs//10) == 0 or epoch == epochs-1):
                print(f"Epoch {epoch}: Loss = {loss:.6f}")
        
        return losses

# Target function
def target_function(x, p=4, q=1):
    return np.column_stack([
        np.cos(p/q * x) * np.cos(x),
        np.cos(p/q * x) * np.sin(x)
    ])

# Generate training data
np.random.seed(42)
X_train = np.random.uniform(0, 2*np.pi, 100).reshape(-1, 1)
y_train = target_function(X_train)

# Create and train network
nn = TwoLayerPerceptron(input_size=1, hidden_size=20, output_size=2)
losses = nn.train(X_train, y_train, epochs=10000, learning_rate=0.05)

# Generate test data for visualization
X_test = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
y_test = target_function(X_test)
y_pred = nn.forward(X_test)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")

plt.subplot(1, 2, 2)
plt.plot(y_test[:, 0], y_test[:, 1], label="True function")
plt.plot(y_pred[:, 0], y_pred[:, 1], '--', label="Network prediction")
plt.title("Function Approximation")
plt.legend()
plt.tight_layout()
plt.show()