import numpy as np
import matplotlib.pyplot as plt

# Define the data
data = {
    'ω1': [(0.1, 1.1), (6.8, 7.1), (-3.5, -4.1), (2, 2.7), (4.1, 2.8), (3.1, 5), (-0.8, -1.3), (0.9, 1.2), (5, 6.4), (3.9, 4)],
    'ω2': [(7.1, 4.2), (-1.4, -4.3), (4.5, 0), (6.3, 1.6), (4.2, 1.9), (1.4, -3.2), (2.4, -4), (2.5, -6.1), (8.4, 3.7), (4.1, -2.2)],
    'ω3': [(-3, -2.9), (0.5, 8.7), (2.9, 2.1), (-0.1, 5.2), (-4, 2.2), (-3.2, 3.7), (-4.4, 6.2), (-4.1, 3.4), (-5.1, 1.6), (1.9, 5.1)],
    'ω4': [(-2, -8.4), (-8.9, 0.2), (-4.2, -7.7), (-8.5, -3.2), (-6.7, -4), (-0.5, -9.2), (-5.3, -6.7), (-8.7, -6.4), (-7.1, -9.7), (-8, -6.3)]
}

# Plot the data
colors = {'ω1': 'r', 'ω2': 'g', 'ω3': 'b', 'ω4': 'y'}

plt.figure(figsize=(10, 8))

for cls, points in data.items():
    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1], c=colors[cls], label=cls)

plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('2D Data Points from Four Classes With different colors')
plt.grid(True)
plt.show()

class Perceptron:
   def __init__(self, learning_rate=0.01, n_iters=1000):
       self.lr = learning_rate
       self.n_iters = n_iters
       self.weights = None
       self.bias = None
       self.activation_func = self._unit_step_func
       self.n_iterations_ = 0
       self.misclassified_ = []

   def _unit_step_func(self, x):
       return np.where(x >= 0, 1, 0)
   
   def fit(self, X, y):
       n_samples, n_features = X.shape

       # initialization
       self.weights = np.zeros(n_features)
       self.bias = 0 
       self.n_iterations_ = 0

       y_ = np.array([1 if i > 0 else 0 for i in y])

       for _ in range(self.n_iters):
           errors = 0
           misclassified = []
           for index, x_i in enumerate(X):
               linear_output = np.dot(x_i, self.weights) + self.bias
               y_predicted = self.activation_func(linear_output)

               update = self.lr * (y_[index] - y_predicted)
               self.weights += update * x_i
               self.bias += update
               
               if update != 0.0:
                   errors += 1
                   misclassified.append((x_i, y[index]))

           self.n_iterations_ += 1
           self.misclassified_.append(misclassified)
           if errors == 0:
               break

       
def plot_decision_boundary(perceptron, X, y, ax, class1, class2, iteration):
    ax.scatter(X[:len(data[class1]), 0], X[:len(data[class1]), 1], color=colors[class1], label=class1)
    ax.scatter(X[len(data[class1]):, 0], X[len(data[class1]):, 1], color=colors[class2], label=class2)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1 = np.array([x0_1, x0_2])
    x2 = -(perceptron.weights[0] * x1 + perceptron.bias) / perceptron.weights[1]

    ax.plot(x1, x2, 'k--')

    misclassified = np.array([x[0] for x in perceptron.misclassified_[iteration]])
    misclassified_labels = np.array([x[1] for x in perceptron.misclassified_[iteration]])
    if misclassified.size > 0:
        misclassified_class1 = misclassified[misclassified_labels == 0]
        misclassified_class2 = misclassified[misclassified_labels == 1]
        if misclassified_class1.size > 0:
            ax.scatter(misclassified_class1[:, 0], misclassified_class1[:, 1], color=colors[class1], marker='x', s=100, label=f'Misclassified {class1}')
        if misclassified_class2.size > 0:
            ax.scatter(misclassified_class2[:, 0], misclassified_class2[:, 1], color=colors[class2], marker='x', s=100, label=f'Misclassified {class2}')
        
    ax.legend()
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(f'Decision Boundary for {class1} and {class2}')
    ax.grid(True)

class_pairs = [('ω1', 'ω2'), ('ω2', 'ω3'), ('ω3', 'ω4')]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (class1, class2) in enumerate(class_pairs):
    X = np.array(data[class1] + data[class2])
    y = np.array([0] * len(data[class1]) + [1] * len(data[class2]))

    perceptron = Perceptron(learning_rate=0.1, n_iters=1000)
    perceptron.fit(X, y)

    print(f'Number of iterations until convergence for {class1} and {class2}: {perceptron.n_iterations_}')

    plot_decision_boundary(perceptron, X, y, axes[i], class1, class2, perceptron.n_iterations_ - 1)

plt.tight_layout()
plt.show()