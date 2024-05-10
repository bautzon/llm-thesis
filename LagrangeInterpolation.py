import numpy as np
import matplotlib.pyplot as plt

# Define the points
x_points = np.array([-9, -4, -1, 7])
y_points = np.array([5, 2, -2, 8])  # Example cubic function values: f(x) = x^3

# Define the Lagrange basis polynomial
def lagrange_basis(x, i, x_points):
    terms = [(x - x_points[j]) / (x_points[i] - x_points[j]) for j in range(len(x_points)) if j != i]
    return np.prod(terms, axis=0)

# Define the Lagrange interpolation polynomial
def lagrange_interpolation(x, x_points, y_points):
    return sum(y_points[i] * lagrange_basis(x, i, x_points) for i in range(len(x_points)))

# Generate x values for plotting
x_dense = np.linspace(min(x_points)-1, max(x_points)+1, 400)
y_dense = [lagrange_interpolation(x, x_points, y_points) for x in x_dense]

# Plotting
plt.figure(figsize=(12, 8))

# Plot each basis polynomial
colors = ['red', 'green', 'blue', 'purple']
for i in range(len(x_points)):
    y_basis = [lagrange_basis(x, i, x_points) * y_points[i] for x in x_dense]
    plt.plot(x_dense, y_basis, label=f'Basis Polynomial $\\ell_{i}(x)$ * $y_{i}$', linestyle='--', color=colors[i])

# Plot the Lagrange interpolation polynomial
plt.plot(x_dense, y_dense, label='Lagrange Interpolation Polynomial', color='black', linewidth=2)

# Mark the data points
plt.scatter(x_points, y_points, color='black', zorder=5)
plt.title('Lagrange Interpolation with Basis Polynomials', fontsize=25)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
