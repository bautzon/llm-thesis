import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.interpolate import CubicSpline

# Define the Runge function
def runge_function(x):
    return 1 / (1 + 25 * x**2)

# Generate equidistant points for different scenarios
x_points_full = np.linspace(-1, 1, 11)
y_points_full = runge_function(x_points_full)
x_points_removed = np.delete(x_points_full, 5)
y_points_removed = np.delete(y_points_full, 5)
x_points_fewer = np.linspace(-1, 1, 5)
y_points_fewer = runge_function(x_points_fewer)

# Dense x values for plotting
x_dense = np.linspace(-1, 1, 400)

# Interpolations
poly_full = lagrange(x_points_full, y_points_full)(x_dense)
spline_full = CubicSpline(x_points_full, y_points_full)(x_dense)
poly_removed = lagrange(x_points_removed, y_points_removed)(x_dense)
spline_removed = CubicSpline(x_points_removed, y_points_removed)(x_dense)
poly_fewer = lagrange(x_points_fewer, y_points_fewer)(x_dense)
spline_fewer = CubicSpline(x_points_fewer, y_points_fewer)(x_dense)

# Plotting all in one figure
plt.figure(figsize=(12, 8))
plt.plot(x_dense, runge_function(x_dense), label='Original Function', color='black')
plt.plot(x_dense, poly_full, label='Polynomial - 11 pts', color='red')
plt.plot(x_dense, spline_full, label='Spline - 11 pts', color='blue', linestyle=':')
#plt.plot(x_dense, poly_removed, label='Polynomial - 10 pts', color='red', linestyle='-.')
plt.plot(x_dense, spline_removed, label='Spline - 10 pts', color='purple', linestyle='-.')
plt.plot(x_dense, poly_fewer, label='Polynomial - 5 pts', color='green', linestyle=':')
#plt.plot(x_dense, spline_fewer, label='Spline - 5 pts', color='brown', linestyle='--')

# Marking data points
plt.scatter(x_points_full, y_points_full, color='gray', zorder=5)  # Original 11 points
plt.scatter(x_points_removed, y_points_removed, color='gray', zorder=5)  # With center point removed
plt.scatter(x_points_fewer, y_points_fewer, color='gray', zorder=5)  # Only 5 points

plt.title('Comparison of Polynomial and Spline Interpolations', fontsize=25)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
