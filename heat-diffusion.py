import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# Define parameters
alpha = 0.1  # Thermal diffusivity
Lx, Ly = 10, 10  # Length of the domain in x and y
Nx, Ny = 100, 100  # Number of grid points in x and y
dx, dy = Lx / Nx, Ly / Ny  # Grid spacing

# Reduce dt to maintain stability (based on the CFL condition)
dt = min(0.25 * dx**2 / alpha, 0.25 * dy**2 / alpha)  # Adjusted time step size for stability

# Create a grid
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y)

# Initial condition: Narrower and taller Gaussian peak to maintain the same volume
u0 = 80 * np.exp(-((X)**2 + (Y)**2) / (2 * 0.5**2))  # Increased peak height, reduced spread

# Initialize the temperature grid
u = u0.copy()

# Create a custom colormap with blue to red gradient
colors = [(0.3, 0.3, 1), (1, 0.1, 0.1)]  # Blue to Red
n_bins = 100  # Discretizes the interpolation into 100 steps
custom_cmap = LinearSegmentedColormap.from_list("blue_red", colors, N=n_bins)

# Use LogNorm for color normalization, adjusting vmin to reduce sensitivity to small values
norm = LogNorm(vmin=0.1, vmax=40)  # Adjusted vmax to match the new peak value

# Create a figure and axis for the animation
fig, ax = plt.subplots()
cax = ax.imshow(u, cmap=custom_cmap, norm=norm, origin='lower', extent=(-Lx/2, Lx/2, -Ly/2, Ly/2))
fig.colorbar(cax)

# Function to update the heat distribution at each time step with reflective boundary conditions
def update_heat(u, dt, dx, dy, alpha):
    """
    Solves the 2D heat equation using a finite difference method.
    
    The heat equation in 2D is given by:
        ∂u/∂t = α ( ∂²u/∂x² + ∂²u/∂y² )
    
    Discretizing the Laplacian (∇²u = ∂²u/∂x² + ∂²u/∂y²) using central differences:
        ∂²u/∂x² ≈ (u[i+1, j] - 2u[i, j] + u[i-1, j]) / dx²
        ∂²u/∂y² ≈ (u[i, j+1] - 2u[i, j] + u[i, j-1]) / dy²
    
    Reflective boundary conditions are applied to prevent heat from leaving the domain.
    """
    # Apply reflective boundary conditions
    u[0, :] = u[1, :]     # Top boundary
    u[-1, :] = u[-2, :]   # Bottom boundary
    u[:, 0] = u[:, 1]     # Left boundary
    u[:, -1] = u[:, -2]   # Right boundary

    # Compute Laplacian using finite differences
    u_xx = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / dx**2
    u_yy = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / dy**2
    
    # Update the temperature field
    u_new = u + alpha * dt * (u_xx + u_yy)
    return u_new

# Animation update function
def animate(i):
    """
    Update function for the animation.
    
    Each frame in the animation shows the solution of the 2D heat equation
    after a time step `dt`.
    """
    global u
    # Run multiple updates per frame to speed up the process
    for _ in range(10):  # Increase the number of updates per frame
        u = update_heat(u, dt, dx, dy, alpha)
    cax.set_data(u)
    return cax,

# Create the animation for continuous live playback
ani = animation.FuncAnimation(fig, animate, interval=20, blit=True, repeat=True)

# Show the animation
plt.show()
