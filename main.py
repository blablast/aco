import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Problems import Problem
from AntsColony import Optimization

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm  # For color gradient

background_color = '#2E2E2E'

# Set up Tkinter window with a dark background
root = tk.Tk()
root.title("Optymalizacja Kolonii Mrówek - Wizualizacja")
root.configure(bg=background_color)  # Dark background for the window

# Set up the problem instance
problem = Problem(problem_name='berlin52', num_ants=50, alpha=1.0, beta=2.0, evaporation_rate=0.3, pheromone_rate=100.0)
optimization = Optimization(problem, num_iterations=200)

# Create separate frames for text and plot elements
top_frame = tk.Frame(root, bg=background_color)
top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

bottom_frame = tk.Frame(root, bg=background_color)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(10, 20))

# Display the name of the problem at the top
problem_label = tk.Label(top_frame, text=f"Problem: {problem.name}", font=("Arial", 16), fg="white", bg=background_color)
problem_label.pack(pady=5)

# Display current iteration and route length with difference from optimal
iteration_label = tk.Label(top_frame, text="Iteracja: 0/100", font=("Arial", 16), fg="white", bg=background_color)
iteration_label.pack(pady=5)

best_route_label = tk.Label(top_frame, text="Trasa: 0 | ⨉ Najlepsza: ? | Δ: ?%", font=("Arial", 14), fg="white", bg=background_color)
best_route_label.pack(pady=5)

# Prepare the figure for the route and pheromone trail plot (Top plot)
fig1 = plt.Figure(figsize=(6, 3), dpi=100)
ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])  # Add margins for visibility
ax1.set_title("Najlepsza Trasa i Ścieżki Feromonowe", color='white', pad=10)
ax1.set_xlabel("X Współrzędne", color='white')
ax1.set_ylabel("Y Współrzędne", color='white')
ax1.tick_params(colors='white')
ax1.set_facecolor("#2E2E2E")
plt.tight_layout()
fig1.patch.set_facecolor(background_color)


# Embed the first figure in the top frame
canvas1 = FigureCanvasTkAgg(fig1, master=top_frame)
canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
canvas1.get_tk_widget().configure(bg=background_color)

# Prepare the figure for the convergence plot (Bottom plot)
fig2 = plt.Figure(figsize=(6, 1), dpi=100)
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])  # Add margins for visibility
line1, = ax2.plot([], [], marker='o', color='#76FF03', markersize=4, linestyle='-', linewidth=1)
ax2.set_xlim(0, optimization.num_iterations)
ax2.set_xlabel("Iteracje", color='white')
ax2.set_ylabel("Najkrótsza trasa", color='white')
ax2.set_title("Zbieżność ACO - Najkrótsza długość trasy w kolejnych iteracjach", color='white', pad=10)
ax2.tick_params(colors='white')
ax2.grid(True, color='#424242')
ax2.set_facecolor("#2E2E2E")
plt.tight_layout()
fig2.patch.set_facecolor(background_color)

# Embed the second figure in the bottom frame
canvas2 = FigureCanvasTkAgg(fig2, master=bottom_frame)
canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
canvas2.get_tk_widget().configure(bg=background_color)

# Initialize lists to store data for the convergence plot
best_route_lengths = []

# Function to plot nodes, pheromone trails, and best route in each iteration
def plot_route_and_pheromones(ax, best_route, pheromones):
    ax.clear()
    nodes = problem.nodes

    # Plot pheromone trails with threshold
    num_nodes = len(nodes)
    max_pheromone = pheromones.max() if pheromones.max() > 0 else 1
    cmap = cm.Reds

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            pheromone_level = pheromones[i, j] / max_pheromone
            if pheromone_level > 0.2:
                x_values = [nodes['x'].iloc[i], nodes['x'].iloc[j]]
                y_values = [nodes['y'].iloc[i], nodes['y'].iloc[j]]
                ax.plot(
                    x_values, y_values, color=cmap(pheromone_level),
                    linewidth=2 + pheromone_level * 5,
                    alpha=pheromone_level
                )

    ax.scatter(nodes['x'], nodes['y'], c='orange', s=20, zorder=10)
    route_x = nodes['x'].iloc[best_route]
    route_y = nodes['y'].iloc[best_route]
    ax.plot(route_x, route_y, 'g-', linewidth=2, zorder=6)

# Define callback function for real-time updates
def plot_callback(info):
    iteration = info['iteration']
    best_route = info['best_route']
    best_length = info['best_length']
    pheromones = optimization.colony.pheromones.trails

    # Update iteration and best route labels
    iteration_label.config(text=f"Iteracja: {iteration}/{optimization.num_iterations}")
    delta_percentage = ((best_length - problem.best_known_length) / problem.best_known_length) * 100 if problem.best_known_length else None
    delta_text = f"{delta_percentage:.1f}%" if delta_percentage is not None else "?"
    best_route_label.config(text=f"Trasa: {best_length:.0f} | ⨉ Najlepsza: {problem.best_known_length} | Δ: {delta_text}")

    best_route_lengths.append(best_length)
    line1.set_data(range(1, iteration + 1), best_route_lengths)
    ax2.set_ylim(int(problem.best_known_length / 100) * 100, max(best_route_lengths) * 1.1)
    canvas2.draw()

    plot_route_and_pheromones(ax1, best_route, pheromones)
    canvas1.draw()

    root.update_idletasks()
    root.update()

# Run optimization with callback
optimization.optimize(callback=plot_callback)

# Start Tkinter main loop
root.mainloop()