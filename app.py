import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from Problems import Problem
from AntsColony import Optimization

# Streamlit setup
st.set_page_config(page_title="Ant Colony Optimization Visualization", layout="wide")
st.title("Optymalizacja Kolonii Mrówek")

# Custom CSS for improved slider and text styling
st.sidebar.markdown("""
    <style>
    .stSlider > div > div > div {
        height: 25px !important;
    }
    .stSlider > div > div > div > div > div {
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# Problem parameters in two columns
problem_name = st.sidebar.selectbox("Problem", options=['att48', 'berlin52', 'ch150', 'rat99'])
col1, col2 = st.sidebar.columns(2)
with col1:
    num_iterations = st.slider("Iteracje", 10, 500, 100)
    evaporation_rate = st.slider("Parowanie", 0.0, 1.0, 0.3)
    alpha = st.slider("Feromony (α)", 0.1, 5.0, 1.0)
with col2:
    num_ants = st.slider("Mrówki", 5, 100, 20)
    pheromone_rate = st.slider("Feromony początkowe", 1, 500, 100)
    beta = st.slider("Heurystyka (β)", 0.1, 5.0, 2.0)

# Author section
st.sidebar.markdown("---")
st.sidebar.subheader("About the Author")
st.sidebar.image(
    "https://media.licdn.com/dms/image/v2/C5603AQHDzvS3uNwB4g/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1657832072141?e=2147483647&v=beta&t=JyOLpgqcrJa0oY4nEaGiZGhAUwDcnqOaO5SOpr9SwtU",
    width=150)
st.sidebar.markdown("""
    **Blazej Strus**  
    Data Scientist | Optimization & Visualization Enthusiast  
    Specializing in developing interactive visualization tools for optimization algorithms.

    📫 [b.strus@gmail.com](mailto:b.strus@gmail.com)   
    🌐 [LinkedIn](https://www.linkedin.com/in/b%C5%82a%C5%BCej-strus-7716192a/)
""")

# Problem setup
problem = Problem(
    problem_name=problem_name,
    num_ants=num_ants,
    alpha=alpha,
    beta=beta,
    evaporation_rate=evaporation_rate,
    pheromone_rate=pheromone_rate
)
optimization = Optimization(problem, num_iterations=num_iterations)

# Streamlit placeholders for dynamic elements
iteration_placeholder = st.empty()
best_route_placeholder = st.empty()
plot_pheromones = st.empty()
plot_convergence = st.empty()
best_route_lengths = []  # For storing convergence data

# Function to plot nodes, pheromone trails, and best route
def plot_route_and_pheromones(ax, best_route, pheromones):
    ax.clear()
    nodes = problem.nodes
    num_nodes = len(nodes)
    max_pheromone = pheromones.max() if pheromones.max() > 0 else 1
    cmap = cm.Reds

    # Plot pheromone trails with a threshold for performance
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            pheromone_level = pheromones[i, j] / max_pheromone
            if pheromone_level > 0.2:
                x_values = [nodes['x'].iloc[i], nodes['x'].iloc[j]]
                y_values = [nodes['y'].iloc[i], nodes['y'].iloc[j]]
                ax.plot(
                    x_values, y_values, color=cmap(pheromone_level),
                    linewidth=2 + pheromone_level * 5,
                    alpha=pheromone_level,
                    solid_capstyle='round'  # Rounded line ends
                )

    ax.scatter(nodes['x'], nodes['y'], c='orange', s=20, zorder=10)
    route_x = nodes['x'].iloc[best_route]
    route_y = nodes['y'].iloc[best_route]
    ax.plot(route_x, route_y, 'g-', linewidth=2, zorder=6)

# Callback function for visualization updates
def plot_callback(info):
    iteration = info['iteration']
    best_route = info['best_route']
    best_length = info['best_length']
    pheromones = optimization.colony.pheromones.trails

    # Calculate delta percentage
    delta_text = f"{best_length - problem.best_known_length:.0f}" if problem.best_known_length > 0 else "?"

    # Display iteration and route information
    iteration_placeholder.markdown(
        f"<div style='text-align: center; font-size: 1.2em;'>"
        f"Iteracja: <span style='color: #FFCC00;'>{iteration}/{num_iterations}</span> | "
        f"Aktualna trasa: <span style='color: #FFCC00;'>{best_length:.0f}</span> | "
        f"Najlepsza znana trasa: <span style='color: #FFCC00;'>{problem.best_known_length}</span> | "
        f"Δ: <span style='color: #FFCC00;'>{delta_text}</span></div>",
        unsafe_allow_html=True
    )

    # Update convergence plot
    best_route_lengths.append(best_length)
    fig2, ax2 = plt.subplots(figsize=(10, 3), dpi=100)
    ax2.plot(range(1, iteration + 1), best_route_lengths, marker='o', color='#76FF03', markersize=4, linestyle='-', linewidth=1)
    ax2.set_xlim(0, num_iterations)
    ax2.set_ylim(int(problem.best_known_length / 100) * 100, int(max(best_route_lengths) / 100) * 100 + 100)
    ax2.set_xlabel("Iteracje", color='white')
    ax2.set_ylabel("Najkrótsza trasa", color='white')
    ax2.grid(True, color='#424242')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.tick_params(colors='white')
    ax2.set_facecolor((0, 0, 0, 0))
    fig2.patch.set_alpha(0)
    plot_convergence.pyplot(fig2)

    # Update route and pheromone plot
    fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=100)
    plot_route_and_pheromones(ax1, best_route, pheromones)
    ax1.set_title("Najlepsza Trasa i Ścieżki Feromonowe", color='white', fontsize=10, pad=10)
    ax1.set_xlabel("X Współrzędne", color='white')
    ax1.set_ylabel("Y Współrzędne", color='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, color='#424242')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.set_facecolor((0, 0, 0, 0))
    fig1.patch.set_alpha(0)
    plot_pheromones.pyplot(fig1)

# Run optimization with callback
optimization.optimize(callback=plot_callback)
