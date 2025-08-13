"""
Wind Farm Cable Length Calculator

This script calculates the cable length needed to connect turbines in a wind farm
to substations, considering a rectangular grid layout with multiple substations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import math


def calculate_grid_dimensions(num_turbines: int) -> Tuple[int, int]:
    """
    Calculate optimal grid dimensions for a rectangular layout.
    
    Args:
        num_turbines: Total number of turbines
        
    Returns:
        Tuple of (rows, cols) for the grid layout
    """
    # Find dimensions that create a roughly square grid
    sqrt_n = math.sqrt(num_turbines)
    rows = int(sqrt_n)
    cols = math.ceil(num_turbines / rows)
    
    # Adjust to ensure we have exactly num_turbines positions
    while rows * cols < num_turbines:
        if rows <= cols:
            rows += 1
        else:
            cols += 1
    
    return rows, cols


def generate_turbine_positions(num_turbines: int, spacing_m: float) -> np.ndarray:
    """
    Generate turbine positions in a rectangular grid layout.
    
    Args:
        num_turbines: Number of turbines
        spacing_m: Spacing between turbines in meters
        
    Returns:
        Array of turbine positions (x, y) in meters
    """
    rows, cols = calculate_grid_dimensions(num_turbines)
    
    positions = []
    turbine_count = 0
    
    for i in range(rows):
        for j in range(cols):
            if turbine_count < num_turbines:
                x = j * spacing_m
                y = i * spacing_m
                positions.append([x, y])
                turbine_count += 1
    
    return np.array(positions)


def calculate_substation_positions(turbine_positions: np.ndarray, num_substations: int) -> np.ndarray:
    """
    Calculate optimal substation positions to minimize cable length.
    For 3 substations, we'll divide turbines into 3 groups and place substations at centroids.
    
    Args:
        turbine_positions: Array of turbine positions
        num_substations: Number of substations
        
    Returns:
        Array of substation positions
    """
    num_turbines = len(turbine_positions)
    turbines_per_substation = num_turbines // num_substations
    
    substation_positions = []
    
    for i in range(num_substations):
        start_idx = i * turbines_per_substation
        if i == num_substations - 1:  # Last substation gets remaining turbines
            end_idx = num_turbines
        else:
            end_idx = (i + 1) * turbines_per_substation
        
        # Get turbines for this substation
        group_turbines = turbine_positions[start_idx:end_idx]
        
        # Calculate centroid
        centroid_x = np.mean(group_turbines[:, 0])
        centroid_y = np.mean(group_turbines[:, 1])
        
        substation_positions.append([centroid_x, centroid_y])
    
    return np.array(substation_positions)


def assign_turbines_to_substations(turbine_positions: np.ndarray, 
                                 substation_positions: np.ndarray) -> List[int]:
    """
    Assign each turbine to the closest substation.
    
    Args:
        turbine_positions: Array of turbine positions
        substation_positions: Array of substation positions
        
    Returns:
        List of substation indices for each turbine
    """
    assignments = []
    
    for turbine_pos in turbine_positions:
        distances = []
        for substation_pos in substation_positions:
            distance = np.sqrt((turbine_pos[0] - substation_pos[0])**2 + 
                             (turbine_pos[1] - substation_pos[1])**2)
            distances.append(distance)
        
        closest_substation = np.argmin(distances)
        assignments.append(closest_substation)
    
    return assignments


def calculate_total_cable_length(turbine_positions: np.ndarray, 
                               substation_positions: np.ndarray,
                               assignments: List[int]) -> float:
    """
    Calculate total cable length from all turbines to their assigned substations.
    
    Args:
        turbine_positions: Array of turbine positions
        substation_positions: Array of substation positions
        assignments: List of substation assignments for each turbine
        
    Returns:
        Total cable length in meters
    """
    total_length = 0.0
    
    for i, turbine_pos in enumerate(turbine_positions):
        substation_idx = assignments[i]
        substation_pos = substation_positions[substation_idx]
        
        cable_length = np.sqrt((turbine_pos[0] - substation_pos[0])**2 + 
                              (turbine_pos[1] - substation_pos[1])**2)
        total_length += cable_length
    
    return total_length


def calculate_cable_costs(total_cable_length_m: float) -> dict:
    """
    Calculate cable costs for different cable scenarios.
    
    Args:
        total_cable_length_m: Total cable length in meters
        
    Returns:
        Dictionary with cost calculations for different scenarios
    """
    # Convert meters to feet for cost calculations
    total_cable_length_ft = total_cable_length_m * 3.28084
    
    # Cable costs per foot
    cable_costs_per_ft = {
        'LVDC': 120.0,  # $120/ft for LVDC
        'MVAC': 12.0,   # $12/ft for MVAC
        'MVDC': 9.0     # $9/ft for MVDC
    }
    
    cost_results = {}
    
    for cable_type, cost_per_ft in cable_costs_per_ft.items():
        # Calculate cable cost
        cable_cost = total_cable_length_ft * cost_per_ft
        
        # Calculate installation cost (2x cable cost)
        installation_cost = cable_cost
        
        # Calculate total cost
        total_cost = cable_cost + installation_cost
        
        cost_results[cable_type] = {
            'cable_cost_usd': cable_cost,
            'installation_cost_usd': installation_cost,
            'total_cost_usd': total_cost,
            'cost_per_ft_usd': cost_per_ft,
            'total_cable_length_ft': total_cable_length_ft
        }
    
    return cost_results


def plot_wind_farm(turbine_positions: np.ndarray, 
                   substation_positions: np.ndarray,
                   assignments: List[int],
                   turbine_diameter_m: float,
                   total_cable_length: float,
                   cable_costs: dict = None):
    """
    Create a matplotlib plot showing the wind farm layout.
    
    Args:
        turbine_positions: Array of turbine positions
        substation_positions: Array of substation positions
        assignments: List of substation assignments for each turbine
        turbine_diameter_m: Turbine diameter for scaling
        total_cable_length: Total cable length for title
        cable_costs: Dictionary with cable cost information
    """
    plt.figure(figsize=(10, 8))
    
    # Define colors for different substations
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Plot cables from turbines to substations
    for i, turbine_pos in enumerate(turbine_positions):
        substation_idx = assignments[i]
        substation_pos = substation_positions[substation_idx]
        
        plt.plot([turbine_pos[0], substation_pos[0]], 
                [turbine_pos[1], substation_pos[1]], 
                color=colors[substation_idx], alpha=0.3, linewidth=2.0)
    
    # Plot turbines
    for i, turbine_pos in enumerate(turbine_positions):
        substation_idx = assignments[i]
        plt.scatter(turbine_pos[0], turbine_pos[1], 
                   c=colors[substation_idx], s=30, marker='o', 
                   edgecolors='black', linewidth=0.5, alpha=0.7)
    
    # Plot substations
    for i, substation_pos in enumerate(substation_positions):
        plt.scatter(substation_pos[0], substation_pos[1], 
                   c=colors[i], s=100, marker='s', 
                   edgecolors='black', linewidth=2, label=f'Substation {i+1}')
    
    plt.xlabel('Distance (m)')
    plt.ylabel('Distance (m)')
    plt.title(f'Wind Farm Layout for 450MW Farm\nTotal Cable Length: {total_cable_length/1000:.2f} km')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add turbine count annotation
    info_text = f'Turbines: {len(turbine_positions)}'
    if cable_costs:
        info_text += f'\nCable Costs:\nLVDC: ${cable_costs["LVDC"]["total_cost_usd"]/1e6:.1f}M\nMVAC: ${cable_costs["MVAC"]["total_cost_usd"]/1e6:.1f}M\nMVDC: ${cable_costs["MVDC"]["total_cost_usd"]/1e6:.1f}M'
    
    plt.text(0.02, 0.98, info_text, 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=12)
    
    plt.tight_layout()
    plt.show()


def calculate_wind_farm_cables(num_turbines: int, 
                             turbine_rating_kw: float,
                             turbine_diameter_m: float, 
                             turbine_spacing_m: float,
                             num_substations: int = 3) -> dict:
    """
    Main function to calculate wind farm cable requirements.
    
    Args:
        num_turbines: Number of turbines in the wind farm
        turbine_rating_kw: Rating of each turbine in kW
        turbine_diameter_m: Diameter of each turbine in meters
        turbine_spacing_m: Spacing between turbines in meters
        num_substations: Number of substations (default 3)
        
    Returns:
        Dictionary with calculation results
    """
    print(f"Wind Farm Cable Calculator")
    print(f"=" * 40)
    print(f"Number of turbines: {num_turbines}")
    print(f"Turbine rating: {turbine_rating_kw} kW")
    print(f"Turbine diameter: {turbine_diameter_m} m")
    print(f"Turbine spacing: {turbine_spacing_m} m")
    print(f"Number of substations: {num_substations}")
    print()
    
    # Generate turbine positions
    turbine_positions = generate_turbine_positions(num_turbines, turbine_spacing_m)
    
    # Calculate substation positions
    substation_positions = calculate_substation_positions(turbine_positions, num_substations)
    
    # Assign turbines to closest substations
    assignments = assign_turbines_to_substations(turbine_positions, substation_positions)
    
    # Calculate total cable length
    total_cable_length = calculate_total_cable_length(turbine_positions, 
                                                    substation_positions, 
                                                    assignments)
    
    # Calculate cable costs for different scenarios
    cable_costs = calculate_cable_costs(total_cable_length)
    
    # Print results
    print(f"Grid dimensions: {calculate_grid_dimensions(num_turbines)}")
    print(f"Total cable length: {total_cable_length:.2f} m ({total_cable_length/1000:.2f} km)")
    print(f"Total cable length: {cable_costs['LVDC']['total_cable_length_ft']:.2f} ft")
    print(f"Average cable length per turbine: {total_cable_length/num_turbines:.2f} m")
    print()
    
    # Print cost analysis
    print("Cable Cost Analysis:")
    print("=" * 50)
    for cable_type, costs in cable_costs.items():
        print(f"{cable_type} Cable System:")
        print(f"  Cable cost:        ${costs['cable_cost_usd']:,.2f}")
        print(f"  Installation cost: ${costs['installation_cost_usd']:,.2f}")
        print(f"  Total cost:        ${costs['total_cost_usd']:,.2f}")
        print(f"  Cost per turbine:  ${costs['total_cost_usd']/num_turbines:,.2f}")
        print()
    
    # Print substation details
    print("Substation Details:")
    print("=" * 30)
    for i in range(num_substations):
        turbines_in_group = assignments.count(i)
        print(f"Substation {i+1}: {turbines_in_group} turbines assigned")
        print(f"  Position: ({substation_positions[i][0]:.1f}, {substation_positions[i][1]:.1f}) m")
    
    # Create plot
    plot_wind_farm(turbine_positions, substation_positions, assignments, 
                   turbine_diameter_m, total_cable_length, cable_costs)
    
    return {
        'total_cable_length_m': total_cable_length,
        'total_cable_length_km': total_cable_length / 1000,
        'total_cable_length_ft': cable_costs['LVDC']['total_cable_length_ft'],
        'average_cable_length_per_turbine_m': total_cable_length / num_turbines,
        'cable_costs': cable_costs,
        'turbine_positions': turbine_positions,
        'substation_positions': substation_positions,
        'assignments': assignments
    }


if __name__ == "__main__":
    # Default parameters (can be modified)
    num_turbines = 90
    turbine_rating_kw = 5000  # From the HOPP config file
    turbine_diameter_m = 126  # Typical for 5 MW turbine
    turbine_spacing_m = 7 * turbine_diameter_m  # 7D spacing is typical
    
    print(f"Using turbine spacing of {turbine_spacing_m} m (7 times diameter)")
    print()
    
    # Calculate cable requirements
    results = calculate_wind_farm_cables(
        num_turbines=num_turbines,
        turbine_rating_kw=turbine_rating_kw,
        turbine_diameter_m=turbine_diameter_m,
        turbine_spacing_m=turbine_spacing_m,
        num_substations=3
    )
