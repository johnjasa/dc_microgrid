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


def calculate_installation_cost_multiplier(num_cables: int) -> float:
    """
    Calculate installation cost multiplier based on number of overlapping cables.
    Uses diminishing returns: 1 cable = 1.0x, 2 = 1.6x, 3 = 2.0x, 4 = 2.2x, etc.
    
    Args:
        num_cables: Number of cables in the bundle
        
    Returns:
        Installation cost multiplier
    """
    if num_cables <= 1:
        return 1.0
    elif num_cables == 2:
        return 1.6
    elif num_cables == 3:
        return 2.0
    elif num_cables == 4:
        return 2.2
    else:
        # For 5+ cables, continue with diminishing returns
        return 2.2 + (num_cables - 4) * 0.1


def calculate_cable_costs_radial(total_cable_length_m: float) -> dict:
    """
    Calculate cable costs for radial cable layout (original method).
    
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
    
    # Installation costs per foot (base cost for single cable)
    installation_costs_per_ft = {
        'LVDC': 120.0,  # $120/ft for LVDC installation
        'MVAC': 12.0,   # $12/ft for MVAC installation
        'MVDC': 9.0     # $9/ft for MVDC installation
    }
    
    cost_results = {}
    
    for cable_type, cost_per_ft in cable_costs_per_ft.items():
        # Calculate cable cost
        cable_cost = total_cable_length_ft * cost_per_ft
        
        # Calculate installation cost (equal to cable cost for radial)
        installation_cost = total_cable_length_ft * installation_costs_per_ft[cable_type]
        
        # Calculate total cost
        total_cost = cable_cost + installation_cost
        
        cost_results[cable_type] = {
            'cable_cost_usd': cable_cost,
            'installation_cost_usd': installation_cost,
            'total_cost_usd': total_cost,
            'cost_per_ft_usd': cost_per_ft,
            'installation_cost_per_ft_usd': installation_costs_per_ft[cable_type],
            'total_cable_length_ft': total_cable_length_ft
        }
    
    return cost_results


def calculate_cable_costs_branched(cable_segments: List[dict]) -> dict:
    """
    Calculate cable costs for branched cable layout with bundling benefits.
    
    Args:
        cable_segments: List of cable segments with length and cable count
        
    Returns:
        Dictionary with cost calculations for different scenarios
    """
    # Cable costs per foot
    cable_costs_per_ft = {
        'LVDC': 120.0,  # $120/ft for LVDC
        'MVAC': 12.0,   # $12/ft for MVAC
        'MVDC': 9.0     # $9/ft for MVDC
    }
    
    # Installation costs per foot (base cost for single cable)
    installation_costs_per_ft = {
        'LVDC': 120.0,  # $120/ft for LVDC installation
        'MVAC': 12.0,   # $12/ft for MVAC installation
        'MVDC': 9.0     # $9/ft for MVDC installation
    }
    
    cost_results = {}
    
    for cable_type, cost_per_ft in cable_costs_per_ft.items():
        cable_cost = 0.0
        installation_cost = 0.0
        total_cable_length_ft = 0.0
        
        for segment in cable_segments:
            length_ft = segment['length_m'] * 3.28084
            num_cables = segment['cable_count']
            
            # Cable cost scales linearly with number of cables
            segment_cable_cost = length_ft * cost_per_ft * num_cables
            
            # Installation cost uses bundling multiplier
            multiplier = calculate_installation_cost_multiplier(num_cables)
            segment_installation_cost = length_ft * installation_costs_per_ft[cable_type] * multiplier
            
            cable_cost += segment_cable_cost
            installation_cost += segment_installation_cost
            total_cable_length_ft += length_ft * num_cables
        
        # Calculate total cost
        total_cost = cable_cost + installation_cost
        
        cost_results[cable_type] = {
            'cable_cost_usd': cable_cost,
            'installation_cost_usd': installation_cost,
            'total_cost_usd': total_cost,
            'cost_per_ft_usd': cost_per_ft,
            'installation_cost_per_ft_usd': installation_costs_per_ft[cable_type],
            'total_cable_length_ft': total_cable_length_ft,
            'cable_segments': cable_segments
        }
    
    return cost_results


def create_branched_cable_layout(turbine_positions: np.ndarray, 
                               substation_positions: np.ndarray,
                               assignments: List[int]) -> List[dict]:
    """
    Create a branched cable layout where cables go from substation through turbines.
    
    Args:
        turbine_positions: Array of turbine positions
        substation_positions: Array of substation positions
        assignments: List of substation assignments for each turbine
        
    Returns:
        List of cable segments with their properties
    """
    cable_segments = []
    
    # Group turbines by substation
    substation_groups = {}
    for i, assignment in enumerate(assignments):
        if assignment not in substation_groups:
            substation_groups[assignment] = []
        substation_groups[assignment].append(i)
    
    # For each substation, create a branched network using minimum spanning tree
    for substation_idx, turbine_indices in substation_groups.items():
        substation_pos = substation_positions[substation_idx]
        
        if len(turbine_indices) == 0:
            continue
            
        # Get positions of turbines assigned to this substation
        group_turbines = [(i, turbine_positions[i]) for i in turbine_indices]
        
        # Create minimum spanning tree from substation to all turbines
        remaining_turbines = group_turbines.copy()
        connected_turbines = set()
        
        # Start by connecting the closest turbine to the substation
        min_distance = float('inf')
        closest_turbine = None
        
        for turbine_idx, turbine_pos in remaining_turbines:
            distance = np.sqrt((turbine_pos[0] - substation_pos[0])**2 + 
                             (turbine_pos[1] - substation_pos[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_turbine = (turbine_idx, turbine_pos)
        
        # Connect first turbine to substation
        turbine_idx, turbine_pos = closest_turbine
        cable_segments.append({
            'from_idx': -1,  # -1 indicates substation
            'to_idx': turbine_idx,
            'from_pos': substation_pos,
            'to_pos': turbine_pos,
            'length_m': min_distance,
            'cable_count': len(remaining_turbines),  # All turbines route through this connection
            'substation_idx': substation_idx
        })
        
        connected_turbines.add(turbine_idx)
        remaining_turbines.remove(closest_turbine)
        
        # Connect remaining turbines using minimum spanning tree approach
        while remaining_turbines:
            min_distance = float('inf')
            closest_connection = None
            
            # Find the closest remaining turbine to any connected turbine
            for turbine_idx, turbine_pos in remaining_turbines:
                # Check distance to substation
                dist_to_substation = np.sqrt((turbine_pos[0] - substation_pos[0])**2 + 
                                           (turbine_pos[1] - substation_pos[1])**2)
                
                # Check distance to each connected turbine
                for connected_idx in connected_turbines:
                    connected_pos = turbine_positions[connected_idx]
                    dist_to_connected = np.sqrt((turbine_pos[0] - connected_pos[0])**2 + 
                                              (turbine_pos[1] - connected_pos[1])**2)
                    
                    # Choose the shorter connection
                    if dist_to_connected < dist_to_substation and dist_to_connected < min_distance:
                        min_distance = dist_to_connected
                        closest_connection = {
                            'turbine': (turbine_idx, turbine_pos),
                            'connect_to': (connected_idx, connected_pos),
                            'distance': dist_to_connected
                        }
                    elif dist_to_substation < min_distance and dist_to_connected >= dist_to_substation:
                        min_distance = dist_to_substation
                        closest_connection = {
                            'turbine': (turbine_idx, turbine_pos),
                            'connect_to': (-1, substation_pos),
                            'distance': dist_to_substation
                        }
            
            # Add the closest connection
            if closest_connection:
                turbine_idx, turbine_pos = closest_connection['turbine']
                connect_idx, connect_pos = closest_connection['connect_to']
                
                # Cable count is number of remaining turbines (including this one)
                cable_count = len(remaining_turbines)
                
                cable_segments.append({
                    'from_idx': connect_idx,
                    'to_idx': turbine_idx,
                    'from_pos': connect_pos,
                    'to_pos': turbine_pos,
                    'length_m': closest_connection['distance'],
                    'cable_count': cable_count,
                    'substation_idx': substation_idx
                })
                
                connected_turbines.add(turbine_idx)
                remaining_turbines.remove(closest_connection['turbine'])
    
    return cable_segments


def calculate_total_cable_length_branched(cable_segments: List[dict]) -> float:
    """
    Calculate total cable length for branched layout.
    
    Args:
        cable_segments: List of cable segments
        
    Returns:
        Total cable length in meters (counting multiple cables)
    """
    total_length = 0.0
    for segment in cable_segments:
        total_length += segment['length_m'] * segment['cable_count']
    return total_length
def plot_wind_farm(turbine_positions: np.ndarray, 
                   substation_positions: np.ndarray,
                   assignments: List[int],
                   turbine_diameter_m: float,
                   total_cable_length: float,
                   cable_costs: dict = None,
                   cable_segments: List[dict] = None,
                   layout_type: str = "radial"):
    """
    Create a matplotlib plot showing the wind farm layout.
    
    Args:
        turbine_positions: Array of turbine positions
        substation_positions: Array of substation positions
        assignments: List of substation assignments for each turbine
        turbine_diameter_m: Turbine diameter for scaling
        total_cable_length: Total cable length for title
        cable_costs: Dictionary with cable cost information
        cable_segments: List of cable segments for branched layout
        layout_type: "radial" or "branched"
    """
    plt.figure(figsize=(12, 10))
    
    # Define colors for different substations
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    if layout_type == "radial" or cable_segments is None:
        # Plot cables from turbines to substations (radial)
        for i, turbine_pos in enumerate(turbine_positions):
            substation_idx = assignments[i]
            substation_pos = substation_positions[substation_idx]
            
            plt.plot([turbine_pos[0], substation_pos[0]], 
                    [turbine_pos[1], substation_pos[1]], 
                    color=colors[substation_idx], alpha=0.3, linewidth=2.0)
    else:
        # Plot branched cable layout
        for segment in cable_segments:
            from_pos = segment['from_pos']
            to_pos = segment['to_pos']
            cable_count = segment['cable_count']
            substation_idx = segment['substation_idx']
            
            # Line thickness based on number of cables
            linewidth = 1.0 + cable_count * 0.5
            alpha = 0.4 + min(cable_count * 0.1, 0.4)
            
            plt.plot([from_pos[0], to_pos[0]], 
                    [from_pos[1], to_pos[1]], 
                    color=colors[substation_idx], 
                    alpha=alpha, 
                    linewidth=linewidth,
                    label=f'{cable_count} cables' if cable_count > 1 else None)
    
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
    plt.title(f'Wind Farm Layout for 450MW Farm ({layout_type.title()} Layout)\nTotal Cable Length: {total_cable_length/1000:.2f} km')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add turbine count annotation
    info_text = f'Turbines: {len(turbine_positions)}\nLayout: {layout_type.title()}'
    if cable_costs:
        info_text += f'\nCable Costs:\nLVDC: ${cable_costs["LVDC"]["total_cost_usd"]/1e6:.1f}M\nMVAC: ${cable_costs["MVAC"]["total_cost_usd"]/1e6:.1f}M\nMVDC: ${cable_costs["MVDC"]["total_cost_usd"]/1e6:.1f}M'
    
    plt.text(0.02, 0.98, info_text, 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=11)
    
    plt.tight_layout()
    filename = f"wind_farm_layout_{layout_type}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as {filename}")


def calculate_wind_farm_cables(num_turbines: int, 
                             turbine_rating_kw: float,
                             turbine_diameter_m: float, 
                             turbine_spacing_m: float,
                             num_substations: int = 3,
                             layout_type: str = "radial") -> dict:
    """
    Main function to calculate wind farm cable requirements.
    
    Args:
        num_turbines: Number of turbines in the wind farm
        turbine_rating_kw: Rating of each turbine in kW
        turbine_diameter_m: Diameter of each turbine in meters
        turbine_spacing_m: Spacing between turbines in meters
        num_substations: Number of substations (default 3)
        layout_type: Cable layout type - "radial" or "branched"
        
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
    print(f"Layout type: {layout_type}")
    print()
    
    # Generate turbine positions
    turbine_positions = generate_turbine_positions(num_turbines, turbine_spacing_m)
    
    # Calculate substation positions
    substation_positions = calculate_substation_positions(turbine_positions, num_substations)
    
    # Assign turbines to closest substations
    assignments = assign_turbines_to_substations(turbine_positions, substation_positions)
    
    # Calculate cable layout and costs based on type
    if layout_type.lower() == "branched":
        # Create branched cable layout
        cable_segments = create_branched_cable_layout(turbine_positions, 
                                                    substation_positions, 
                                                    assignments)
        
        # Calculate total cable length
        total_cable_length = calculate_total_cable_length_branched(cable_segments)
        
        # Calculate cable costs for branched layout
        cable_costs = calculate_cable_costs_branched(cable_segments)
        
        # Print branched layout details
        print("Branched Cable Layout:")
        print("=" * 30)
        print(f"Number of cable segments: {len(cable_segments)}")
        
        # Print segment details
        for i, segment in enumerate(cable_segments):
            print(f"Segment {i+1}: {segment['length_m']:.1f}m, {segment['cable_count']} cables")
            
    else:
        # Use radial layout (original)
        cable_segments = None
        
        # Calculate total cable length
        total_cable_length = calculate_total_cable_length(turbine_positions, 
                                                        substation_positions, 
                                                        assignments)
        
        # Calculate cable costs for radial layout
        cable_costs = calculate_cable_costs_radial(total_cable_length)
    
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
                   turbine_diameter_m, total_cable_length, cable_costs,
                   cable_segments, layout_type)
    
    return {
        'total_cable_length_m': total_cable_length,
        'total_cable_length_km': total_cable_length / 1000,
        'total_cable_length_ft': cable_costs['LVDC']['total_cable_length_ft'],
        'average_cable_length_per_turbine_m': total_cable_length / num_turbines,
        'cable_costs': cable_costs,
        'turbine_positions': turbine_positions,
        'substation_positions': substation_positions,
        'assignments': assignments,
        'layout_type': layout_type,
        'cable_segments': cable_segments
    }


if __name__ == "__main__":
    # Default parameters (can be modified)
    num_turbines = 90
    turbine_rating_kw = 5000  # From the HOPP config file
    turbine_diameter_m = 126  # Typical for 5 MW turbine
    turbine_spacing_m = 7 * turbine_diameter_m  # 7D spacing is typical
    
    print(f"Using turbine spacing of {turbine_spacing_m} m (7 times diameter)")
    print()
    
    # Calculate cable requirements for both layouts
    print("RADIAL LAYOUT ANALYSIS")
    print("=" * 60)
    results_radial = calculate_wind_farm_cables(
        num_turbines=num_turbines,
        turbine_rating_kw=turbine_rating_kw,
        turbine_diameter_m=turbine_diameter_m,
        turbine_spacing_m=turbine_spacing_m,
        num_substations=3,
        layout_type="radial"
    )
    
    print("\n" + "=" * 60)
    print("BRANCHED LAYOUT ANALYSIS")
    print("=" * 60)
    results_branched = calculate_wind_farm_cables(
        num_turbines=num_turbines,
        turbine_rating_kw=turbine_rating_kw,
        turbine_diameter_m=turbine_diameter_m,
        turbine_spacing_m=turbine_spacing_m,
        num_substations=3,
        layout_type="branched"
    )
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON ANALYSIS")
    print("=" * 60)
    
    print(f"Cable Length Comparison:")
    print(f"  Radial layout:   {results_radial['total_cable_length_km']:.2f} km")
    print(f"  Branched layout: {results_branched['total_cable_length_km']:.2f} km")
    print(f"  Difference:      {(results_branched['total_cable_length_km'] - results_radial['total_cable_length_km']):.2f} km")
    print(f"  Savings:         {(1 - results_branched['total_cable_length_km']/results_radial['total_cable_length_km'])*100:.1f}%")
    print()
    
    print("Cost Comparison (MVDC):")
    radial_cost = results_radial['cable_costs']['MVDC']['total_cost_usd']
    branched_cost = results_branched['cable_costs']['MVDC']['total_cost_usd']
    print(f"  Radial layout:   ${radial_cost/1e6:.2f}M")
    print(f"  Branched layout: ${branched_cost/1e6:.2f}M")
    print(f"  Savings:         ${(radial_cost - branched_cost)/1e6:.2f}M ({(1 - branched_cost/radial_cost)*100:.1f}%)")
    print()
    
    print("Installation Cost Benefits:")
    radial_install = results_radial['cable_costs']['MVDC']['installation_cost_usd']
    branched_install = results_branched['cable_costs']['MVDC']['installation_cost_usd']
    print(f"  Radial installation:   ${radial_install/1e6:.2f}M")
    print(f"  Branched installation: ${branched_install/1e6:.2f}M")
    print(f"  Installation savings:  ${(radial_install - branched_install)/1e6:.2f}M ({(1 - branched_install/radial_install)*100:.1f}%)")
