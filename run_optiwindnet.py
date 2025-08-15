from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om

import pytest

optiwindnet = pytest.importorskip("optiwindnet")

import ard
import ard.utils.io
import ard.collection.optiwindnet_wrap as ard_own


def make_modeling_options(x_turbines, y_turbines, x_substations, y_substations):

    # specify the configuration/specification files to use
    filename_turbine_spec = (
        Path(ard.__file__).parents[1]
        / "examples"
        / "data"
        / "turbine_spec_IEA-3p4-130-RWT.yaml"
    )  # toolset generalized turbine specification
    data_turbine_spec = ard.utils.io.load_turbine_spec(filename_turbine_spec)

    # set up the modeling options
    N_turbines = len(x_turbines)
    N_substations = len(x_substations)
    modeling_options = {
        "farm": {
            "N_turbines": N_turbines,
            "N_substations": N_substations,
            "x_substations": x_substations,
            "y_substations": y_substations,
            "x_turbines": x_turbines,
            "y_turbines": y_turbines,
        },
        "turbine": data_turbine_spec,
        "collection": {
            "max_turbines_per_string": 8,
            "model_options": dict(
                topology="branched",
                feeder_route="segmented",
                feeder_limit="unlimited",
            ),
            "solver_name": "highs",
            "solver_options": dict(
                time_limit=10,
                mip_gap=0.005,  # TODO ???
            ),
        },
    }

    return modeling_options


# create the farm layout specification
n_rows = 9
n_cols = 10
n_turbines = n_rows * n_cols  # 90 turbines

x_spacing = 130.0 * 7
y_spacing = 130.0 * 7

x_grid = np.linspace(-(n_cols - 1) / 2, (n_cols - 1) / 2, n_cols)
y_grid = np.linspace(-(n_rows - 1) / 2, (n_rows - 1) / 2, n_rows)
xv, yv = np.meshgrid(x_grid, y_grid)
x_turbines = (xv.flatten() * x_spacing).astype(np.float64)
y_turbines = (yv.flatten() * y_spacing).astype(np.float64)

substation_spacing = 3000.
angles_deg = np.arange(0, 360, 72)
angles_rad = np.deg2rad(angles_deg)
x_substations = (substation_spacing * np.cos(angles_rad)).astype(np.float64)
y_substations = (substation_spacing * np.sin(angles_rad)).astype(np.float64)

modeling_options = make_modeling_options(
    x_turbines=x_turbines,
    y_turbines=y_turbines,
    x_substations=x_substations,
    y_substations=y_substations,
)

# create the OpenMDAO model
model = om.Group()
optiwindnet_coll = model.add_subsystem(
    "optiwindnet_coll",
    ard_own.OptiwindnetCollection(
        modeling_options=modeling_options,
    ),
    promotes=["*"],
)

prob = om.Problem(model)
prob.setup()
prob.run_model()

prob.model.list_outputs(print_arrays=True, units=True)

# Get the graph object from the collection system for analysis
try:
    graph = optiwindnet_coll.graph
    vertex_coords = graph.graph["VertexC"]
except Exception as e:
    print(f"Could not access graph data: {e}")
    graph = None
    vertex_coords = None

# Cable cost calculations and discrete cable sizing
# Convert reference costs to cost per foot for comparison with base costs
meters_to_feet = 3.28084

# Base costs for 3 turbines (given), include installation factor
base_cable_costs_per_ft = {
    'LVDC': 120.0 * 3 * 2.0,  # $120/ft for LVDC
    'MVAC': 12.0 * 3 * 2.0,   # $12/ft for MVAC
    'MVDC': 9.0 * 3 * 2.0,     # $9/ft for MVDC
}

# Discrete cable types with scaling based on reference designs
# Using relative scaling from ref_cable_costs_per_m
ref_cable_costs_per_m = {
    'XLPE_185' : {'max_turbines_supplied' : 3, 'eur_cost_per_m': 369.},
    'XLPE_400' : {'max_turbines_supplied' : 5, 'eur_cost_per_m': 429.},
    'XLPE_1000' : {'max_turbines_supplied' : 7, 'eur_cost_per_m': 737.},
    'XLPE_2000' : {'max_turbines_supplied' : 9, 'eur_cost_per_m': 1050.},
}

# Calculate scaling factors relative to 3-turbine cable (XLPE_185)
base_ref_cost = ref_cable_costs_per_m['XLPE_185']['eur_cost_per_m']
cable_scaling_factors = {}
for ref_type, ref_data in ref_cable_costs_per_m.items():
    scaling_factor = ref_data['eur_cost_per_m'] / base_ref_cost
    cable_scaling_factors[ref_data['max_turbines_supplied']] = {
        'factor': scaling_factor,
        'ref_type': ref_type
    }

def select_cable_type(turbine_load):
    """Select appropriate discrete cable type based on load"""
    # Find the smallest cable that can handle the load
    for capacity in sorted(cable_scaling_factors.keys()):
        if turbine_load <= capacity:
            return capacity, cable_scaling_factors[capacity]
    # If load exceeds largest cable, use the largest available
    max_capacity = max(cable_scaling_factors.keys())
    return max_capacity, cable_scaling_factors[max_capacity]

# Calculate costs for each scenario and collect cable sizing information
edge_cable_info = {}  # Store cable type info for each edge
total_cable_costs = {}

if graph is not None:
    for cable_scenario, base_cost_per_ft in base_cable_costs_per_ft.items():
        total_cost = 0.0
        total_length_ft = 0.0
        cable_type_usage = {}  # Track usage of each discrete cable type
        
        for u, v, edge_data in graph.edges(data=True):
            # Get cable length from edge data or calculate from coordinates
            if 'length' in edge_data:
                cable_length_m = edge_data['length']
            else:
                # Calculate length from coordinates if not in edge data
                coord_u = vertex_coords[u]
                coord_v = vertex_coords[v]
                cable_length_m = np.sqrt((coord_u[0] - coord_v[0])**2 + (coord_u[1] - coord_v[1])**2)
            
            cable_length_ft = cable_length_m * meters_to_feet
            turbine_load = edge_data.get('load', 1)
            
            # Select discrete cable type
            cable_capacity, cable_info = select_cable_type(turbine_load)
            scaling_factor = cable_info['factor']
            ref_type = cable_info['ref_type']
            
            # Store cable info for plotting
            edge_key = (u, v)
            if edge_key not in edge_cable_info:
                edge_cable_info[edge_key] = {}
            edge_cable_info[edge_key][cable_scenario] = {
                'capacity': cable_capacity,
                'ref_type': ref_type,
                'load': turbine_load,
                'length_ft': cable_length_ft
            }
            
            # Calculate cost with discrete cable sizing
            cable_cost = cable_length_ft * base_cost_per_ft * scaling_factor
            total_cost += cable_cost
            total_length_ft += cable_length_ft
            
            # Track cable type usage
            if ref_type not in cable_type_usage:
                cable_type_usage[ref_type] = {'length': 0, 'count': 0}
            cable_type_usage[ref_type]['length'] += cable_length_ft
            cable_type_usage[ref_type]['count'] += 1
        
        total_cable_costs[cable_scenario] = {
            'total_cost': total_cost,
            'total_length_ft': total_length_ft,
            'cable_type_usage': cable_type_usage
        }

# Plot the turbine layout and cable connections
fig, ax = plt.subplots(figsize=(14, 10))

# Plot turbines
ax.scatter(x_turbines, y_turbines, c='blue', s=50, label='Wind Turbines')
for idx, (x, y) in enumerate(zip(x_turbines, y_turbines)):
    ax.text(x + 50, y + 50, str(idx), ha="left", va="bottom", fontsize=8)

# Plot substations
ax.scatter(x_substations, y_substations, c='red', s=100, marker='s', label='Substations')
for idx, (x, y) in enumerate(zip(x_substations, y_substations)):
    ax.text(x + 100, y + 100, f'S{idx}', ha="left", va="bottom", fontsize=10, fontweight='bold')

# Plot cables with discrete sizing (using MVAC scenario for visualization)
if graph is not None and edge_cable_info:
    # Define colors and line styles for different cable types
    cable_type_styles = {
        'XLPE_185': {'color': 'green', 'linewidth': 1.5, 'linestyle': '-', 'label': '≤3 turbines'},
        'XLPE_400': {'color': 'orange', 'linewidth': 2.5, 'linestyle': '-', 'label': '≤5 turbines'},
        'XLPE_1000': {'color': 'purple', 'linewidth': 3.5, 'linestyle': '-', 'label': '≤7 turbines'},
        'XLPE_2000': {'color': 'brown', 'linewidth': 4.5, 'linestyle': '-', 'label': '≤9 turbines'},
    }
    
    plotted_types = set()  # Track which cable types we've plotted for legend
    
    for u, v, edge_data in graph.edges(data=True):
        x_coords = [vertex_coords[u, 0], vertex_coords[v, 0]]
        y_coords = [vertex_coords[u, 1], vertex_coords[v, 1]]
        
        edge_key = (u, v)
        # Use MVAC scenario for visualization
        if edge_key in edge_cable_info and 'MVAC' in edge_cable_info[edge_key]:
            cable_info = edge_cable_info[edge_key]['MVAC']
            ref_type = cable_info['ref_type']
            
            if ref_type in cable_type_styles:
                style = cable_type_styles[ref_type]
                
                # Only include label for legend if this cable type hasn't been plotted yet
                label = style['label'] if ref_type not in plotted_types else ""
                if ref_type not in plotted_types:
                    plotted_types.add(ref_type)
                
                ax.plot(x_coords, y_coords, 
                       color=style['color'], 
                       linewidth=style['linewidth'],
                       linestyle=style['linestyle'],
                       alpha=0.7,
                       label=label)
        else:
            # Fallback for edges without cable info
            ax.plot(x_coords, y_coords, 'k-', linewidth=1.0, alpha=0.5)
    
    print(f"Plotted {graph.number_of_edges()} cable connections with discrete cable sizing")
else:
    print("Could not plot cables with discrete sizing")

# Add cost summary text to the plot
if total_cable_costs:
    # Prepare summary text
    summary_lines = ["Cable Cost Summary:"]
    for scenario, costs in total_cable_costs.items():
        summary_lines.append(f"{scenario}: ${costs['total_cost']:,.0f}")
    
    # Add text box with summary
    summary_text = '\n'.join(summary_lines)
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            fontsize=10)

ax.axis("equal")
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Wind Farm Layout with Discrete Cable Sizing')
plt.tight_layout()
plt.savefig("turbine_layout.png", dpi=300, bbox_inches='tight')
plt.show()


# Print detailed cost summary
if total_cable_costs:
    print("\n" + "="*80)
    print("CABLE COST ANALYSIS - DISCRETE CABLE SIZING")
    print("="*80)
    
    # Overall summary
    total_length_ft = next(iter(total_cable_costs.values()))['total_length_ft']
    print(f"Total cable length: {total_length_ft:.1f} feet ({total_length_ft*0.3048/1000:.1f} km)")
    print(f"Number of cable segments: {graph.number_of_edges() if graph else 'N/A'}")
    
    print("\nCost breakdown by cable technology:")
    print("-"*80)
    
    for scenario, costs in total_cable_costs.items():
        print(f"\n{scenario} Technology:")
        print(f"  Total Cost: ${costs['total_cost']:>12,.0f}")
        print(f"  Cable Type Usage:")
        
        for cable_type, usage in costs['cable_type_usage'].items():
            length_km = usage['length'] * 0.3048 / 1000
            print(f"    {cable_type:>10}: {usage['count']:>3} segments, {length_km:>6.1f} km")
    
    print("-"*80)
    cost_values = [costs['total_cost'] for costs in total_cable_costs.values()]
    print(f"Cost Range: ${min(cost_values):>12,.0f} - ${max(cost_values):>12,.0f}")
    print("="*80)
    print("Note: Costs use discrete cable sizing based on reference designs")
    print("      Cable types selected based on minimum capacity needed")
else:
    print("\nNo cable cost data available")
