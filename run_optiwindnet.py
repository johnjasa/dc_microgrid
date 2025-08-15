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

# Plot the turbine layout and cable connections
fig, ax = plt.subplots(figsize=(12, 10))

# Plot turbines
ax.scatter(x_turbines, y_turbines, c='blue', s=50, label='Wind Turbines')
for idx, (x, y) in enumerate(zip(x_turbines, y_turbines)):
    ax.text(x + 50, y + 50, str(idx), ha="left", va="bottom", fontsize=8)

# Plot substations
ax.scatter(x_substations, y_substations, c='red', s=100, marker='s', label='Substations')
for idx, (x, y) in enumerate(zip(x_substations, y_substations)):
    ax.text(x + 100, y + 100, f'S{idx}', ha="left", va="bottom", fontsize=10, fontweight='bold')

# Get the graph object from the collection system to plot cables
try:
    graph = optiwindnet_coll.graph
    vertex_coords = graph.graph["VertexC"]
    
    # Collect load information for all edges
    edge_loads = []
    
    # Plot cables between connected nodes with thickness based on load
    for u, v, edge_data in graph.edges(data=True):
        x_coords = [vertex_coords[u, 0], vertex_coords[v, 0]]
        y_coords = [vertex_coords[u, 1], vertex_coords[v, 1]]
        
        # Get the number of turbines this cable supports from edge data
        turbine_load = edge_data.get('load', 1)
        edge_loads.append(turbine_load)
        
        # Scale line width based on sqrt of turbine load (min 1.0, max 10.0 - area-based scaling)
        line_width = max(1.0, min(10.0, np.sqrt(turbine_load) * 2.0))
        
        ax.plot(x_coords, y_coords, 'k-', linewidth=line_width, alpha=0.7)
    
    print(f"Plotted {graph.number_of_edges()} cable connections")
    if edge_loads:
        print(f"Line widths scaled by turbine load (min: {min(edge_loads):.0f}, max: {max(edge_loads):.0f} turbines)")
except Exception as e:
    print(f"Could not plot cables: {e}")
    print("Showing turbine and substation layout only")

ax.axis("equal")
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Wind Farm Layout with Cable Connections')
plt.tight_layout()
plt.savefig("turbine_layout.png", dpi=300, bbox_inches='tight')
plt.show()

# Cable cost calculations
# User-definable cable costs per foot
cable_costs_per_ft = {
    'LVDC': 120.0,  # $120/ft for LVDC
    'MVAC': 12.0,   # $12/ft for MVAC
    'MVDC': 9.0     # $9/ft for MVDC
}

# Convert meters to feet for cost calculation
meters_to_feet = 3.28084

# Calculate costs for each cable type
total_cable_costs = {}

for cable_type, cost_per_ft in cable_costs_per_ft.items():
    total_cost = 0.0
    total_length_ft = 0.0
    
    # Get cable lengths and loads from the graph
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
        
        # Cost scaling based on sqrt of load (for cable sizing)
        cost_multiplier = np.sqrt(turbine_load)
        cable_cost = cable_length_ft * cost_per_ft * cost_multiplier
        
        total_cost += cable_cost
        total_length_ft += cable_length_ft
    
    total_cable_costs[cable_type] = {
        'total_cost': total_cost,
        'total_length_ft': total_length_ft,
        'cost_per_ft': cost_per_ft
    }

# Print cost summary
print("\n" + "="*60)
print("CABLE COST ANALYSIS")
print("="*60)
print(f"Total cable length: {total_length_ft:.1f} feet ({total_length_ft*0.3048/1000:.1f} km)")
print(f"Number of cable segments: {graph.number_of_edges()}")
print("\nCost breakdown by cable type:")
print("-"*60)

for cable_type, costs in total_cable_costs.items():
    print(f"{cable_type:>6}: ${costs['total_cost']:>12,.0f} (@ ${costs['cost_per_ft']:>6.1f}/ft)")

print("-"*60)
print(f"{'RANGE':>6}: ${min(costs['total_cost'] for costs in total_cable_costs.values()):>12,.0f} - ${max(costs['total_cost'] for costs in total_cable_costs.values()):>12,.0f}")
print("="*60)
print("Note: Costs include √(load) multiplier for cable sizing")
