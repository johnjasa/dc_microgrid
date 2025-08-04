import pickle
import matplotlib.pyplot as plt
import os


def load_and_plot_sweep_results():
    """
    Load the three sweep result pickle files and create plots for each.
    Plots LCOH vs parameter value for each sweep.
    """
    
    # Define the pickle files and their corresponding plot info
    sweep_files = [
        {
            'dc_filename': 'pv_efficiency_sweep_results_dc.pkl',
            'ac_filename': 'pv_efficiency_sweep_results_ac.pkl',
            'title': 'PV Power Systems Efficiency Sweep',
            'xlabel': 'PV Power Systems Efficiency (%)',
            'param_key': 'parameter_value'
        },
        {
            'dc_filename': 'wind_loss_sweep_results_dc.pkl',
            'ac_filename': 'wind_loss_sweep_results_ac.pkl',
            'title': 'Wind Power Systems Efficiency Sweep',
            'xlabel': 'Wind Electrical Efficiency (%)',
            'param_key': 'parameter_value'
        }
    ]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(6, 10))
    fig.suptitle('LCOH Sensitivity Analysis (DC vs AC)', fontsize=16, fontweight='bold')
    
    all_results = {}
    
    for i, sweep_info in enumerate(sweep_files):
        dc_filename = sweep_info['dc_filename']
        ac_filename = sweep_info['ac_filename']
        
        # Load and plot DC results
        dc_data = None
        ac_data = None
        
        # Load DC data
        if os.path.exists(dc_filename):
            try:
                with open(dc_filename, 'rb') as f:
                    dc_results = pickle.load(f)
                
                param_values = []
                lcoh_values = []
                
                for result in dc_results['results_data']:
                    if result.get('lcoh') is not None:
                        if sweep_info['param_key'] == 'battery_availability_loss_percent':
                            param_values.append(result['battery_availability_loss_percent'])
                        else:
                            param_value = result['parameter_value']
                            # Convert wind loss to efficiency (100 - loss)
                            if 'wind_loss' in dc_filename:
                                param_value = 100 - param_value
                            param_values.append(param_value)
                        lcoh_values.append(result['lcoh'])
                
                if param_values:
                    # Sort by parameter value for cleaner plotting
                    sorted_data = sorted(zip(param_values, lcoh_values))
                    dc_data = sorted_data
                    
            except Exception as e:
                print(f"Error loading DC file {dc_filename}: {e}")
        
        # Load AC data (skip for battery since only DC exists)
        if ac_filename != dc_filename and os.path.exists(ac_filename):
            try:
                with open(ac_filename, 'rb') as f:
                    ac_results = pickle.load(f)
                
                param_values = []
                lcoh_values = []
                
                for result in ac_results['results_data']:
                    if result.get('lcoh') is not None:
                        if sweep_info['param_key'] == 'battery_availability_loss_percent':
                            param_values.append(result['battery_availability_loss_percent'])
                        else:
                            param_value = result['parameter_value']
                            # Convert wind loss to efficiency (100 - loss)
                            if 'wind_loss' in ac_filename:
                                param_value = 100 - param_value
                            param_values.append(param_value)
                        lcoh_values.append(result['lcoh'])
                
                if param_values:
                    # Sort by parameter value for cleaner plotting
                    sorted_data = sorted(zip(param_values, lcoh_values))
                    ac_data = sorted_data
                    
            except Exception as e:
                print(f"Error loading AC file {ac_filename}: {e}")
        
        # Check if we have any data to plot
        if not dc_data and not ac_data:
            axes[i].text(0.5, 0.5, f'No data files\nfound for\n{sweep_info["title"]}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(sweep_info['title'])
            continue
        
        # Plot DC data if available
        if dc_data:
            param_values, lcoh_values = zip(*dc_data)
            axes[i].plot(param_values, lcoh_values, 'o-', linewidth=2, markersize=6, color='tab:blue', label='DC')
            
            # Add DC-DC label positioned off the data line
            label_x = param_values[0] + 0.05 * (param_values[-1] - param_values[0])
            label_y = lcoh_values[0] + 0.05 * (max(lcoh_values) - min(lcoh_values)) - 0.02
            axes[i].text(label_x, label_y, 'DC microgrid', fontsize=10, color='tab:blue', fontweight='bold')
            
            # Store results for summary
            all_results[f"{sweep_info['title']} (DC)"] = dc_results
            
            # Print DC summary statistics
            print(f"\n{sweep_info['title']} (DC) Summary:")
            print(f"  Parameter range: {min(param_values):.2f} - {max(param_values):.2f}")
            print(f"  LCOH range: {min(lcoh_values):.4f} - {max(lcoh_values):.4f} $/kg H2")
            print(f"  Successful runs: {dc_results['run_info']['successful_runs']}/{dc_results['run_info']['total_runs']}")
        
        # Plot AC data if available and different from DC
        if ac_data and ac_filename != dc_filename:
            param_values, lcoh_values = zip(*ac_data)
            axes[i].plot(param_values, lcoh_values, 's-', linewidth=2, markersize=6, color='tab:orange', label='AC')
            
            # Add AC-DC label positioned off the data line
            label_x = param_values[-1] - 0.05 * (param_values[-1] - param_values[0])
            label_y = lcoh_values[-1] - 0.05 * (max(lcoh_values) - min(lcoh_values)) + 0.05
            axes[i].text(label_x, label_y, 'AC microgrid', fontsize=10, color='tab:orange', fontweight='bold')
            
            # Store results for summary
            all_results[f"{sweep_info['title']} (AC)"] = ac_results
            
            # Print AC summary statistics
            print(f"\n{sweep_info['title']} (AC) Summary:")
            print(f"  Parameter range: {min(param_values):.2f} - {max(param_values):.2f}")
            print(f"  LCOH range: {min(lcoh_values):.4f} - {max(lcoh_values):.4f} $/kg H2")
            print(f"  Successful runs: {ac_results['run_info']['successful_runs']}/{ac_results['run_info']['total_runs']}")
        
        # Set plot properties
        axes[i].set_xlabel(sweep_info['xlabel'])
        axes[i].set_ylabel('LCOH ($/kg H2)')
        axes[i].set_title(sweep_info['title'])
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'sweep_results_comparison.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {plot_filename}")
    
    return all_results




if __name__ == "__main__":
    print("Loading and plotting sweep results...")
    
    # Create comparison plot (default)
    results = load_and_plot_sweep_results()
