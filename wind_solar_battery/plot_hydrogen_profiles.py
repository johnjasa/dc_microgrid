import pickle
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_july_hydrogen_production():
    """
    Plot hydrogen production profiles for July (7th month), comparing 
    the first DC result against the first AC result from the sweep data.
    """
    
    # Files to load (using PV efficiency sweep as it has the first/baseline results)
    dc_filename = 'pv_efficiency_sweep_results_dc.pkl'
    ac_filename = 'pv_efficiency_sweep_results_ac.pkl'
    
    dc_data = None
    ac_data = None
    
    # Load DC results
    if os.path.exists(dc_filename):
        try:
            with open(dc_filename, 'rb') as f:
                dc_results = pickle.load(f)
            
            # Get the first successful result
            for result in dc_results['results_data']:
                if result.get('hydrogen_out') is not None:
                    dc_data = result['hydrogen_out']
                    dc_params = {
                        'pv_efficiency': result['pv_efficiency'],
                        'wind_loss': result['wind_loss'],
                        'battery_loss': result['battery_availability_loss_percent']
                    }
                    break
                    
            if dc_data is not None:
                print(f"Loaded DC data with parameters:")
                print(f"  PV Efficiency: {dc_params['pv_efficiency']:.2f}%")
                print(f"  Wind Loss: {dc_params['wind_loss']:.2f}%")
                print(f"  Battery Loss: {dc_params['battery_loss']:.2f}%")
            else:
                print("No successful DC results found")
                
        except Exception as e:
            print(f"Error loading DC file {dc_filename}: {e}")
    else:
        print(f"DC file {dc_filename} not found")
    
    # Load AC results
    if os.path.exists(ac_filename):
        try:
            with open(ac_filename, 'rb') as f:
                ac_results = pickle.load(f)
            
            # Get the first successful result
            for result in ac_results['results_data']:
                if result.get('hydrogen_out') is not None:
                    ac_data = result['hydrogen_out']
                    ac_params = {
                        'pv_efficiency': result['pv_efficiency'],
                        'wind_loss': result['wind_loss'],
                        'battery_loss': result['battery_availability_loss_percent']
                    }
                    break
                    
            if ac_data is not None:
                print(f"Loaded AC data with parameters:")
                print(f"  PV Efficiency: {ac_params['pv_efficiency']:.2f}%")
                print(f"  Wind Loss: {ac_params['wind_loss']:.2f}%")
                print(f"  Battery Loss: {ac_params['battery_loss']:.2f}%")
            else:
                print("No successful AC results found")
                
        except Exception as e:
            print(f"Error loading AC file {ac_filename}: {e}")
    else:
        print(f"AC file {ac_filename} not found")
    
    # Check if we have data to plot
    if dc_data is None and ac_data is None:
        print("No data available to plot")
        return
    
    # Create the plot with compact size for paper
    plt.figure(figsize=(8, 5))
    
    # Extract July data (7th month = index 6, assuming monthly data)
    # If the data is hourly for the full year, July would be hours 4344-5087 (31 days * 24 hours)
    # Let's first check the data structure
    
    if dc_data is not None:
        print(f"DC hydrogen_out data shape: {np.array(dc_data).shape}")
        data_length = len(dc_data)
        
        # Assume hourly data for full year (8760 hours)
        if data_length == 8760:
            # First week of July: hours 4344-4511 (7 days * 24 hours, starting from hour 4344)
            july_start = 4344  # Start of July (day 181 of year)
            july_end = july_start + 7 * 24  # 7 days * 24 hours
            
            july_dc_data = np.array(dc_data[july_start:july_end])
            
            # Create hourly axis for first week of July (168 hours = 7 days * 24 hours)
            hours = np.arange(1, len(july_dc_data) + 1)  # Hours 1-168 for first week of July
            
            plt.plot(hours, july_dc_data, '-', linewidth=2, 
                    color='tab:blue', label='DC Microgrid', alpha=0.8)
            
        else:
            print(f"Unexpected data length: {data_length}. Expected 8760 for hourly annual data.")
    
    if ac_data is not None:
        print(f"AC hydrogen_out data shape: {np.array(ac_data).shape}")
        data_length = len(ac_data)
        
        # Assume hourly data for full year (8760 hours)
        if data_length == 8760:
            # First week of July: hours 4344-4511 (7 days * 24 hours)
            july_start = 4344
            july_end = july_start + 7 * 24
            
            july_ac_data = np.array(ac_data[july_start:july_end])
            
            # Create hourly axis for first week of July (168 hours = 7 days * 24 hours)
            hours = np.arange(1, len(july_ac_data) + 1)  # Hours 1-168 for first week of July
            
            plt.plot(hours, july_ac_data, '-', linewidth=2, 
                    color='tab:orange', label='AC Microgrid', alpha=0.8)
            
        else:
            print(f"Unexpected data length: {data_length}. Expected 8760 for hourly annual data.")
    
    # Set plot properties
    plt.xlabel('First Week of July', fontsize=14)
    plt.ylabel('Hydrogen Production (kg/hr)', fontsize=14)
    # plt.title('Hydrogen Production Profile - July 1-7 Comparison\n(DC vs AC Microgrid Configurations)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc=0)
    
    # Set x-axis ticks for better readability (show every 24 hours = daily marks)
    day_marks = np.arange(1, 169, 24)  # Every 24 hours (daily marks) for 7 days
    day_labels = [f"July {i}" for i in range(1, 8)]  # July 1, July 2, ..., July 7
    plt.xticks(day_marks, day_labels, rotation=0, fontsize=10)
    
    # Add some styling
    plt.tight_layout()
    
    # Add text box with total production information
    text_content = []
    if dc_data is not None and len(dc_data) == 8760:
        july_start = 4344
        july_end = july_start + 7 * 24  # First week only
        july_dc_data = np.array(dc_data[july_start:july_end])
        july_dc_total = np.sum(july_dc_data)
        text_content.append(f"DC Week 1 Total: {july_dc_total:.0f} kg")
    
    if ac_data is not None and len(ac_data) == 8760:
        july_start = 4344
        july_end = july_start + 7 * 24  # First week only
        july_ac_data = np.array(ac_data[july_start:july_end])
        july_ac_total = np.sum(july_ac_data)
        text_content.append(f"AC Week 1 Total: {july_ac_total:.0f} kg")
    
    if text_content:
        text_str = '\n'.join(text_content)
        plt.text(0.68, 0.98, text_str, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, fontweight='bold')
    
    # Save the plot
    plot_filename = 'hydrogen_production_july_week1_comparison.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {plot_filename}")
    
    # Print some summary statistics
    if dc_data is not None and len(dc_data) == 8760:
        july_start = 4344
        july_end = july_start + 7 * 24  # First week only
        july_dc_data = np.array(dc_data[july_start:july_end])
        
        july_dc_total = np.sum(july_dc_data)
        july_dc_avg = np.mean(july_dc_data)
        print(f"\nDC Microgrid - July Week 1 Summary:")
        print(f"  Average hourly H2 production: {july_dc_avg:.2f} kg/hr")
        print(f"  Total Week 1 H2 production: {july_dc_total:.2f} kg")
        print(f"  Peak hourly production: {np.max(july_dc_data):.2f} kg/hr")
        print(f"  Minimum hourly production: {np.min(july_dc_data):.2f} kg/hr")
    
    if ac_data is not None and len(ac_data) == 8760:
        july_start = 4344
        july_end = july_start + 7 * 24  # First week only
        july_ac_data = np.array(ac_data[july_start:july_end])
        
        july_ac_total = np.sum(july_ac_data)
        july_ac_avg = np.mean(july_ac_data)
        print(f"\nAC Microgrid - July Week 1 Summary:")
        print(f"  Average hourly H2 production: {july_ac_avg:.2f} kg/hr")
        print(f"  Total Week 1 H2 production: {july_ac_total:.2f} kg")
        print(f"  Peak hourly production: {np.max(july_ac_data):.2f} kg/hr")
        print(f"  Minimum hourly production: {np.min(july_ac_data):.2f} kg/hr")
    
    if dc_data is not None and ac_data is not None and len(dc_data) == 8760 and len(ac_data) == 8760:
        july_start = 4344
        july_end = july_start + 7 * 24  # First week only
        july_dc_data = np.array(dc_data[july_start:july_end])
        july_ac_data = np.array(ac_data[july_start:july_end])
        july_dc_avg = np.mean(july_dc_data)
        july_ac_avg = np.mean(july_ac_data)
        
        efficiency_ratio = july_dc_avg / july_ac_avg if july_ac_avg > 0 else 0
        print(f"\nComparison (Week 1):")
        print(f"  DC/AC efficiency ratio: {efficiency_ratio:.3f}")
        print(f"  DC advantage: {(efficiency_ratio - 1) * 100:.1f}% higher average production")
    
    plt.show()
    
    return plot_filename


if __name__ == "__main__":
    print("Loading hydrogen production data and creating July comparison plot...")
    plot_filename = plot_july_hydrogen_production()
