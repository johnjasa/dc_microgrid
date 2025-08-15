import pickle
from h2integrate.core.h2integrate_model import H2IntegrateModel


def run_separate_efficiency_sweeps(type="dc"):
    """
    Run three separate sweeps for PV efficiency, wind losses, and battery availability losses.
    Each sweep holds the other two parameters constant at their first values.
    """
    
    if type == "dc":
        # Define the efficiency/loss values to sweep
        pv_efficiency_percentages = [
            97.95,
            97.66,
            97.37,
            97.07,
            96.78,
            96.49,
            93.56,
            90.63,
            90.00,
            ]
        wind_loss_percentages = [
            4.90,
            5.48,
            6.06,
            6.64,
            7.21,
            13.01,
        ]
        battery_availability_loss_percentages = [
            2.12,
            2.40,
            2.67,
            2.95,
            3.24,
            3.51,
            6.32,
            9.10,
            11.86,
            ]
    else:  # case is "ac"
        # Define the efficiency/loss values to sweep
        pv_efficiency_percentages = [
            93.99,
            93.70,
            93.41,
            93.11,
            92.82,
            92.53,
            90.,
            90.,
            90.,
            ]
        wind_loss_percentages = [
            9.27,
            9.68,
            10.08,
            10.47,
            10.86,
            14.43,
            17.54,
        ]
        battery_availability_loss_percentages = [
            4.09,
            6.41,
            8.63,
            10.74,
            12.75,
            14.68,
            ]
        
    # Results from the cable layout costs
    if type == "dc":
        wind_installed_cost_mw = 1228308 + 15.289e6 / 450
    elif type == "ac":
        wind_installed_cost_mw = 1228308 + 20.385e6 / 450
        
    # Convert battery availability losses to decimal format (divide by 100)
    battery_availability_losses = [loss/100 for loss in battery_availability_loss_percentages]
    
    # Baseline values (first value from each list)
    baseline_pv_eff = pv_efficiency_percentages[0]
    baseline_wind_loss = wind_loss_percentages[0]
    baseline_battery_loss = battery_availability_losses[0]
    
    # Path to the hopp config file
    hopp_config_path = "../tech_inputs/hopp_config_wind_solar_battery.yaml"
    
    print("Starting separate parameter efficiency sweeps...")
    print(f"Baseline values: PV={baseline_pv_eff}%, Wind Loss={baseline_wind_loss}%, Battery Loss={baseline_battery_loss*100:.2f}%")
    
    # Run PV efficiency sweep
    print(f"\n=== PV Efficiency Sweep (holding Wind Loss at {baseline_wind_loss}%, Battery Loss at {baseline_battery_loss*100:.2f}%) ===")
    pv_results = run_single_parameter_sweep(
        "pv", pv_efficiency_percentages, hopp_config_path,
        baseline_pv_eff, baseline_wind_loss, baseline_battery_loss, wind_installed_cost_mw
    )
    
    # Save PV results
    pv_pickle = f"pv_efficiency_sweep_results_{type}.pkl"
    with open(pv_pickle, 'wb') as f:
        pickle.dump(pv_results, f)
    print(f"PV sweep results saved to: {pv_pickle}")
    
    # Run Wind loss sweep  
    print(f"\n=== Wind Loss Sweep (holding PV Efficiency at {baseline_pv_eff}%, Battery Loss at {baseline_battery_loss*100:.2f}%) ===")
    wind_results = run_single_parameter_sweep(
        "wind", wind_loss_percentages, hopp_config_path,
        baseline_pv_eff, baseline_wind_loss, baseline_battery_loss, wind_installed_cost_mw
    )
    
    # Save Wind results
    wind_pickle = f"wind_loss_sweep_results_{type}.pkl"
    with open(wind_pickle, 'wb') as f:
        pickle.dump(wind_results, f)
    print(f"Wind sweep results saved to: {wind_pickle}")
    
    # Run Battery availability loss sweep
    print(f"\n=== Battery Availability Loss Sweep (holding PV Efficiency at {baseline_pv_eff}%, Wind Loss at {baseline_wind_loss}%) ===")
    battery_results = run_single_parameter_sweep(
        "battery", battery_availability_losses, hopp_config_path,
        baseline_pv_eff, baseline_wind_loss, baseline_battery_loss, wind_installed_cost_mw
    )
    
    # Save Battery results
    battery_pickle = f"battery_availability_sweep_results_{type}.pkl"
    with open(battery_pickle, 'wb') as f:
        pickle.dump(battery_results, f)
    print(f"Battery sweep results saved to: {battery_pickle}")
    
    print(f"\nAll sweeps completed!")
    print(f"Results saved to three separate files:")
    print(f"  - {pv_pickle}")
    print(f"  - {wind_pickle}")
    print(f"  - {battery_pickle}")
    
    return pv_results, wind_results, battery_results


def run_single_parameter_sweep(param_type, param_values, hopp_config_path, baseline_pv, baseline_wind, baseline_battery, wind_installed_cost_mw=0):
    """
    Run a sweep for a single parameter type, holding others constant.
    
    Args:
        param_type: 'pv', 'wind', or 'battery'
        param_values: List of values to sweep
        hopp_config_path: Path to HOPP config file
        baseline_pv: Baseline PV efficiency
        baseline_wind: Baseline wind loss
        baseline_battery: Baseline battery availability loss
        wind_installed_cost_mw: Modifier to add to wind installed cost
    """
    results_data = []
    
    for i, value in enumerate(param_values):
        # Set parameters based on type
        if param_type == 'pv':
            pv_eff = value
            wind_loss = baseline_wind
            batt_loss = baseline_battery
            param_name = "PV Efficiency"
            param_units = "%"
        elif param_type == 'wind':
            pv_eff = baseline_pv
            wind_loss = value
            batt_loss = baseline_battery
            param_name = "Wind Loss"
            param_units = "%"
        elif param_type == 'battery':
            pv_eff = baseline_pv
            wind_loss = baseline_wind
            batt_loss = value
            param_name = "Battery Availability Loss"
            param_units = "%"
            display_value = value * 100  # Convert to percentage for display
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
        
        if param_type == 'battery':
            print(f"Run {i+1}/{len(param_values)}: {param_name}={display_value:.2f}{param_units}")
        else:
            print(f"Run {i+1}/{len(param_values)}: {param_name}={value}{param_units}")
        
        # Modify the hopp config file
        modify_hopp_config_multi_params(hopp_config_path, pv_eff, wind_loss, batt_loss, wind_installed_cost_mw)
        
        try:
            # Run the H2Integrate model
            gh = H2IntegrateModel("wind_solar_battery.yaml")
            gh.run()
            gh.post_process()
            
            # Extract results
            lcoe = gh.prob['financials_group_profast.LCOE'][0]
            lcoh = gh.prob['financials_group_profast.LCOH'][0]
            hydrogen_out = gh.prob['electrolyzer.hydrogen_out']

            # Store results
            result_entry = {
                'parameter_type': param_type,
                'parameter_value': value,
                'pv_efficiency': pv_eff,
                'wind_loss': wind_loss,
                'battery_availability_loss': batt_loss,
                'battery_availability_loss_percent': batt_loss * 100,
                'lcoe': lcoe,
                'lcoh': lcoh,
                'hydrogen_out': hydrogen_out
            }
            results_data.append(result_entry)
            
            if lcoe is not None and lcoh is not None:
                print(f"  Completed: LCOE = {lcoe:.4f}, LCOH = {lcoh:.4f}")
            else:
                print(f"  Completed: Result extraction failed")
                
        except Exception as e:
            print(f"  Error in run {i+1}: {str(e)}")
            result_entry = {
                'parameter_type': param_type,
                'parameter_value': value,
                'pv_efficiency': pv_eff,
                'wind_loss': wind_loss,
                'battery_availability_loss': batt_loss,
                'battery_availability_loss_percent': batt_loss * 100,
                'lcoe': None,
                'lcoh': None,
                'hydrogen_out': None,
                'error': str(e)
            }
            results_data.append(result_entry)
    
    # Create results summary
    results = {
        'parameter_type': param_type,
        'results_data': results_data,
        'parameter_values': param_values,
        'baseline_values': {
            'pv_efficiency': baseline_pv,
            'wind_loss': baseline_wind,
            'battery_availability_loss': baseline_battery
        },
        'run_info': {
            'total_runs': len(param_values),
            'successful_runs': sum(1 for r in results_data if r.get('lcoe') is not None),
            'failed_runs': sum(1 for r in results_data if r.get('lcoe') is None)
        }
    }
    
    print(f"  Successful runs: {results['run_info']['successful_runs']}/{results['run_info']['total_runs']}")
    
    return results


def modify_hopp_config_multi_params(config_path, pv_inverter_efficiency, wind_loss, battery_availability_loss, wind_installed_cost_mw=0):
    """
    Modify the hopp config file in place to set multiple parameters:
    - PV inverter efficiency (inv_eff)
    - Wind turbine electrical efficiency loss (elec_eff_loss) 
    - Battery availability loss (availability_loss)
    - Wind installed cost modifier (added to wind_installed_cost_mw)
    """
    with open(config_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        # Update PV inverter efficiency
        if 'inv_eff' in line and 'pv:' in ''.join(lines[max(0, len(new_lines)-10):len(new_lines)]):
            parts = line.split(':', 1)
            if len(parts) == 2:
                key, _ = parts
                new_line = f"{key}: {pv_inverter_efficiency}\n"
                new_lines.append(new_line)
                continue
        
        # Update wind electrical efficiency loss
        elif 'elec_eff_loss' in line and 'wind:' in ''.join(lines[max(0, len(new_lines)-10):len(new_lines)]):
            parts = line.split(':', 1)
            if len(parts) == 2:
                key, _ = parts
                new_line = f"{key}: {wind_loss}\n"
                new_lines.append(new_line)
                continue
        
        # Update battery availability loss
        elif 'availability_loss' in line and 'battery:' in ''.join(lines[max(0, len(new_lines)-10):len(new_lines)]):
            parts = line.split(':', 1)
            if len(parts) == 2:
                key, _ = parts
                new_line = f"{key}: {battery_availability_loss}\n"
                new_lines.append(new_line)
                continue
        
        # Update wind installed cost by adding the modifier to the base cost
        elif 'wind_installed_cost_mw' in line and 'cost_info' in ''.join(lines[max(0, len(new_lines)-10):len(new_lines)]):
            parts = line.split(':', 1)
            if len(parts) == 2:
                key, value_part = parts
                # Extract the base cost value (remove comments if any)
                base_cost_str = value_part.split('#')[0].strip()
                try:
                    base_cost = float(base_cost_str)
                    new_cost = wind_installed_cost_mw
                    # Preserve any comment that was in the original line
                    if '#' in value_part:
                        comment = '#' + value_part.split('#', 1)[1]
                        new_line = f"{key}: {new_cost} {comment}"
                    else:
                        new_line = f"{key}: {new_cost}\n"
                    new_lines.append(new_line)
                    continue
                except ValueError:
                    # If we can't parse the cost, keep the original line
                    pass
        
        new_lines.append(line)

    with open(config_path, 'w') as f:
        f.writelines(new_lines)


def load_multi_results(pickle_filename="multi_efficiency_sweep_results.pkl"):
    """
    Load and display results from a previous multi-parameter sweep.
    """
    try:
        with open(pickle_filename, 'rb') as f:
            results = pickle.load(f)
        
        print("Loaded multi-parameter efficiency sweep results:")
        print("PV Eff [%] | Wind Loss [%] | Batt Loss [%] | LCOE")
        print("-" * 55)
        
        for result in results['results_data']:
            if result.get('lcoe') is not None:
                print(f"{result['pv_efficiency']:>9.2f} | {result['wind_loss']:>12.2f} | {result['battery_availability_loss_percent']:>12.2f} | {result['lcoe']:.4f}")
            else:
                print(f"{result['pv_efficiency']:>9.2f} | {result['wind_loss']:>12.2f} | {result['battery_availability_loss_percent']:>12.2f} | Failed")
        
        return results
        
    except FileNotFoundError:
        print(f"Results file {pickle_filename} not found.")
        return None
    except Exception as e:
        print(f"Error loading results: {e}")
        return None


def run_single_parameter_sweeps():
    """
    Run individual sweeps for each parameter type (PV, Wind, Battery) separately.
    This is useful for understanding individual parameter sensitivities.
    """
    print("Running individual parameter sweeps...")
    
    base_config = {
        'pv_efficiency': 87.7,  # baseline from config
        'wind_loss': 2.0,       # baseline from config  
        'battery_availability_loss': 0.01  # baseline from config
    }
    
    # PV efficiency sweep
    print("\n=== PV Efficiency Sweep ===")
    pv_efficiency_percentages = [97.95, 97.66, 97.37, 97.07, 96.78, 96.49, 93.56, 90.63, 87.70]
    pv_results = run_parameter_sweep('pv', pv_efficiency_percentages, base_config)
    
    # Wind loss sweep
    print("\n=== Wind Loss Sweep ===")
    wind_loss_percentages = [4.90, 5.48, 6.06, 6.64, 7.21, 13.01]
    wind_results = run_parameter_sweep('wind', wind_loss_percentages, base_config)
    
    # Battery availability loss sweep
    print("\n=== Battery Availability Loss Sweep ===")
    battery_availability_loss_percentages = [2.12, 2.40, 2.67, 2.95, 3.24, 3.51, 6.32, 9.10, 11.86]
    battery_availability_losses = [loss/100 for loss in battery_availability_loss_percentages]
    battery_results = run_parameter_sweep('battery', battery_availability_losses, base_config)
    
    # Save individual results
    individual_results = {
        'pv_sweep': pv_results,
        'wind_sweep': wind_results,
        'battery_sweep': battery_results,
        'base_config': base_config
    }
    
    pickle_filename = "individual_parameter_sweeps.pkl"
    with open(pickle_filename, 'wb') as f:
        pickle.dump(individual_results, f)
    
    print(f"\nIndividual parameter sweep results saved to: {pickle_filename}")
    
    return individual_results


def run_parameter_sweep(param_type, param_values, base_config):
    """
    Run a sweep for a single parameter type.
    """
    hopp_config_path = "../tech_inputs/hopp_config_wind_solar_battery.yaml"
    results = []
    
    for i, value in enumerate(param_values):
        print(f"  Run {i+1}/{len(param_values)}: {param_type} = {value}")
        
        # Set parameters based on type
        if param_type == 'pv':
            pv_eff = value
            wind_loss = base_config['wind_loss']
            batt_loss = base_config['battery_availability_loss']
        elif param_type == 'wind':
            pv_eff = base_config['pv_efficiency']
            wind_loss = value
            batt_loss = base_config['battery_availability_loss']
        elif param_type == 'battery':
            pv_eff = base_config['pv_efficiency']
            wind_loss = base_config['wind_loss']
            batt_loss = value
        
        modify_hopp_config_multi_params(hopp_config_path, pv_eff, wind_loss, batt_loss, 0)
        
        try:
            gh = H2IntegrateModel("wind_solar_battery.yaml")
            gh.run()
            gh.post_process()
            lcoe = gh.prob['financials_group_profast.LCOE'][0]
            
            results.append({
                'parameter_value': value,
                'lcoe': lcoe
            })
            
            print(f"    LCOE = {lcoe:.4f}")
            
        except Exception as e:
            print(f"    Error: {str(e)}")
            results.append({
                'parameter_value': value,
                'lcoe': None,
                'error': str(e)
            })
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "individual":
        # Run individual parameter sweeps (legacy function)
        results = run_single_parameter_sweeps()
    else:
        # Run separate parameter sweeps (new default behavior)
        results = run_separate_efficiency_sweeps(type="ac")
        results = run_separate_efficiency_sweeps(type="dc")
