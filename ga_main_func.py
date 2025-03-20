def gamain_func(
        system='two_link',  # Options: 'two_link' or 'simple'
        numICs=20000,  # Number of initial conditions for data generation
        T_step=50,  # Number of timesteps per sample
        dt=0.02,  # Timestep interval
        seed=1,  # Random seed for reproducibility
        num_meas=4,  # Number of measurements
        num_inputs=2,  # Number of inputs
        use_ga=True,  # Whether to use genetic algorithm optimization
        ga_params=None,  # Dictionary to override GA hyperparameters
        fix_params=None,
        training_params=None,  # Dictionary to override training hyperparameters
        device=None  # Device to run on (optional)
):
    """
    Run the autoencoder experiment, including data generation,
    optional genetic algorithm hyperparameter tuning, training,
    and result plotting.

    Args:
        system (str): 'two_link' or 'simple' system.
        numICs (int): Number of initial conditions for data generation.
        T_step (int): Number of timesteps per sample.
        dt (float): Timestep interval.
        seed (int): Random seed for reproducibility.
        num_meas (int): Number of measurements.
        num_inputs (int): Number of inputs.
        use_ga (bool): If True, perform genetic algorithm hyperparameter tuning.
        ga_params (dict): GA hyperparameters, e.g.,
            {
                'generations': 6,
                'pop_size': 6,
                'eps': 500,
                'tournament_size': 2,
                'mutation_rate': 0.2,
                'param_ranges': { ... },
                'elitism_count': 1
            }
        training_params (dict): Training parameters, e.g.,
            {
                'eps_final': 5000,
                'check_epoch': 10,
                'lr': 1e-3,
                'batch_size': 256,
                'S_p': 30
            }
        device (torch.device): Device to run on (defaults to GPU if available).

    Returns:
        dict: Contains the trained model, total elapsed time, training time,
              training results, and best model path.
    """
    import time
    import math
    import torch
    import matplotlib.pyplot as plt

    # Import your helper modules
    from help_func import self_feeding, enc_self_feeding
    from nn_structure import AUTOENCODER
    from training import trainingfcn
    from data_generation import DataGenerator, TwoLinkRobotDataGenerator
    from plotting import plot_results, plot_debug
    from ga_optimizer import run_genetic_algorithm

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    overall_start = time.time()

    # ---- System Params ----------
    # Use input parameters for num_meas and num_inputs
    Num_meas = num_meas
    Num_inputs = num_inputs
    # ----------------------------

    # ------- Data Generation Params ----------
    # For reproducibility, you might want to set random seeds here if needed.
    if system == 'simple':
        x1range = (-0.5, 0.5)
        x2range = x1range
        mu = -0.05
        lam = -1
    elif system == 'two_link':
        q1_range = (-math.pi/2, math.pi/2)
        q2_range = (-313.2/2 * math.pi/180, 313.2/2 * math.pi/180)
        dq1_range = (-1, 1)
        dq2_range = dq1_range
        tau_max = 1
    # -----------------------------------------

    if system == 'simple':
        train_tensor, test_tensor, val_tensor = DataGenerator(x1range, x2range, numICs, mu, lam, T_step, dt)
    elif system == 'two_link':
        train_tensor, test_tensor, val_tensor = TwoLinkRobotDataGenerator(q1_range, q2_range, dq1_range, dq2_range, numICs, T_step, dt, tau_max)

    # ---- GA Parameters -------------
    # Set defaults for GA parameters if not provided
    if ga_params is None:
        ga_params = {
            'generations': 6,
            'pop_size': 6,
            'eps': 500,
            'tournament_size': 2,
            'mutation_rate': 0.2,
            'param_ranges': {
                "Num_x_Obsv": (4, 20),
                "Num_u_Obsv": (2, 20),
                "Num_x_Neurons": (10, 50),
                "Num_u_Neurons": (10, 50),
                "Num_hidden_x": (1, 3),
                "Num_hidden_u": (1, 3),
                "alpha0": (0.01, 1.0),
                "alpha1": (1e-9, 1e-5),
                "alpha2": (1e-18, 1e-12)
            },
            'elitism_count': 1
        }
    # ---- GA Parameters -------------
    # Set defaults for GA parameters if not provided
    if fix_params is None:
        fix_params = {
            'Num_x_Obsv': 3,
            'Num_u_Obsv': 3,
            'Num_x_Neurons': 30,
            'Num_u_Neurons': 30,
            'Num_hidden_x': 2,
            'Num_hidden_u': 2,
            'alpha0': 0.01,
            'alpha1': 1e-7,
            'alpha2': 1e-15
            }


    # ---- Training Parameters -------------
    if training_params is None:
        training_params = {
            'eps_final': 5000,    # Final number of epochs for training
            'check_epoch': 10,    # Checkpoint epochs
            'lr': 1e-3,           # Learning rate
            'batch_size': 256,    # Batch size
            'S_p': 30             # Additional training parameter (as in your code)
        }
    eps_final   = training_params['eps_final']
    check_epoch = training_params['check_epoch']
    lr          = training_params['lr']
    batch_size  = training_params['batch_size']
    S_p         = training_params['S_p']

    # T is the number of timesteps per sample
    T = train_tensor.shape[1]
    M = 1  # Number of models to run

    # If not using GA, set default network hyperparameters.
    if not use_ga:
        Num_x_Obsv    = fix_params['Num_x_Obsv']
        Num_u_Obsv    = fix_params['Num_u_Obsv']
        Num_x_Neurons = fix_params['Num_x_Neurons']
        Num_u_Neurons = fix_params['Num_u_Neurons']
        Num_hidden_x  = fix_params['Num_hidden_x']
        Num_hidden_u  = fix_params['Num_hidden_u']
        alpha         = [fix_params['alpha0'], fix_params['alpha1'], fix_params['alpha2']]
    else:
        # --- Run Genetic Algorithm for Hyperparameter Optimization ---
        best_params = run_genetic_algorithm(
            check_epoch, Num_meas, Num_inputs,
            train_tensor, test_tensor,
            ga_params['tournament_size'],
            ga_params['mutation_rate'],
            ga_params['generations'],
            ga_params['pop_size'],
            ga_params['eps'],
            param_ranges=ga_params['param_ranges'],
            elitism_count=ga_params['elitism_count']
        )
        # Override hyperparameters with GA results
        Num_meas      = best_params['Num_meas']
        Num_inputs    = best_params['Num_inputs']
        Num_x_Obsv    = best_params['Num_x_Obsv']
        Num_u_Obsv    = best_params['Num_u_Obsv']
        Num_x_Neurons = best_params['Num_x_Neurons']
        Num_u_Neurons = best_params['Num_u_Neurons']
        Num_hidden_x  = best_params['Num_hidden_x']
        Num_hidden_u  = best_params['Num_hidden_u']
        alpha         = [best_params['alpha0'], best_params['alpha1'], best_params['alpha2']]

    print(f"Train tensor shape: {train_tensor.shape}")
    print(f"Test tensor shape: {test_tensor.shape}")
    print(f"Validation tensor shape: {val_tensor.shape}")

    # ---- Create the Model ----
    model = AUTOENCODER(
        Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons,
        Num_u_Obsv, Num_u_Neurons, Num_hidden_x,
        Num_hidden_x, Num_hidden_u, Num_hidden_u
    )

    # --- Training Loop ---
    training_start = time.time()

    # trainingfcn returns multiple results including the best model file path
    results = trainingfcn(
        eps_final, check_epoch, lr, batch_size, S_p, T, alpha,
        Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons,
        Num_u_Obsv, Num_u_Neurons, Num_hidden_x, Num_hidden_x,
        Num_hidden_u, Num_hidden_u, train_tensor, test_tensor, M,
        device=device
    )
    (Lowest_loss, Models_loss_list, Best_Model, Lowest_loss_index,
     Running_Losses_Array, Lgx_Array, Lgu_Array,
     L3_Array, L4_Array, L5_Array, L6_Array) = results

    # Load the parameters of the best model
    model.load_state_dict(torch.load(Best_Model))
    print(f"Loaded model parameters from Model: {Best_Model}")

    training_end = time.time()
    total_time = training_end - overall_start
    total_training_time = training_end - training_start

    print(f"Total elapsed time: {total_time}")
    print(f"Total training time: {total_training_time}")

    # ----- Plotting Results and Debug Information -----
    plot_debug(model, val_tensor, train_tensor, S_p, Num_meas, Num_x_Obsv, T)
    plot_results(model, val_tensor, train_tensor, S_p, Num_meas, Num_x_Obsv, T)
    plt.show(block=False)

    return {
        "model": model,
        "total_time": total_time,
        "total_training_time": total_training_time,
        "training_results": results,
        "best_model_path": Best_Model
    }

