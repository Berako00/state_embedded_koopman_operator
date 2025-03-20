import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random as r
import tkinter
import tkinter.messagebox
import customtkinter

from ga_main_func import gamain_func  # Replace with your actual module name

# Redirect class to capture stdout and write to a text widget
class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert("end", string)
        self.text_widget.see("end")  # Scroll to the end as new text appears

    def flush(self):
        pass

customtkinter.set_appearance_mode("System")  # Modes: "System", "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Configure main window
        self.title("WAP Software")
        self.geometry("1200x700")

        # Configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame,text="Genetic Algorithm",font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.import_generate_tabview = customtkinter.CTkTabview(self.sidebar_frame, width=290, height=150)
        self.import_generate_tabview.grid(row=1, column=0, padx=10, pady=(10, 10), sticky="nsew")
        self.import_generate_tabview.add("Import Data")
        self.import_generate_tabview.add("Generate Data")

        # ---------------------------
        # Import Tab
        # ---------------------------
        self.import_frame = self.import_generate_tabview.tab("Import Data")
        self.dir_label = customtkinter.CTkLabel(self.import_frame, text="Dir:")
        self.dir_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.dir_entry = customtkinter.CTkEntry(self.import_frame)
        self.dir_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        # Allow the directory entry to expand:
        self.import_frame.grid_columnconfigure(1, weight=1)

        # ---------------------------
        # Generate Tab
        # ---------------------------
        self.generate_frame = self.import_generate_tabview.tab("Generate Data")
        # Row 0: System Selector
        self.system_selector_label = customtkinter.CTkLabel(self.generate_frame, text="System:")
        self.system_selector_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.system_selector = customtkinter.CTkOptionMenu(self.generate_frame, values=["Simple System", "Two Link"])
        self.system_selector.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Row 1: Nº Ini. Conditions (numICs)
        self.numICs_label = customtkinter.CTkLabel(self.generate_frame, text="Nº Ini. Conditions:")
        self.numICs_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.numICs_entry = customtkinter.CTkEntry(self.generate_frame)
        self.numICs_entry.insert(0, "20000")
        self.numICs_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # Row 2: Simulation Time (T_step)
        self.T_step_label = customtkinter.CTkLabel(self.generate_frame, text="Simulation Time:")
        self.T_step_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.T_step_entry = customtkinter.CTkEntry(self.generate_frame)
        self.T_step_entry.insert(0, "50")
        self.T_step_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # Row 3: Sampling Time (dt)
        self.dt_label = customtkinter.CTkLabel(self.generate_frame, text="Sampling Time:")
        self.dt_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.dt_entry = customtkinter.CTkEntry(self.generate_frame)
        self.dt_entry.insert(0, "0.02")
        self.dt_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        # Set the default to be Generate Data
        self.import_generate_tabview.set("Generate Data")
        self.system_selector.set("Two Link")

        # Ensure the second column expands in the Generate tab:
        self.generate_frame.grid_columnconfigure(1, weight=1)

        # Add a selector (dropdown) in column 0, row 0 to change tabs
        self.tab_selector = customtkinter.CTkOptionMenu(self.sidebar_frame,values=["Genetic Algorithm", "Fixed Parameters"],command=self.change_tab_event)
        self.tab_selector.grid(row=2, column=0, padx=20, pady=20, sticky="ew")

        self.n_meas_label = customtkinter.CTkLabel(self.sidebar_frame, text="Nº States:")
        self.n_meas_label.grid(row=3, column=0, padx=20, pady=(10, 0))
        self.n_meas_entry = customtkinter.CTkEntry(self.sidebar_frame, width=100)
        self.n_meas_entry.grid(row=4, column=0, padx=(0, 5), pady=5)

        self.n_inputs_label = customtkinter.CTkLabel(self.sidebar_frame, text="Nº Inputs:")
        self.n_inputs_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.n_inputs_entry = customtkinter.CTkEntry(self.sidebar_frame, width=100)
        self.n_inputs_entry.grid(row=6, column=0, padx=(0, 5), pady=5)

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame,text="Appearance Mode:",anchor="w")
        self.appearance_mode_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,values=["Light", "Dark", "System"],command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 10))

        # Create the tabview in column 1, row 0
        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=1, padx=(10, 0), pady=(0, 0), sticky="nsew")
        self.tabview.add("Genetic Algorithm")
        self.tabview.add("Fixed Parameters")
        self.tabview.tab("Genetic Algorithm").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Fixed Parameters").grid_columnconfigure(0, weight=1)

        # Hide the default tab header (segmented button)
        self.tabview._segmented_button.grid_forget()

        # --- Frame for Variable Inputs in the Genetic Algorithm Tab ---
        # --- GA Input Frame as a 4x4 Grid ---
        self.ga_input_frame = customtkinter.CTkFrame(self.tabview.tab("Genetic Algorithm"))
        self.ga_input_frame.grid(row=0, column=0, padx=20, pady=(5, 0), sticky="nsew")

        # Configure 4 rows and 4 columns for the frame
        for i in range(4):
            self.ga_input_frame.grid_rowconfigure(i, weight=1)
        for j in range(4):
            self.ga_input_frame.grid_columnconfigure(j, weight=1)

        # Row 0, Col 0: Nº x Hidden Layers (range)
        self.cell_0_0 = customtkinter.CTkFrame(self.ga_input_frame)
        self.cell_0_0.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.n_x_hidden_layers_label = customtkinter.CTkLabel(self.cell_0_0, text="Nº x Hidden Layers:")
        self.n_x_hidden_layers_label.grid(row=0, column=0, columnspan=2, sticky="w")
        self.n_x_hidden_layers_from = customtkinter.CTkEntry(self.cell_0_0, width=50, placeholder_text="from")
        self.n_x_hidden_layers_from.grid(row=1, column=0, padx=2, pady=2)
        self.n_x_hidden_layers_to = customtkinter.CTkEntry(self.cell_0_0, width=50, placeholder_text="to")
        self.n_x_hidden_layers_to.grid(row=1, column=1, padx=2, pady=2)

        # Row 0, Col 1: Nº u Hidden Layers (range)
        self.cell_0_1 = customtkinter.CTkFrame(self.ga_input_frame)
        self.cell_0_1.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.n_u_hidden_layers_label = customtkinter.CTkLabel(self.cell_0_1, text="Nº u Hidden Layers:")
        self.n_u_hidden_layers_label.grid(row=0, column=0, columnspan=2, sticky="w")
        self.n_u_hidden_layers_from = customtkinter.CTkEntry(self.cell_0_1, width=50, placeholder_text="from")
        self.n_u_hidden_layers_from.grid(row=1, column=0, padx=2, pady=2)
        self.n_u_hidden_layers_to = customtkinter.CTkEntry(self.cell_0_1, width=50, placeholder_text="to")
        self.n_u_hidden_layers_to.grid(row=1, column=1, padx=2, pady=2)

        # Row 0, Col 2: Nº x Neurons (range)
        self.cell_0_2 = customtkinter.CTkFrame(self.ga_input_frame)
        self.cell_0_2.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        self.n_x_neurons_label = customtkinter.CTkLabel(self.cell_0_2, text="Nº x Neurons:")
        self.n_x_neurons_label.grid(row=0, column=0, columnspan=2, sticky="w")
        self.n_x_neurons_from = customtkinter.CTkEntry(self.cell_0_2, width=50, placeholder_text="from")
        self.n_x_neurons_from.grid(row=1, column=0, padx=2, pady=2)
        self.n_x_neurons_to = customtkinter.CTkEntry(self.cell_0_2, width=50, placeholder_text="to")
        self.n_x_neurons_to.grid(row=1, column=1, padx=2, pady=2)

        # Row 0, Col 3: Nº u Neurons (range)
        self.cell_0_3 = customtkinter.CTkFrame(self.ga_input_frame)
        self.cell_0_3.grid(row=0, column=3, padx=5, pady=5, sticky="nsew")
        self.n_u_neurons_label = customtkinter.CTkLabel(self.cell_0_3, text="Nº u Neurons:")
        self.n_u_neurons_label.grid(row=0, column=0, columnspan=2, sticky="w")
        self.n_u_neurons_from = customtkinter.CTkEntry(self.cell_0_3, width=50, placeholder_text="from")
        self.n_u_neurons_from.grid(row=1, column=0, padx=2, pady=2)
        self.n_u_neurons_to = customtkinter.CTkEntry(self.cell_0_3, width=50, placeholder_text="to")
        self.n_u_neurons_to.grid(row=1, column=1, padx=2, pady=2)

        # Row 1, Col 0: Nº x Observables (range)
        self.cell_1_0 = customtkinter.CTkFrame(self.ga_input_frame)
        self.cell_1_0.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.n_x_observables_label = customtkinter.CTkLabel(self.cell_1_0, text="Nº x Observables:")
        self.n_x_observables_label.grid(row=0, column=0, columnspan=2, sticky="w")
        self.n_x_observables_from = customtkinter.CTkEntry(self.cell_1_0, width=50, placeholder_text="from")
        self.n_x_observables_from.grid(row=1, column=0, padx=2, pady=2)
        self.n_x_observables_to = customtkinter.CTkEntry(self.cell_1_0, width=50, placeholder_text="to")
        self.n_x_observables_to.grid(row=1, column=1, padx=2, pady=2)

        # Row 1, Col 1: Nº u Observables (range)
        self.cell_1_1 = customtkinter.CTkFrame(self.ga_input_frame)
        self.cell_1_1.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.n_u_observables_label = customtkinter.CTkLabel(self.cell_1_1, text="Nº u Observables:")
        self.n_u_observables_label.grid(row=0, column=0, columnspan=2, sticky="w")
        self.n_u_observables_from = customtkinter.CTkEntry(self.cell_1_1, width=50, placeholder_text="from")
        self.n_u_observables_from.grid(row=1, column=0, padx=2, pady=2)
        self.n_u_observables_to = customtkinter.CTkEntry(self.cell_1_1, width=50, placeholder_text="to")
        self.n_u_observables_to.grid(row=1, column=1, padx=2, pady=2)

        # Row 1, Col 2: Nº Generations (single)
        self.cell_1_2 = customtkinter.CTkFrame(self.ga_input_frame)
        self.cell_1_2.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")
        self.n_generations_label = customtkinter.CTkLabel(self.cell_1_2, text="Nº Generations:")
        self.n_generations_label.grid(row=0, column=0, sticky="w")
        self.n_generations_entry = customtkinter.CTkEntry(self.cell_1_2, width=100)
        self.n_generations_entry.grid(row=1, column=0, padx=2, pady=2)

        # Row 1, Col 3: Nº Population (single)
        self.cell_1_3 = customtkinter.CTkFrame(self.ga_input_frame)
        self.cell_1_3.grid(row=1, column=3, padx=5, pady=5, sticky="nsew")
        self.n_population_label = customtkinter.CTkLabel(self.cell_1_3, text="Nº Population:")
        self.n_population_label.grid(row=0, column=0, sticky="w")
        self.n_population_entry = customtkinter.CTkEntry(self.cell_1_3, width=100)
        self.n_population_entry.grid(row=1, column=0, padx=2, pady=2)

        # Row 2, Col 0: Nº Pre. Epochs (single)
        self.cell_2_0 = customtkinter.CTkFrame(self.ga_input_frame)
        self.cell_2_0.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        self.n_pre_epochs_label = customtkinter.CTkLabel(self.cell_2_0, text="Nº Pre. Epochs:")
        self.n_pre_epochs_label.grid(row=0, column=0, sticky="w")
        self.n_pre_epochs_entry = customtkinter.CTkEntry(self.cell_2_0, width=100)
        self.n_pre_epochs_entry.grid(row=1, column=0, padx=2, pady=2)

        # Row 2, Col 1: Nº Check Epochs (single)
        self.cell_2_1 = customtkinter.CTkFrame(self.ga_input_frame)
        self.cell_2_1.grid(row=2, column=1, padx=5, pady=5, sticky="nsew")
        self.n_check_epochs_label = customtkinter.CTkLabel(self.cell_2_1, text="Nº Check Epochs:")
        self.n_check_epochs_label.grid(row=0, column=0, sticky="w")
        self.n_check_epochs_entry = customtkinter.CTkEntry(self.cell_2_1, width=100)
        self.n_check_epochs_entry.grid(row=1, column=0, padx=2, pady=2)

        # Row 2, Col 2: Nº Train Epochs (single)
        self.cell_2_2 = customtkinter.CTkFrame(self.ga_input_frame)
        self.cell_2_2.grid(row=2, column=2, padx=5, pady=5, sticky="nsew")
        self.n_train_epochs_label = customtkinter.CTkLabel(self.cell_2_2, text="Nº Train Epochs:")
        self.n_train_epochs_label.grid(row=0, column=0, sticky="w")
        self.n_train_epochs_entry = customtkinter.CTkEntry(self.cell_2_2, width=100)
        self.n_train_epochs_entry.grid(row=1, column=0, padx=2, pady=2)

        # Row 2, Col 3: Tournament Size (single)
        self.cell_2_3 = customtkinter.CTkFrame(self.ga_input_frame)
        self.cell_2_3.grid(row=2, column=3, padx=5, pady=5, sticky="nsew")
        self.tournament_size_label = customtkinter.CTkLabel(self.cell_2_3, text="Tournament Size:")
        self.tournament_size_label.grid(row=0, column=0, sticky="w")
        self.tournament_size_entry = customtkinter.CTkEntry(self.cell_2_3, width=100)
        self.tournament_size_entry.grid(row=1, column=0, padx=2, pady=2)

        # Row 3, Col 0: Mutation rate (single)
        self.cell_3_0 = customtkinter.CTkFrame(self.ga_input_frame)
        self.cell_3_0.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
        self.mutation_rate_label = customtkinter.CTkLabel(self.cell_3_0, text="Mutation rate:")
        self.mutation_rate_label.grid(row=0, column=0, sticky="w")
        self.mutation_rate_entry = customtkinter.CTkEntry(self.cell_3_0, width=100)
        self.mutation_rate_entry.grid(row=1, column=0, padx=2, pady=2)

        # Row 3, Col 1: alpha0 (range)
        self.cell_3_1 = customtkinter.CTkFrame(self.ga_input_frame)
        self.cell_3_1.grid(row=3, column=1, padx=5, pady=5, sticky="nsew")
        self.alpha0_label = customtkinter.CTkLabel(self.cell_3_1, text="alpha0:")
        self.alpha0_label.grid(row=0, column=0, columnspan=2, sticky="w")
        self.alpha0_from = customtkinter.CTkEntry(self.cell_3_1, width=50, placeholder_text="from")
        self.alpha0_from.grid(row=1, column=0, padx=2, pady=2)
        self.alpha0_to = customtkinter.CTkEntry(self.cell_3_1, width=50, placeholder_text="to")
        self.alpha0_to.grid(row=1, column=1, padx=2, pady=2)

        # Row 3, Col 2: alpha1 (range)
        self.cell_3_2 = customtkinter.CTkFrame(self.ga_input_frame)
        self.cell_3_2.grid(row=3, column=2, padx=5, pady=5, sticky="nsew")
        self.alpha1_label = customtkinter.CTkLabel(self.cell_3_2, text="alpha1:")
        self.alpha1_label.grid(row=0, column=0, columnspan=2, sticky="w")
        self.alpha1_from = customtkinter.CTkEntry(self.cell_3_2, width=50, placeholder_text="from")
        self.alpha1_from.grid(row=1, column=0, padx=2, pady=2)
        self.alpha1_to = customtkinter.CTkEntry(self.cell_3_2, width=50, placeholder_text="to")
        self.alpha1_to.grid(row=1, column=1, padx=2, pady=2)

        # Row 3, Col 3: alpha2 (range)
        self.cell_3_3 = customtkinter.CTkFrame(self.ga_input_frame)
        self.cell_3_3.grid(row=3, column=3, padx=5, pady=5, sticky="nsew")
        self.alpha2_label = customtkinter.CTkLabel(self.cell_3_3, text="alpha2:")
        self.alpha2_label.grid(row=0, column=0, columnspan=2, sticky="w")
        self.alpha2_from = customtkinter.CTkEntry(self.cell_3_3, width=50, placeholder_text="from")
        self.alpha2_from.grid(row=1, column=0, padx=2, pady=2)
        self.alpha2_to = customtkinter.CTkEntry(self.cell_3_3, width=50, placeholder_text="to")
        self.alpha2_to.grid(row=1, column=1, padx=2, pady=2)

        # --- Fixed Parameters 4×4 Layout ---
        # Create a single frame to hold the 4×4 grid
        self.fixed_params_frame = customtkinter.CTkFrame(self.tabview.tab("Fixed Parameters"))
        self.fixed_params_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # Configure 4 rows and 4 columns
        for i in range(4):
            self.fixed_params_frame.grid_rowconfigure(i, weight=1)
        for j in range(4):
            self.fixed_params_frame.grid_columnconfigure(j, weight=1)

        # ---------------------------
        # Row 0
        # ---------------------------
        # Row 0, Col 0: Nº x Observables
        self.cell_0_0 = customtkinter.CTkFrame(self.fixed_params_frame)
        self.cell_0_0.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.num_x_obsv_label = customtkinter.CTkLabel(self.cell_0_0, text="Nº x Observables:")
        self.num_x_obsv_label.grid(row=0, column=0, sticky="w")
        self.num_x_obsv_entry = customtkinter.CTkEntry(self.cell_0_0, width=100)
        self.num_x_obsv_entry.insert(0, "3")
        self.num_x_obsv_entry.grid(row=1, column=0, padx=2, pady=2)

        # Row 0, Col 1: Nº u Observables
        self.cell_0_1 = customtkinter.CTkFrame(self.fixed_params_frame)
        self.cell_0_1.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.num_u_obsv_label = customtkinter.CTkLabel(self.cell_0_1, text="Nº u Observables:")
        self.num_u_obsv_label.grid(row=0, column=0, sticky="w")
        self.num_u_obsv_entry = customtkinter.CTkEntry(self.cell_0_1, width=100)
        self.num_u_obsv_entry.insert(0, "2")
        self.num_u_obsv_entry.grid(row=1, column=0, padx=2, pady=2)

        # Row 0, Col 2: Nº x Neurons
        self.cell_0_2 = customtkinter.CTkFrame(self.fixed_params_frame)
        self.cell_0_2.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        self.num_x_neurons_label = customtkinter.CTkLabel(self.cell_0_2, text="Nº x Neurons:")
        self.num_x_neurons_label.grid(row=0, column=0, sticky="w")
        self.num_x_neurons_entry = customtkinter.CTkEntry(self.cell_0_2, width=100)
        self.num_x_neurons_entry.insert(0, "30")
        self.num_x_neurons_entry.grid(row=1, column=0, padx=2, pady=2)

        # Row 0, Col 3: Nº u Neurons
        self.cell_0_3 = customtkinter.CTkFrame(self.fixed_params_frame)
        self.cell_0_3.grid(row=0, column=3, padx=5, pady=5, sticky="nsew")
        self.num_u_neurons_label = customtkinter.CTkLabel(self.cell_0_3, text="Nº u Neurons:")
        self.num_u_neurons_label.grid(row=0, column=0, sticky="w")
        self.num_u_neurons_entry = customtkinter.CTkEntry(self.cell_0_3, width=100)
        self.num_u_neurons_entry.insert(0, "30")
        self.num_u_neurons_entry.grid(row=1, column=0, padx=2, pady=2)

        # ---------------------------
        # Row 1
        # ---------------------------
        # Row 1, Col 0: Nº x Hidden Layers
        self.cell_1_0 = customtkinter.CTkFrame(self.fixed_params_frame)
        self.cell_1_0.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.num_hidden_x_label = customtkinter.CTkLabel(self.cell_1_0, text="Nº x Hidden Layers:")
        self.num_hidden_x_label.grid(row=0, column=0, sticky="w")
        self.num_hidden_x_entry = customtkinter.CTkEntry(self.cell_1_0, width=100)
        self.num_hidden_x_entry.insert(0, "2")
        self.num_hidden_x_entry.grid(row=1, column=0, padx=2, pady=2)

        # Row 1, Col 1: Nº u Hidden Layers
        self.cell_1_1 = customtkinter.CTkFrame(self.fixed_params_frame)
        self.cell_1_1.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.num_hidden_u_label = customtkinter.CTkLabel(self.cell_1_1, text="Nº u Hidden Layers:")
        self.num_hidden_u_label.grid(row=0, column=0, sticky="w")
        self.num_hidden_u_entry = customtkinter.CTkEntry(self.cell_1_1, width=100)
        self.num_hidden_u_entry.insert(0, "2")
        self.num_hidden_u_entry.grid(row=1, column=0, padx=2, pady=2)

        # Row 1, Col 2: Nº Check Epochs
        self.cell_1_2 = customtkinter.CTkFrame(self.fixed_params_frame)
        self.cell_1_2.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")
        self.check_epoch_label = customtkinter.CTkLabel(self.cell_1_2, text="Nº Check Epochs:")
        self.check_epoch_label.grid(row=0, column=0, sticky="w")
        self.check_epoch_entry = customtkinter.CTkEntry(self.cell_1_2, width=100)
        self.check_epoch_entry.insert(0, "10")
        self.check_epoch_entry.grid(row=1, column=0, padx=2, pady=2)

        # Row 1, Col 3: Nº Train Epochs (eps_final)
        self.cell_1_3 = customtkinter.CTkFrame(self.fixed_params_frame)
        self.cell_1_3.grid(row=1, column=3, padx=5, pady=5, sticky="nsew")
        self.eps_final_label = customtkinter.CTkLabel(self.cell_1_3, text="Nº Train Epochs:")
        self.eps_final_label.grid(row=0, column=0, sticky="w")
        self.eps_final_entry = customtkinter.CTkEntry(self.cell_1_3, width=100)
        self.eps_final_entry.insert(0, "20")
        self.eps_final_entry.grid(row=1, column=0, padx=2, pady=2)

        # ---------------------------
        # Row 2
        # ---------------------------
        # Here we place alpha0, alpha1, alpha2 (single values, not ranges)
        # Row 2, Col 0: alpha0
        self.cell_2_0 = customtkinter.CTkFrame(self.fixed_params_frame)
        self.cell_2_0.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        self.alpha0_label = customtkinter.CTkLabel(self.cell_2_0, text="Alpha 0:")
        self.alpha0_label.grid(row=0, column=0, sticky="w")
        self.alpha0_entry = customtkinter.CTkEntry(self.cell_2_0, width=100)
        self.alpha0_entry.insert(0, "0.01")  # Example default
        self.alpha0_entry.grid(row=1, column=0, padx=2, pady=2)

        # Row 2, Col 1: alpha1
        self.cell_2_1 = customtkinter.CTkFrame(self.fixed_params_frame)
        self.cell_2_1.grid(row=2, column=1, padx=5, pady=5, sticky="nsew")
        self.alpha1_label = customtkinter.CTkLabel(self.cell_2_1, text="Alpha 1:")
        self.alpha1_label.grid(row=0, column=0, sticky="w")
        self.alpha1_entry = customtkinter.CTkEntry(self.cell_2_1, width=100)
        self.alpha1_entry.insert(0, "1e-6")  # Example default
        self.alpha1_entry.grid(row=1, column=0, padx=2, pady=2)

        # Row 2, Col 2: alpha2
        self.cell_2_2 = customtkinter.CTkFrame(self.fixed_params_frame)
        self.cell_2_2.grid(row=2, column=2, padx=5, pady=5, sticky="nsew")
        self.alpha2_label = customtkinter.CTkLabel(self.cell_2_2, text="Alpha 2:")
        self.alpha2_label.grid(row=0, column=0, sticky="w")
        self.alpha2_entry = customtkinter.CTkEntry(self.cell_2_2, width=100)
        self.alpha2_entry.insert(0, "1e-12")  # Example default
        self.alpha2_entry.grid(row=1, column=0, padx=2, pady=2)

        # Row 2, Col 3: (optional placeholder or additional parameter)
        self.cell_2_3 = customtkinter.CTkFrame(self.fixed_params_frame)
        self.cell_2_3.grid(row=2, column=3, padx=5, pady=5, sticky="nsew")
        placeholder_label = customtkinter.CTkLabel(self.cell_2_3, text="(Empty)")
        placeholder_label.grid(row=0, column=0, sticky="nsew")

        # Optionally, set a default tab
        self.tabview.set("Genetic Algorithm")
        self.tab_selector.set("Genetic Algorithm")

        # Create the output "run window" in row 1, column 2
        # This text box will show all prints from gamain_func
        self.run_window = customtkinter.CTkTextbox(self, fg_color="black", text_color="white")
        self.run_window.grid(row=2, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # Redirect sys.stdout to our run window
        sys.stdout = RedirectText(self.run_window)

        # Create start button
        self.start_button = customtkinter.CTkButton(self, text="Start", command=self.start_button_event, width=80)
        self.start_button.grid(row=3, column=3, padx=20, pady=20, sticky="se")

        # Set default appearance mode
        self.appearance_mode_optionemenu.set("Dark")

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def change_tab_event(self, selected_tab: str):
        self.tabview.set(selected_tab)

    def import_button_event(self):
        print("import button click")

    def start_button_event(self):
        try:
            # Retrieve and convert GUI inputs
            num_meas = int(self.n_meas_entry.get())
            num_inputs = int(self.n_inputs_entry.get())

            generate = True if self.import_generate_tabview.get().strip() == "Generate Data" else False
            if generate == True:
                system_value = self.system_selector.get().strip()  # e.g., "Simple System" or "Two Link"
                if system_value.lower() == "two link":
                    system = "two_link"
                    print(system)
                else:
                    system = "simple_system"  # or another string based on your needs
                    print(system)

                # Extract the remaining values with defaults if empty:
                numICs = int(self.numICs_entry.get())
                T_step = int(self.T_step_entry.get())
                dt = float(self.dt_entry.get())

            use_ga = True if self.tab_selector.get() == "Genetic Algorithm" else False


            if use_ga == True:
                fix_params = {'Num_x_Obsv': 0, 'Num_u_Obsv': 0, 'Num_x_Neurons': 0, 'Num_u_Neurons': 0,'Num_hidden_x': 0, 'Num_hidden_u': 0, 'alpha0': 0, 'alpha1': 0, 'alpha2': 0}

                check_epoch = int(self.n_check_epochs_entry.get())
                eps_final = int(self.n_train_epochs_entry.get())
                generations = int(self.n_generations_entry.get())
                pop_size = int(self.n_population_entry.get())
                eps = int(self.n_pre_epochs_entry.get())
                tournament_size = int(self.tournament_size_entry.get())
                mutation_rate = float(self.mutation_rate_entry.get())

                # Get parameter range inputs for GA parameters:
                num_x_obsv_from = int(self.n_x_observables_from.get())
                num_x_obsv_to = int(self.n_x_observables_to.get())
                num_u_obsv_from = int(self.n_u_observables_from.get())
                num_u_obsv_to = int(self.n_u_observables_to.get())
                num_x_neurons_from = int(self.n_x_neurons_from.get())
                num_x_neurons_to = int(self.n_x_neurons_to.get())
                num_u_neurons_from = int(self.n_u_neurons_from.get())
                num_u_neurons_to = int(self.n_u_neurons_to.get())
                num_hidden_x_from = int(self.n_x_hidden_layers_from.get())
                num_hidden_x_to = int(self.n_x_hidden_layers_to.get())
                num_hidden_u_from = int(self.n_u_hidden_layers_from.get())
                num_hidden_u_to = int(self.n_u_hidden_layers_to.get())
                alpha0_from = float(self.alpha0_from.get())
                alpha0_to = float(self.alpha0_to.get())
                alpha1_from = float(self.alpha1_from.get())
                alpha1_to = float(self.alpha1_to.get())
                alpha2_from = float(self.alpha2_from.get())
                alpha2_to = float(self.alpha2_to.get())

                # Prepare the GA parameters dictionary
                ga_params = {
                    'generations': generations,
                    'pop_size': pop_size,
                    'eps': eps,
                    'tournament_size': tournament_size,
                    'mutation_rate': mutation_rate,
                    'param_ranges': {
                        "Num_x_Obsv": (num_x_obsv_from, num_x_obsv_to),
                        "Num_u_Obsv": (num_u_obsv_from, num_u_obsv_to),
                        "Num_x_Neurons": (num_x_neurons_from, num_x_neurons_to),
                        "Num_u_Neurons": (num_u_neurons_from, num_u_neurons_to),
                        "Num_hidden_x": (num_hidden_x_from, num_hidden_x_to),
                        "Num_hidden_u": (num_hidden_u_from, num_hidden_u_to),
                        "alpha0": (alpha0_from, alpha0_to),
                        "alpha1": (alpha1_from, alpha1_to),
                        "alpha2": (alpha2_from, alpha2_to)
                    },
                    'elitism_count': 1
                }

            else:
                ga_params = {'generations': 0,'pop_size': 0,'eps': 0,'tournament_size': 0,'mutation_rate': 0,'param_ranges': {"Num_x_Obsv": (0, 0),"Num_u_Obsv": (0, 0),"Num_x_Neurons": (0, 0),"Num_u_Neurons": (0, 0),"Num_hidden_x": (0, 0),"Num_hidden_u": (0, 0),"alpha0": (0, 0),"alpha1": (0, 0),"alpha2": (0, 0)},'elitism_count': 0}

                check_epoch = int(self.check_epoch_entry.get())
                eps_final = int(self.eps_final_entry.get())

                # Get parameter inputs for fixed parameters:
                num_x_obsv = int(self.num_x_obsv_entry.get())
                num_u_obsv = int(self.num_u_obsv_entry.get())
                num_x_neurons = int(self.num_x_neurons_entry.get())
                num_u_neurons = int(self.num_u_neurons_entry.get())
                num_hidden_x = int(self.num_hidden_x_entry.get())
                num_hidden_u = int(self.num_hidden_u_entry.get())
                alpha0_fix = float(self.alpha0_entry.get())
                alpha1_fix = float(self.alpha1_entry.get())
                alpha2_fix = float(self.alpha2_entry.get())

                fix_params = {
                    'Num_x_Obsv': num_x_obsv,
                    'Num_u_Obsv': num_u_obsv,
                    'Num_x_Neurons': num_x_neurons,
                    'Num_u_Neurons': num_u_neurons,
                    'Num_hidden_x': num_hidden_x,
                    'Num_hidden_u': num_hidden_u,
                    'alpha0': alpha0_fix,
                    'alpha1': alpha1_fix,
                    'alpha2': alpha2_fix
                }

            # Here we use example hard-coded values:
            training_params = {
                'eps_final': eps_final,
                'check_epoch': check_epoch,
                'lr': 1e-3,
                'batch_size': 256,
                'S_p': 30
            }

            # Now call the function with the dynamic inputs
            results = gamain_func(
                system=system,
                numICs=numICs,
                T_step=T_step,
                dt=dt,
                num_meas = num_meas,
                num_inputs = num_inputs,
                use_ga = use_ga,
                ga_params = ga_params,
                fix_params = fix_params,
                training_params=training_params
            )

            trained_model = results["model"]
            print("Model training completed successfully!")
        except Exception as e:
            print("Error processing inputs:", e)

if __name__ == "__main__":
    app = App()
    app.mainloop()
