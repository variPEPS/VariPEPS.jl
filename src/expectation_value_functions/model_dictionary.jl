model_dict = Dict(:Heisenberg_square => [use_Heisenberg_square_lattice,observables_Heisenberg_model_square],
                    :J1J2model_square => [use_J1J2model_square_lattice, observables_Heisenberg_model_square],
                    :Heisenberg_kagome => [use_Heisenberg_kagome_lattice, observables_spin_operators_kagome],
                    :Heisenberg_honeycomb => [use_Heisenberg_honeycomb, observables_Heisenberg_model_honeycomb],
                    :KitaevGamma => [use_Kitaev_Γ_model, observables_Heisenberg_model_honeycomb],
                    :Heisenberg_triangular => [use_Heisenberg_triangular_lattice,observables_Heisenberg_model_triangular],
                    :Kitaev_Γ_model_dice => [use_Kitaev_Γ_model_dice, observables_Kitaev_Γ_model_dice])
