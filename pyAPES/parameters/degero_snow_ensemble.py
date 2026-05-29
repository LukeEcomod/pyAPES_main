from itertools import product

physics_option_levels = {
    'DENSTY': (1, 2),
    'HYDRL': (0, 1),
    'CONDCT': (0, 1),
    'EXCHNG': (0, 1),
    'ALBEDO': (1, 2),
    'SNFRAC': (0, 1),
}

physics_combinations = list(product(
    physics_option_levels['DENSTY'],
    physics_option_levels['HYDRL'],
    physics_option_levels['CONDCT'],
    physics_option_levels['EXCHNG'],
    physics_option_levels['ALBEDO'],
    physics_option_levels['SNFRAC'],
))

Nsim = len(physics_combinations)

degero_snow_fsm2_physics_parameters = {
    'count': Nsim, # Number of simulations
    'scenario': 'degero_snow_fsm2_physics',
    'canopy': {
        'forestfloor': {
            'snowpack': {
                'fsm2': {
                    'physics_options': {
                        'DENSTY': tuple(combo[0] for combo in physics_combinations),
                        'HYDRL': tuple(combo[1] for combo in physics_combinations),
                        'CONDCT': tuple(combo[2] for combo in physics_combinations),
                        'EXCHNG': tuple(combo[3] for combo in physics_combinations),
                        'ALBEDO': tuple(combo[4] for combo in physics_combinations),
                        'SNFRAC': tuple(combo[5] for combo in physics_combinations),
                    }
                }
            }
        }
    }
}