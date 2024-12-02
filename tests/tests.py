from hot_deck_class import HotDeckImputer
import polars as pl 

if __name__ == '__main__':
    # CREATE DATA FOR TESTS
    donor_data = {
    'donor_assets': [50000, 20000, 300000, 2000, 
                     10000, 10000, 200, 2000, 4000, 500000],
    'race_cell': ['Black','Black','Black','White','White',
                     'White','Black','White','Black','Black'],
    'sex_cell': ['M','F','F','M','F',
                     'M','F','F','M','F'],
    'work_cell': [1,0,1,0,1,
                     0,1,1,1,0],
    'weight': [1, 2, 1, 2, 1,
               2, 1, 2, 1, 2]
    }

    donor_data = pl.DataFrame(donor_data)

    recipient_data = {
        'race_cell': ['Black','Black','Black','White','White',
                        'White','Black','White','Black','Black','Black','Black','White','White'],
        'sex_cell': ['M','F','F','M','F',
                        'M','F','F','M','F', 'F', 'M', 'M', 'F'],
        'work_cell': [1,0,1,0,1,
                        0,1,1,1,0,0,1,0,1],
        'weight': [1, 3, 2, 3, 2,
                1, 4, 2, 1, 3, 4, 2, 1, 1]
    }

    recipient_data = pl.DataFrame(recipient_data)

    # INITIALIZE IMPUTER
    imputer = HotDeckImputer(donor_data = donor_data, 
                         imputation_var = 'donor_assets', 
                         weight_var = 'weight', 
                         recipient_data = recipient_data)
    
    # TEST CPI AGE ADJUSTMENT
    imputer.age_dollar_amounts(donor_year_cpi = 223.1, imp_year_cpi = 322.1)

    # DEFINE CELLS
    variables = ['race_cell','sex_cell']

    imputer.define_cells(variables)
    imputer.generate_cells()

    # SPLIT SPECIFIC CELLS
    imputer.split_cell("race_cell == 'Black' & sex_cell == 'F'", "work_cell")

    # IMPUTE
    imputer.impute()

    # APPLY RANDOM NOISE TO OUTPUTS
    imputer.apply_random_noise(variation_stdev = (1/6), floor_noise = 1.5)

    # GENERATE ANALYSIS FILE
    imputer.gen_analysis_file('hot_deck_stats')


