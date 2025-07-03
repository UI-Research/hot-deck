## About
This repo contains Python code that implements hot deck imputation using Polars dataframes. It generalizes methods that are possible

Hot deck imputation involves randomly sampling individuals to create data for rows missing information. In many microsimulation settings at Urban, this concept is applied across datasets as well, where information missing in one dataset is inferred from another dataset. The basic process is as follows:

* Define categorical cells that are avaialable in both datasets, for example race.
* Divide the data into categories available in donor and source data.
* Split specific cells further if desired.
* Impute by randomly selecting observations from donor cells, and applying their values to recipients. 
* Compare the data among donor and recipient cells to ensure that relevant characteristics translate well from donor data to recipient data.

## Example Implementation in Python
### Install the package 
In the command line, do: `pip install git+https://github.com/UI-Research/hot-deck`
### Generate data tracking asset values and race, sex, and work
```
from hot_deck_class import HotDeckImputer
# Data where we know asset values, i.e. the 'donor'
donor_data = {
    'assets': [50000, 20000, 300000, 2000, 
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

# Data where we don't know asset values, i.e. the 'recipient'
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
```
### Instantiate HotDeckImputer
```
imputer = HotDeckImputer(donor_data = donor_data, 
                         imputation_var = 'assets', 
                         weight_var = 'weight', 
                         recipient_data = recipient_data)
```

### Age dollar amounts to align data collected in different years
```
imputer.age_dollar_amounts(donor_year_cpi = 223.1, imp_year_cpi = 322.1)
```

### Define cells according to race and sex
```
# Input as a list
variables = ['race_cell','sex_cell']
# Define every combination of race and sex, then partition data into cells
imputer.define_cells(variables)
imputer.generate_cells()
# View the definitions
imputer.cell_definitions
```

### Split specific cells up where sample allows
```
imputer.split_cell("race_cell == 'Black' & sex_cell == 'F'", "work_cell")
```

### Collapse specific cells up where sample allows
Collapses all conditions that start with `"race_cell == 'Black'"` into one bin. Updates `self.donor_cells`, `self.recipient_cells`, and `self.cell_definitions`. In this example, cells differentiated by work status and sex will be concatenated together for Black respondents.
```
imputer.collapse_cell(base_condition = "race_cell == 'Black'")
```
### Impute data
```
imputer.impute()
```
### Add random noise to smooth the results
```
imputer.apply_random_noise(variation_stdev = (1/6), floor_noise = 1.5)
```
### Generate file comparing donor data vs. recipient data
```
imputer.gen_analysis_file('hot_deck_stats')
```
