import pytest
import polars as pl
from hot_deck_imputer import HotDeckImputer

@pytest.fixture
def donor_data():
    data = {
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
    return pl.DataFrame(data)

@pytest.fixture
def recipient_data():
    data = {
        'race_cell': ['Black','Black','Black','White','White',
                      'White','Black','White','Black','Black','Black','Black','White','White'],
        'sex_cell': ['M','F','F','M','F',
                     'M','F','F','M','F', 'F', 'M', 'M', 'F'],
        'work_cell': [1,0,1,0,1,
                      0,1,1,1,0,0,1,0,1],
        'weight': [1, 3, 2, 3, 2,
                   1, 4, 2, 1, 3, 4, 2, 1, 1]
    }
    return pl.DataFrame(data)

@pytest.fixture
def imputer(donor_data, recipient_data):
    return HotDeckImputer(donor_data=donor_data, 
                          imputation_var='donor_assets', 
                          weight_var='weight', 
                          recipient_data=recipient_data)

def test_age_dollar_amounts(imputer):
    pre_aging_amounts = imputer.donor_data['donor_assets']
    imputer.age_dollar_amounts(donor_year_cpi=223.1, imp_year_cpi=322.1)
    assert pre_aging_amounts = 

def test_define_cells(imputer):
    variables = ['race_cell', 'sex_cell']
    imputer.define_cells(variables)
    imputer.generate_cells()
    assert imputer.cell_definitions is not None
    assert imputer.donor_cells is not None
    assert imputer.recipient_cells is not None

def test_split_cell(imputer):
    imputer.define_cells(['race_cell', 'sex_cell'])
    imputer.generate_cells()
    imputer.split_cell("race_cell == 'Black' & sex_cell == 'F'", "work_cell")
    # Add assertions to verify the split

def test_impute(imputer):
    imputer.define_cells(['race_cell', 'sex_cell'])
    imputer.generate_cells()
    imputer.split_cell("race_cell == 'Black' & sex_cell == 'F'", "work_cell")
    imputer.impute()
    # Add assertions to verify the imputation

if __name__ == '__main__':
    pytest.main()