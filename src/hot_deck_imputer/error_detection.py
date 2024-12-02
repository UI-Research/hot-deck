"""
Class defining assertion utilities for hot deck imputer.
Functions:
    validate_data: Validate the input data and parameters.
    check_variable_consistency: Check if the unique values and types of cell variables are the same in donor and recipient datasets.
"""
import numpy as np
import polars as pl

def validate_data(self):
    """
    Validate the input data and parameters.
    :raises ValueError: If data are empty, imputation variable is missing from donor data, 
                        imputation variable is already in recipient data, 
                        weight variable is missing, weight variable contains missing values
    """
    # Check for non-empty DataFrames
    if self.donor_data.is_empty():
        raise ValueError("Donor data is empty")
    if self.recipient_data.is_empty():
        raise ValueError("Recipient data is empty")
    
    # Check for imputation variable's presence on both sides
    if self.imputation_var not in self.donor_data.columns:
        raise ValueError(f"Column '{self.imputation_var}' is missing from donor data")
    if self.imputation_var in self.recipient_data.columns:
        raise ValueError(f"Column '{self.imputation_var}' is already in recipient data, does not need to be imputed")
    
    # Check for weight variables + missingness if weighted imputation is requiured
    if self.weight_var is not None:
        if self.weight_var not in self.donor_data.columns:
            raise ValueError(f"Column '{self.weight_var}' is missing from donor data")
        if self.weight_var not in self.recipient_data.columns:
            raise ValueError(f"Column '{self.weight_var}' is missing from recipient data")

        # Check for missing values in required columns
        if self.donor_data[self.weight_var].null_count() > 0:
            raise ValueError(f"Column '{self.weight_var}' in donor data contains {self.donor_data[self.weight_var].null_count()} missing values")
        if self.recipient_data[self.weight_var].null_count() > 0:
            raise ValueError(f"Column '{self.weight_var}' in recipient data contains {self.recipient_data[self.weight_var].null_count()} missing values")
        
    return

def check_variable_consistency(self, variables:list):
    """
    Non-callable method to check if the unique values and types of the variables
    are the same in donor and recipient datasets.
    :param variables: List of variables to check
    :raises TypeError: If data types do not match between donor and recipient
    :raises ValueError: If unique values do not match between donor and recipient
    """
    for var in variables:
        donor_unique = self.donor_data[var].unique()
        recipient_unique = self.recipient_data[var].unique()

        # Check if the types match
        if self.donor_data[var].dtype != self.recipient_data[var].dtype:
            raise TypeError(f"Data types for variable '{var}' do not match between donor and recipient datasets.")
        
        # Check if the unique values match
        if set(donor_unique) != set(recipient_unique):
            raise ValueError(f"Unique values for variable '{var}' do not match between donor and recipient datasets.")
        
    return