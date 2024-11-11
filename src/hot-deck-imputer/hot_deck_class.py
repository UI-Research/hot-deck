"""
Class defining DYNASIM-FEH file reader.

DYNASIM FEH produces three files:

    - header file,
    - family file, and
    - person file.


read_feh and save_feh modules define functionality for accessing, processing, and writing files.
This module defines classes that can be used to edit those functionalities.
"""
import numpy as np
import polars as pl

class HotDeckImputer:
    def __init__(self, donor_data:pl.DataFrame, 
                 imputation_var:str, weight_var:str,
                 recipient_data:pl.DataFrame):
        """
        Initialize with the dataset. Donor data is the source for the hot deck.
        Recipient data is the dataset that will receive the imputation.
        """
        self.donor_data = donor_data.clone()
        self.imputation_var = imputation_var
        self.weight_var = weight_var
        self.recipient_data = recipient_data.clone()

        # Cell definition attributes to be defined in methods
        self.cell_definitions = None
        self.donor_cells = None
        self.recipient_cells = None

        # Random noise attributes defined in methods
        self.random_noise = None
    
    def take_cell_definitions(self, cell_definitions):
        """
        Method to accept cell definitions.
        :param cell_definitions: A list of conditions for defining cells
        """
        self.cell_definitions = cell_definitions

    def generate_cells(self):
        """
        Method to generate cells based on cell definitions.
        It splits the data according to the conditions provided in the cell_definitions.
        """
        if not self.cell_definitions:
            raise ValueError("Cell definitions are not provided")

        # Create empty dictionary to store the partitions
        donor_cells = {}
        recipient_cells = {}
        
        for i, condition in enumerate(self.cell_definitions):
            # Create cell based on condition
            donor_cells[f'{condition}'] = self.donor_data.query(condition)
            recipient_cells[f'{condition}'] = self.recipient_data.query(condition)
        
        self.donor_cells = donor_cells
        self.recipient_cells = recipient_cells
        return
    
    def _check_variable_consistency(self, variables):
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

    def define_cells(self, variables):
        """
        Method to define all possible cell definitions given a list of input variables.
        :param variables: A list of column names (variables) from the data to partition by.
        For example: ['homeowner_hh_flag', 'member_over_60']
        :return: A list of strings representing all possible conditions
        """
        # First, check if the variables are consistent across donor and recipient datasets
        self._check_variable_consistency(variables)

        # Extract unique values from the donor data for each variable
        var_values = {var: self.donor_data[var].unique() for var in variables}

        # Generate all possible combinations of variable values
        var_combinations = list(itertools.product(*var_values.values()))

        # Create the condition strings
        cell_definitions = []
        for combination in var_combinations:
            conditions = [
            f"({variables[i]} == '{combination[i]}')" if isinstance(combination[i], str) else f"({variables[i]} == {combination[i]})"
            for i in range(len(combination))
            ]
            cell_definitions.append(' & '.join(conditions))

        self.cell_definitions = cell_definitions
        return 

    def split_cell(self, cell_condition, split_condition):
        """
        Method to split an individual cell further based on a new condition.
        :param cell: Dataframe representing the cell to be split
        :param split_condition: A condition string to further split the cell.
        :return: Two dataframes representing the split
        """
        # combine conditions together
        combined_condition = f'{cell_condition} & {split_condition}'
        combined_not_condition = f'{cell_condition} & not {split_condition}'
        
        # get the data for cells that are going to be split
        split_donor = self.donor_cells[cell_condition]
        split_recipient = self.recipient_cells[cell_condition]

        # Remove the original cell from the donor and recipient cell dictionaries
        del self.donor_cells[cell_condition]
        del self.recipient_cells[cell_condition]

        # Add the newly split cells into the dictionaries
        self.donor_cells[combined_condition] = split_donor.query(combined_condition)
        self.donor_cells[combined_not_condition] = split_donor.query(combined_not_condition)

        self.recipient_cells[combined_condition] = split_recipient.query(combined_condition)
        self.recipient_cells[combined_not_condition] = split_recipient.query(combined_not_condition)

        # Update cell definitions
        self.cell_definitions.remove(cell_condition)
        self.cell_definitions.append(combined_condition)
        self.cell_definitions.append(combined_not_condition)
        return 
    
    def summarize_cells(self):
        results = {}
        for i, recipient_cell in self.recipient_cells.items():

            ### Donor cell + source variable
            donor_cell = self.donor_cells.get(i)
            source_var = donor_cell[f'{self.imputation_var}'].copy()

            # Donor stat generation
            donor_stats = DescrStatsW(source_var, weights=donor_cell[self.weight_var], ddof=0)
            # Recipient stat generation
            source_var = recipient_cell[f'imp_{self.imputation_var}'].copy()
            recipient_stats = DescrStatsW(source_var, weights=recipient_cell[self.weight_var], ddof=0)

            data = {
                'statistic': [
                    '95int_low', 'mean', '95int_high', 'stddev', 'var', 'stderr', 'sum', 'obs'
                ],
                'donor': [
                    donor_stats.mean - 1.96 * donor_stats.std_mean,  # 95% CI low for donor
                    donor_stats.mean,                                # Mean for donor
                    donor_stats.mean + 1.96 * donor_stats.std_mean,  # 95% CI high for donor
                    donor_stats.std,                                 # Stddev for donor
                    donor_stats.var,                                 # Variance for donor
                    donor_stats.std_mean,                            # Std error for donor
                    donor_stats.sum_weights,                         # Weighted sum for donor
                    donor_cell.shape[0]                              # Observations for donor
                ],

                'imp': [
                    recipient_stats.mean - 1.96 * recipient_stats.std_mean,  # 95% CI low for imp
                    recipient_stats.mean,                                    # Mean for imp
                    recipient_stats.mean + 1.96 * recipient_stats.std_mean,  # 95% CI high for imp
                    recipient_stats.std,                                     # Stddev for imp
                    recipient_stats.var,                                     # Variance for imp
                    recipient_stats.std_mean,                                # Std error for imp
                    recipient_stats.sum_weights,                             # Weighted sum for imp
                    recipient_cell.shape[0]                                  # Observations for imp
                ]
            }

            # Convert dictionary to DataFrame
            stats_df = pd.DataFrame(data)
            stats_df['diff'] = stats_df['imp'] - stats_df['donor']
            stats_df['imp_to_donor_ratio'] = stats_df['imp']/stats_df['donor']

            # Remove scientific notation for clarity
            numeric_cols = ['donor', 'imp', 'diff']
            stats_df[numeric_cols] = stats_df[numeric_cols].applymap(lambda x: f"{x:,.2f}")

            results[i] = stats_df

        return results

    def gen_analysis_file(self, out_file, out_path):
        data = self.summarize_cells()

        # Create an Excel writer
        with pd.ExcelWriter(f'{out_path}/{out_file}.xlsx') as writer:
            # Set a starting row variable to track where to place data in the single sheet
            start_row = 0
            
            # Loop over each cell's data
            for key, df in data.items():
                # Write a separator row with the key as a label
                separator = pd.DataFrame([['cell', key]], columns=['statistic', 'donor'])
                separator.to_excel(writer, sheet_name='cells', index=False, header=False, startrow=start_row)
                
                # Move start_row down by 1 to allow space after the separator
                start_row += 1
                
                # Write the actual DataFrame below the separator row
                df.to_excel(writer, sheet_name='cells', startrow=start_row, index=False)
                
                # Move start_row down by the number of rows in the df plus an extra row for spacing
                start_row += len(df) + 2

        print(f"Cell data written to '{out_path}\{out_file}.xlsx'.")

    def apply_random_noise(self, variation_stdev, floor_noise = None):
        """
        Add random noise to smooth out issue of clustering
            * Within each cell, sort by asset value in donor data 
            * Get a lagged variable for each row showing asset value of next neighbor
            * Compute for the whole cell, the average distance between asset values and their neighbors.
            * Add noise to every recipient- a RV with mean 0 and standard deviation of 1/6th of the mean distance for that cell
        """
        imputed_recipient_cells = []

        for condition, donor_cell in self.donor_cells.items():
            # Sort donor cell by asset value
            donor_cell = donor_cell.sort_values(by='liquid_assets').reset_index(drop=True)

            # Calculate the next neighbor values
            donor_cell['next_val'] = donor_cell[f'{self.imputation_var}'].shift(-1)

            # Compute the distance to prior and next neighbor
            donor_cell['next_distance'] = np.subtract(donor_cell['next_val'], donor_cell[f'{self.imputation_var}'])

            # Calculate the average neighbor distance for the cell, ignoring NaN values
            ## First get mean of each row and then get mean of the result
            mean_distance = donor_cell['next_distance'].mean()

            # Calculate noise level as a proportion of the mean distance between neighbors
            noise_stdev = mean_distance * variation_stdev

            # Calculate the threshold value based on relevant floor for asset tests
            if floor_noise is not None:
                print(f'Cell:\n{condition}')
                threshold = floor_noise
                print(f'Min threshold for noise injection:\n{threshold}')
            else:
                print(f'Cell:\n{condition}')
                threshold = self.donor_data[f'{self.imputation_var}'].min()
                print(f'Min threshold for noise injection:\n{threshold}')

            # Generate random noise for each recipient in the cell
            # Only apply this random noise for those who are less than some factor of the standard deviation of neighboring distances
            # i.e. if floor_stdev_multiplier = 2, observations with <2x the standard deviation of neighboring distances are left alone
            recipient_cell = self.recipient_cells[condition]

            # identify the observations who are above the threshold, who will have random noise added
            # when there is no thresholding by floor_stdev_multiplier, this is handled by the minimum identified above
            ge_thresh = recipient_cell[f'imp_{self.imputation_var}'] >= threshold
            noise = np.random.normal(loc=0, scale=noise_stdev, size=ge_thresh.sum())
            print(f'max noise: {noise.max()}, min noise: {noise.min()}')

            # Apply noise to the imputed liquid assets in the recipient cell
            recipient_cell.loc[ge_thresh, f'imp_{self.imputation_var}'] += noise
            min_donor_val = donor_cell[f'{self.imputation_var}'].min()
            recipient_cell.loc[ge_thresh, f'imp_{self.imputation_var}'] = recipient_cell.loc[ge_thresh, f'imp_{self.imputation_var}'].clip(lower = min_donor_val) # ensure nobody is made to have negative values

            # Update recipient data with noisy values
            self.recipient_cells[condition][f'imp_{self.imputation_var}'] = recipient_cell[f'imp_{self.imputation_var}']
            imputed_recipient_cells.append(recipient_cell)

        # Store the variation standard deviation parameter
        self.random_noise = variation_stdev
        self.recipient_data = pd.concat(imputed_recipient_cells)
        
        return

    def summarize_column(self, data, column_name):
        """
        Summarize a column in donor_data, returning basic statistics.
        :param column_name: The column to summarize
        :return: A dictionary with summary statistics
        """
        # Check if the column exists in the DataFrame
        if column_name not in self.donor_data.columns:
            raise ValueError(f"Column '{column_name}' does not exist in donor_data.")
        
        # Calculate summary statistics
        summary_stats = {
            'mean': data[column_name].mean(),
            'median': data[column_name].median(),
            'min': data[column_name].min(),
            'max': data[column_name].max(),
            'std_dev': data[column_name].std(),
            'count': data[column_name].count(),
            'missing_values': data[column_name].isna().sum()
        }

        return summary_stats

    def age_dollar_amounts(self, donor_year_cpi, imp_year_cpi):
        """
        Age the imputed values to the target year. Relevant when the source data and target data differ.
        https://www.cbo.gov/data/budget-economic-data#4 for CPI indexes
        """
        
        print(f'Summary of {self.imputation_var} pre CPI aging:\n{self.summarize_column(self.donor_data, self.imputation_var)}')
        scaling_factor = imp_year_cpi / donor_year_cpi
        self.donor_data[self.imputation_var] *= scaling_factor
        print(f'Summary of {self.imputation_var} post CPI aging:\n{self.summarize_column(self.donor_data, self.imputation_var)}')

        return

    def impute(self):
        """
        Impute the missing values in the recipient data using the donor data for corresponding cells.
        This method assumes that both donor and recipient data have been partitioned using generate_cells.
        """
        if not self.cell_definitions:
            raise ValueError("Cell definitions are not provided")
        
        # List to hold imputed recipient cells
        imputed_recipient_cells = []

        # For each recipient cell, find the corresponding donor cell and perform imputation
        for i, recipient_cell in self.recipient_cells.items():
            donor_cell = self.donor_cells.get(i)
            
            if donor_cell is not None and not donor_cell.empty:
                # Perform weighted random selection for the required number of values
                if self.weight_var:
                    weights = donor_cell[self.weight_var].values
                    donor_values = donor_cell[self.imputation_var].dropna().values

                    # Randomly select `missing_count` values from the donor set using the weights
                    # Using weighted selection according to probability proportional to weights https://documentation.sas.com/doc/en/statcdc/14.2/statug/statug_surveyimpute_details25.htm#statug.surveyimpute.weightedDet
                    selected_values = np.random.choice(donor_values, size=len(recipient_cell), replace=True, p=weights / weights.sum())
                else:
                    # Without weights, simply sample donor values
                    donor_values = donor_cell[self.imputation_var].dropna().values
                    selected_values = np.random.choice(donor_values, size=len(recipient_cell), replace=True)

                # Set imputed var values
                recipient_cell[f'imp_{self.imputation_var}'] = selected_values.copy()
                # Add the imputed recipient cell to the list
                imputed_recipient_cells.append(recipient_cell)

            else:
                # If no donors are available, imputation is not performed (or can apply other fallback logic here)
                print(f"No donors available for {i}, global mean applied")
                recipient_cell[f'imp_{self.imputation_var}'] = np.average(self.donor_data[self.imputation_var], 
                                                                          self.donor_data[self.weight_var])

                # Add the imputed recipient cell to the list
                imputed_recipient_cells.append(recipient_cell)

        # Combine all the imputed recipient cells into one DataFrame
        self.recipient_data = pd.concat(imputed_recipient_cells)

        return