"""
Class defining HotDeckImputer for imputing missing values in recipient data using donor data.

Hot deck imputation is a statistical method where missing values in a dataset
are replaced with values from a similar donor dataset. This class provides
methods for validating data, defining cells, and performing imputation.

Attributes:
- donor_data (pl.DataFrame): The source dataset for imputation.
- recipient_data (pl.DataFrame): The dataset that will receive imputed values.
- imputation_var (str): The variable to be imputed.
- weight_var (str): The variable used for weighted sampling.
- random_seed (int, optional): Seed for reproducibility of random operations.

Methods:
- generate_cells: Generates cells based on defined conditions.
- define_cells: Defines all possible cell definitions based on input variables.
- split_cell: Splits an individual cell based on a new condition.
- collapse_cell: Collapses multiple cells into a single cell based on a base condition.
- summarize_cells: Summarizes the imputation results for each cell.
- gen_analysis_file: Generates an Excel file summarizing the imputation results.
- apply_random_noise: Applies random noise to the imputed values to smooth out clustering issues.
- summarize_column: Summarizes a column in the data, returning basic statistics.
- age_dollar_amounts: Ages the imputed values to a target year using CPI indexes.

Hidden Methods:
- _validate_data: Validates the input data and parameters.
- _check_variable_consistency: Checks if the unique values and types of variables are consistent across datasets.
- _parse_condition: Parses a condition string and returns a Polars expression.
"""
import numpy as np
import polars as pl
import itertools
from statsmodels.stats.weightstats import DescrStatsW
from xlsxwriter import Workbook
import os
import hot_deck_imputer.error_detection as error_detection 

class HotDeckImputer:
    def __init__(self, donor_data:pl.DataFrame, 
                 imputation_var:str, weight_var:str,
                 recipient_data:pl.DataFrame,
                 random_seed:int = None):
        """
        Initialize the HotDeckImputer class with donor and recipient datasets.

        Parameters:
        - donor_data (pl.DataFrame): The dataset providing values for imputation.
        - imputation_var (str): The variable to be imputed.
        - weight_var (str): The variable used for weighted sampling.
        - recipient_data (pl.DataFrame): The dataset receiving the imputed values.
        - random_seed (int, optional): Seed for random number generation to ensure reproducibility.

        Notes:
        - The donor and recipient data are cloned to avoid modifying the original datasets.
        """
        # Set attributes for the class
        self.donor_data = donor_data.clone()
        self.imputation_var = imputation_var
        self.weight_var = weight_var
        self.recipient_data = recipient_data.clone()
        self.random_seed = random_seed

        # Cell definition attributes to be defined in methods
        self.cell_definitions = None
        self.donor_cells = None
        self.recipient_cells = None

        # Random noise attributes defined in methods
        self.random_noise = None

        # Validate input data
        self._validate_data()

    def _validate_data(self):
        """
        Validate the input data and parameters.
        """
        error_detection.validate_data(self)
        return
            
    def _check_variable_consistency(self, variables:list):
        """
        Non-callable method to check if the unique values and types of the variables
        used for cell definition are the same in donor and recipient datasets.
        :param variables: List of variables to check
        :raises TypeError: If data types do not match between donor and recipient
        :raises ValueError: If unique values do not match between donor and recipient
        """
        error_detection.check_variable_consistency(self, variables)
        return
    
    def _parse_condition(self, condition:str):
        """
        Parse a condition string and return a Polars expression.
        :param condition: The condition string to parse.
        :type condition: str
        :return: The Polars expression.
        :rtype: pl.Expr
        :raises: None
        """
        # Remove outer parentheses
        condition = condition.strip("()")
        
        # Split the condition into individual criteria
        criteria = condition.split(" & ")
        
        # Initialize combined expression with a default "true" condition
        combined_expression = pl.lit(True)
        
        # Parse each criterion and combine them using logical AND
        for criterion in criteria:
            # Remove any extra parentheses and spaces
            criterion = criterion.strip("()").strip()
            column, value = criterion.split("==")
            column = column.strip()
            value = value.strip().strip("'")
            # Detect the type of the value
            if value.isdigit():
                value = int(value)
            elif value.replace('.', '', 1).isdigit():
                value = float(value)
            expr = pl.col(column) == value
            # Combine the expressions
            combined_expression &= expr
        
        return combined_expression

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

        # Remove the extra recipient_cells = statement
        for i, condition in enumerate(self.cell_definitions):
            # Create cell based on condition
            filter_expr = self._parse_condition(condition)
            # Filter the donor and recipient data based on the condition
            donor_cells[f'{condition}'] = self.donor_data.filter(filter_expr)
            recipient_cells[f'{condition}'] = self.recipient_data.filter(filter_expr)

        self.donor_cells = donor_cells
        self.recipient_cells = recipient_cells
        return
    
    def define_cells(self, variables:list):
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
            f"{variables[i]} == '{combination[i]}'" if isinstance(combination[i], str) else f"{variables[i]} == {combination[i]}"
            for i in range(len(combination))
            ]
            cell_definitions.append(' & '.join(conditions))

        self.cell_definitions = cell_definitions
        return 

    def split_cell(self, cell_condition:str, split_column:str):
        """
        Method to split an individual cell further based on a new condition.
        :param cell_condition: A condition string representing the cell to be split.
        :param split_column: The column to check for unique values to split the cell.
        :return: None
        """
        # Get the data for the cell that is going to be split
        split_donor = self.donor_cells[cell_condition]
        split_recipient = self.recipient_cells[cell_condition]
        
        # Get unique values in the split column
        unique_values = split_donor.select(split_column).unique().to_series().to_list()
        
        # Remove the original cell from the donor and recipient cell dictionaries
        del self.donor_cells[cell_condition]
        del self.recipient_cells[cell_condition]

        # Split the cell based on unique values in the split column
        for value in unique_values:
            split_condition = f"{split_column} == {value}"
            combined_condition = f"{cell_condition} & {split_condition}"
            split_expr = self._parse_condition(combined_condition)
            
            # Add the newly split cells into the dictionaries
            self.donor_cells[combined_condition] = split_donor.filter(split_expr)
            self.recipient_cells[combined_condition] = split_recipient.filter(split_expr)
            
            # Update cell definitions
            self.cell_definitions.append(combined_condition)
        
        # Remove the original cell condition from cell definitions
        self.cell_definitions.remove(cell_condition)

        print("Cell splitting completed successfully.")
        print("To rerun with these cells submit imputer.impute() again.")
        return
    
    def collapse_cell(self, base_condition: str):
        """
        Method to collapse multiple cells into a single cell based on a base condition.
        :param base_condition: A condition string representing the base condition to identify cells to collapse.
        :return: None
        """
        # Initialize combined donor and recipient data
        combined_donor = None
        combined_recipient = None

        # Identify all matching cells in cell_definitions
        matching_cells = [
                condition for condition in self.cell_definitions
                if base_condition in condition 
            ]
        
        # Identify the conditions to be collapsed and repopulate concatenated recipient + donor data
        for match in matching_cells:
                if combined_donor is None:
                    combined_donor = self.donor_cells[match]
                    combined_recipient = self.recipient_cells[match]
                else:
                    combined_donor = combined_donor.vstack(self.donor_cells[match])
                    combined_recipient = combined_recipient.vstack(self.recipient_cells[match])

                # Remove the old cells from the donor and recipient cell data
                del self.donor_cells[match]
                del self.recipient_cells[match]

                # Remove the old cell condition from cell definitions
                self.cell_definitions.remove(match)

        self.donor_cells[base_condition] = combined_donor
        self.recipient_cells[base_condition] = combined_recipient

        # Update cell definitions
        self.cell_definitions.append(base_condition)

        print("Cell collapsing completed successfully.")
        print("To rerun with these cells submit imputer.impute() again.")
        return

    
    def summarize_cells(self):
        results = {}
        for i, recipient_cell in self.recipient_cells.items():

            # Donor stat generation
            donor_cell = self.donor_cells.get(i)
            source_var = donor_cell[self.imputation_var]

            if self.weight_var in donor_cell.columns:
                donor_stats = DescrStatsW(source_var, weights=donor_cell[self.weight_var], ddof=0)
            else:
                donor_stats = DescrStatsW(source_var, ddof=0)

            # Recipient stat generation
            source_var = recipient_cell[f'imp_{self.imputation_var}']
            if self.weight_var in recipient_cell.columns:
                recipient_stats = DescrStatsW(source_var, weights=recipient_cell[self.weight_var], ddof=0)
            else:
                recipient_stats = DescrStatsW(source_var, ddof=0)

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
                    np.float64(donor_cell.shape[0])                  # Observations for donor
                ],

                'imp': [
                    recipient_stats.mean - 1.96 * recipient_stats.std_mean,  # 95% CI low for imp
                    recipient_stats.mean,                                    # Mean for imp
                    recipient_stats.mean + 1.96 * recipient_stats.std_mean,  # 95% CI high for imp
                    recipient_stats.std,                                     # Stddev for imp
                    recipient_stats.var,                                     # Variance for imp
                    recipient_stats.std_mean,                                # Std error for imp
                    recipient_stats.sum_weights,                             # Weighted sum for imp
                    np.float64(recipient_cell.shape[0])                      # Observations for imp
                ]
            }

            # Convert dictionary to DataFrame
            stats_df = pl.DataFrame(data)
            stats_df = stats_df.with_columns((stats_df['imp'] - stats_df['donor']).alias('diff'))
            stats_df = stats_df.with_columns((stats_df['imp']/stats_df['donor']).alias('imp_to_donor_ratio'))

            results[i] = stats_df

        return results

    def gen_analysis_file(self, out_file:str, out_path:str =''):
        """
        Generate an analysis file summarizing the imputation results.
        :param out_file (str): Name of the output file.
        :param out_path (str): Path to save the output file.
        :return: None
        """
        if out_path == '':
            out_path = '.'
        # Ensure the output directory exists
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"The directory '{out_path}' does not exist.")

        # Construct the full file path
        full_path = os.path.join(out_path, f'{out_file}.xlsx')

        # Get dictionary of DFs for each cell
        data = self.summarize_cells()
        
        # Get iterator for worksheet locations
        row = 1
        col = 0

        # Iterate through each cell's data
        with Workbook(full_path) as wb:  
            ws = wb.add_worksheet('Summary')
            for key, df in data.items():
                ws.write(row-1, col, key)
                # Write table to excel  
                df.write_excel(workbook = wb, 
                               worksheet = ws,
                               position = (row, col),
                               table_style="Table Style Light 1",
                               autofit = True)
            
                # 2 row gap between each DF's results
                row = row + df.shape[0] + 3
        print(f"Cell data written to '{out_path}\\{out_file}.xlsx'.")

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
            donor_cell = donor_cell.sort(by=self.imputation_var)

            # Calculate the next neighbor values
            donor_cell = donor_cell.with_columns(
                donor_cell[self.imputation_var].shift(-1).alias('next_val')
            )

            # Compute the distance to prior and next neighbor
            donor_cell = donor_cell.with_columns(
                (donor_cell['next_val'] - donor_cell[self.imputation_var]).alias('next_distance')
            )
            
            # Calculate the average neighbor distance for the cell, ignoring NaN values
            ## First get mean of each row and then get mean of the result
            mean_distance = donor_cell['next_distance'].mean()

            # Calculate noise level as a proportion of the mean distance between neighbors
            noise_stdev = mean_distance * variation_stdev

            # Calculate the threshold value based on relevant floor for asset tests
            if floor_noise is not None:
                threshold = floor_noise
            else:
                threshold = self.donor_data[f'{self.imputation_var}'].min()

            # Generate random noise for each recipient in the cell
            # Only apply this random noise for those who are less than some factor of the standard deviation of neighboring distances
            # i.e. if floor_stdev_multiplier = 2, observations with <2x the standard deviation of neighboring distances are left alone
            recipient_cell = self.recipient_cells[condition]

            # identify the observations who are above the threshold, who will have random noise added
            # when there is no thresholding by floor_stdev_multiplier, this is handled by the minimum identified above
            ge_thresh = recipient_cell[f'imp_{self.imputation_var}'] >= threshold
            noise = np.random.normal(loc=0, scale=noise_stdev, size=recipient_cell.shape[0])
            
            # Indicate to user that noise was not generated if all values are below the threshold
            if ge_thresh.sum() == 0:
                print(f'\nCell:\n{condition}')
                imputation_var_column = f'imp_{self.imputation_var}'
                print(f'NO NOISE GENERATED for cell due to thresholding.\n' 
                        f'All values are below the threshold of {threshold}\n'
                        f'Mean value of cell observations for imp_{self.imputation_var}: ' 
                        f'{recipient_cell[imputation_var_column].mean()}')

            # Apply noise to the imputed liquid assets in the recipient cell
            recipient_cell = recipient_cell.with_columns(
                pl.when(ge_thresh)
                .then(pl.col(f'imp_{self.imputation_var}') + noise)
                .otherwise(pl.col(f'imp_{self.imputation_var}'))
                .alias(f'imp_{self.imputation_var}')
            )

            # Ensure that values that have noise applied are not below the minimum donor value
            min_donor_val = donor_cell[self.imputation_var].min()
            recipient_cell = recipient_cell.with_columns(
                pl.col(f'imp_{self.imputation_var}')
                .clip(lower_bound = min_donor_val)
                .alias(f'imp_{self.imputation_var}')
            )

            # Update recipient data with noisy values
            self.recipient_cells[condition] = recipient_cell.with_columns(
                pl.col(f'imp_{self.imputation_var}')
            )
            imputed_recipient_cells.append(recipient_cell)
        # Store the variation standard deviation parameter
        self.random_noise = variation_stdev
        self.recipient_data = pl.concat(imputed_recipient_cells)
        
        return

    def summarize_column(self, data, column_name):
        """
        Summarize a column in data, returning basic statistics.
        :param column_name: The column to summarize
        :return: A dictionary with summary statistics
        """
        # Check if the column exists in the DataFrame
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' does not exist in '{data}'.")
        
        # Calculate summary statistics
        summary_stats = {
            'mean': data[column_name].mean(),
            'median': data[column_name].median(),
            'min': data[column_name].min(),
            'max': data[column_name].max(),
            'std_dev': data[column_name].std(),
            'count': data[column_name].count(),
            'missing_values': data[column_name].is_null().sum()
        }

        return summary_stats


    '''TODO: convert into a plotting method to see the imputed values vs. donor values
    import seaborn as sns 
    import matplotlib.pyplot as plt

    # Compare the input to the output in the process
    plot_data = imputer.recipient_data[['perwt','imp_liquid_assets']]
    plot_data_donor = imputer.donor_data[['perwt','liquid_assets']]
    plot_data['log_liquid_assets'] = np.log(plot_data['imp_liquid_assets'])
    plot_data_donor['log_liquid_assets'] = np.log(plot_data_donor['liquid_assets'])

    # Create a weighted histogram plot for donor data
    sns.kdeplot(data=plot_data_donor, x='log_liquid_assets', weights='perwt', color='green', label='Donor', alpha=0.5)

    # Create a weighted histogram plot for recipient data
    sns.kdeplot(data=plot_data, x='log_liquid_assets', weights='perwt', color='blue', label='Recipient', alpha=0.5)

    # Add titles and labels
    plt.title('Logged liquid assets post noise injection')
    plt.xlabel('Logged Liquid Assets Value')
    plt.ylabel('Weighted Density')

    # Add a legend to differentiate between recipient and donor data
    plt.legend()

    # Show the plot
    plt.show()
    '''
    
    def age_dollar_amounts(self, donor_year_cpi, imp_year_cpi):
        """
        Age the imputed values to the target year. Relevant when the source data and target data differ.
        https://www.cbo.gov/data/budget-economic-data#4 for CPI indexes
        """
        
        print(f'Summary of {self.imputation_var} pre CPI aging:\n{self.summarize_column(self.donor_data, self.imputation_var)}')
        scaling_factor = imp_year_cpi / donor_year_cpi

        self.donor_data = self.donor_data.with_columns(
            (pl.col(self.imputation_var) * scaling_factor).alias(self.imputation_var)
        )
        print(f'Summary of {self.imputation_var} post CPI aging:\n{self.summarize_column(self.donor_data, self.imputation_var)}')

        return

    def impute(self, replace=True, additional_vars = []):
        """
        Impute the missing values in the recipient data using the donor data for corresponding cells.
        This method assumes that both donor and recipient data have been partitioned using generate_cells.
        Parameters:
            replace (bool): Whether sampling is with replacement (default True).
            additional_vars (list): List of additional variables to pull from donors in the imputation. 
        """
        if not self.cell_definitions:
            raise ValueError("Cell definitions are not provided")
        
        # List to hold imputed recipient cells
        imputed_recipient_cells = []

        # For each recipient cell, find the corresponding donor cell and perform imputation
        for condition, recipient_cell in self.recipient_cells.items():
            donor_cell = self.donor_cells.get(condition)
                    # Set the random seed if provided
            if self.random_seed is not None:
                np.random.seed(self.random_seed)

            if donor_cell is not None and not donor_cell.shape[0] == 0:
                # Perform weighted random selection for the required number of values
                if self.weight_var:
                    weights = np.asarray(donor_cell[self.weight_var]).astype('float64')
    
                    # Randomly select `missing_count` values from the donor set using the weights
                    # Using weighted selection according to probability proportional to weights https://documentation.sas.com/doc/en/statcdc/14.2/statug/statug_surveyimpute_details25.htm#statug.surveyimpute.weightedDet
                    selected_indices = np.random.choice(np.arange(donor_cell.shape[0]), size=len(recipient_cell), replace=replace, p=weights / weights.sum())

                else:
                    # Without weights, simply sample donor values
                    selected_indices = np.random.choice(np.arange(donor_cell.shape[0]), size=len(recipient_cell), replace=replace)
                    
                # After getting the indices, extract the target variable value and, if there are additional variables, extract those as well
                selected_values = donor_cell[selected_indices]

                # Add the imputed values to the recipient cell
                recipient_cell = recipient_cell.with_columns(
                    pl.Series(f'imp_{self.imputation_var}', selected_values[self.imputation_var])
                )

                # Extract and add the imputed values for additional variables
                for var in additional_vars:
                    if var in donor_cell.columns:
                        selected_values = donor_cell[selected_indices][var]
                        recipient_cell = recipient_cell.with_columns(
                            pl.Series(f'imp_{var}', selected_values)
                        )
                    else:
                        print(f"Warning: Variable '{var}' not found in donor data. Skipping.")

                # Add the imputed recipient cell to the list
                imputed_recipient_cells.append(recipient_cell)
                self.recipient_cells[condition] = recipient_cell.clone()

            else:
                # If no donors are available, imputation is not performed (or can apply other fallback logic here)
                print(f"No donors available for {condition}, global mean applied")
                recipient_cell = recipient_cell.with_columns(
                    pl.lit(np.average(self.donor_data[self.imputation_var], 
                                      weights=self.donor_data[self.weight_var])
                            ).alias(f'imp_{self.imputation_var}')
                                                                                            )   
                # Skip additional_vars since there are no donor cells
                print(f"Skipping additional_vars for {condition} as no donor cells are available.")
                for var in additional_vars:
                    if var in donor_cell.columns:
                        recipient_cell = recipient_cell.with_columns(
                            pl.lit(None).alias(f'imp_{var}')
                        )
                    else:
                        print(f"Warning: Variable '{var}' not found in donor data. Skipping.")
                # Add the imputed recipient cell to the list
                imputed_recipient_cells.append(recipient_cell)
                self.recipient_cell = recipient_cell.clone()
        
        for item in imputed_recipient_cells:
            print(item)

        # Combine all the imputed recipient cells into one DataFrame
        self.recipient_data = pl.concat(imputed_recipient_cells)

        return