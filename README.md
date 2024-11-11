This repo contains R and Python code that reads the output of DYNASIM's Family and Earnings History (FEH) module. 

## Python

The Python reader is implemented in the module Python/read_feh.py. The function `read_feh_data_file()` is all that is needed for reading a DYNASIM file.

The following is a usage example:

```
import Python.read_feh as feh

header_file = 'C:/Users/dcosic/Documents/Dynasim/Dynasim-core/run/dynasipp_header_even.dat'
person_file = 'C:/Users/dcosic/Documents/Dynasim/Dynasim-core/run/dynasipp_person_even.dat'
family_file = 'C:/Users/dcosic/Documents/Dynasim/Dynasim-core/run/dynasipp_family_even.dat'

# Read 10 records from each person and family file
perdata = feh.read_feh_data_file(header_file, person_file, file_type='person', count=10)
famdata = feh.read_feh_data_file(header_file, family_file, file_type='family', count=10)

```

## R
For the R reader, its main piece is the code in `read_codebook.R` that reads DYNASIM's codebook and constructs column names for the data frame. 

DYNASIM codebook specifies the logical layout of the person and family records. A record generally includes so called *static* variables, which contain a single value, and *micro time-series*, which contain values for a range of years. For each variable on a record, the codebook specifies its name, starting position, and end position. For micro time-series, the codebook also specifies the year range.

> [!IMPORTANT]
> All variables are 32-bit integers. The values of starting and ending position of a variable are not used.

A sample codebook is in `codebook_2087ds.sipp2006`. To obtain column names, execute the following code:

```
library(tidyverse)
source("read_codebook.R")
rec_struct <- get_col_names(paste0(here::here(), '/codebook_2087ds.sipp2006'))
```

The `read_feh.R` file contains code that reads a DYNASIM output file, but this repo does not contain any sample output files.

## Data

### Sex

Sex is encoded in variable SEX.

* Men:   SEX==1
* Women: SEX==2

### Race and Ethnicity

Race and ethnicity are encoded in variable ETHNCTY.

* Asian:    4000 <= ETHNCTY <= 4999
* Black:    2000 <= ETHNCTY <= 2999
* Hispanic: 
  * 1014 <= ETHNCTY <= 1020 or
  * 2014 <= ETHNCTY <= 2020 or
  * 3014 <= ETHNCTY <= 3020 or 
  * 4014 <= ETHNCTY <= 4020
* White:    1000 <= ETHNCTY <= 1999            
* Other:    3000 <= ETHNCTY <= 3999

### Education

Educational attainment is encoded in the vector GRADECAT as the number of completed grades. It can take values from 0 to 18.

