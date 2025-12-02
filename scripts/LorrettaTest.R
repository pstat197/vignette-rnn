library(here)
library(tidyverse)

# load data
# set folder where data is saved
folder <- 'data'

# read csv
imdb <- read_csv(here(folder, 'IMDB Dataset.csv'))
