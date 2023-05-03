
library("tidyverse")

# Load in Gene Exp Data
shoemaker_data <- read.csv("data/shoemaker_data.csv") %>% 
  remove_rownames %>% 
  column_to_rownames("X")

# Load in metadata
shoemaker_meta <- read.csv("data/shoemaker_meta.csv") %>% 
  remove_rownames %>% 
  column_to_rownames("X") %>% 
  select(-X.1)

# Check to make sure we have the correct number of timepoints for each observation
shoemaker_meta %>% 
  group_by(Group, Replicate) %>% 
  summarize(n())

# 5 different treatment groups, 3 replicates each with 14 time points
# C - Control
# K - Kawasaki strain - a low pathogenicity seasonal H1N1 influenza virus
# M - California Strain - a mildly pathogenic virus from the 2009 pandemic season
# VH - Vietnam Strain - a highly pathogenic H5N1 avian influenza virus
# VL - Vietnam Strain but at lower dosage

# Check dimensions of expression data
dim(shoemaker_data)
# 39544 rows - corresponds to genes, names are RefSeq convention
# 209 columns - 5 groups x 3 replicates x 14 time points

# VH replicate 1 is missing one time point, 3 hours
shoemaker_meta %>% 
  filter(Group == "VH", Replicate==1) %>% 
  select(Timepoint) %>% 
  arrange(Timepoint)

# Look at dist of means/vars of genes
vars <- shoemaker_data %>% apply(1, var)
summary(vars)
hist(vars)

# no genes with means near 0...
means <- shoemaker_data %>% apply(1, mean)
summary(means)
hist(means)
