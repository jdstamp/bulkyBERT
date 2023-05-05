
library(tidyverse)
library(str2str)
library(rhdf5)

# pull in data and reshape
shoemaker <- read.csv("data/shoemaker_data.csv") %>% 
  remove_rownames() %>% 
  pivot_longer(-X, names_to="Sample") %>% 
  pivot_wider(names_from=X, values_from=value)

# filter to genes within top 10% variance
gene_vars <- apply(shoemaker[,-1], 2, var)
cutoff <- quantile(gene_vars, 0.9)
shoemaker <- shoemaker[,c(TRUE, gene_vars > cutoff)]

# reformat metadata
shoemaker_meta <- read.csv("data/shoemaker_meta.csv") %>% 
  select(-X.1) %>% 
  select("Sample"=X, Group, Replicate, Timepoint)

# obtain list of distinct timepoints in order
timepoints <- shoemaker_meta %>% 
  pull(Timepoint) %>% 
  unique() %>% 
  sort()

# filter out replicate with 13 timepoints
samples_summary <- shoemaker_meta %>% 
  group_by(Group, Replicate) %>% 
  summarise("n_timepoints"=n()) %>% 
  filter(n_timepoints == 14) %>% 
  ungroup()

group_labels <- samples_summary$Group
gene_labels <- colnames(shoemaker[,-1])

shoemaker_reformat <-
  map2(samples_summary$Group, samples_summary$Replicate, function(group, replicate, data) {
    # for a specific replicate, pull corresponding gene expression values
    # place into a dataframe of genes x timepoints
    ex <- shoemaker_meta %>%
      filter(Group == group, Replicate == replicate) %>%
      select(Sample, Timepoint) %>%
      left_join(shoemaker, by = "Sample") %>%
      pivot_longer(-c(Sample, Timepoint),
                   names_to = "Gene",
                   values_to = "Expression") %>%
      select(-Sample) %>%
      mutate(Timepoint = paste0("Timepoint_", as.integer(factor(
        Timepoint, levels = timepoints
      )))) %>%
      pivot_wider(names_from = Timepoint, values_from = Expression) %>%
      select(Gene, c(paste0("Timepoint_", 1:14)))
    
    # replace non-positive values with small positive number
    ex[, 2:15][ex[, 2:15] <= 0] <- 1e-5
    
    # obtain timepoints 15 and 16 by adding some noise multiplicatively
    noise_1 <- rnorm(nrow(ex), sd = 0.1)
    noise_2 <- rnorm(nrow(ex), sd = 0.1)
    ex <- ex %>%
      mutate("Timepoint_15" = Timepoint_14 * exp(noise_1)) %>%
      mutate("Timepoint_16" = Timepoint_15 * exp(noise_2))
    
    ex <- ex %>% 
      column_to_rownames("Gene")
    return(ex)
  }, data = shoemaker) %>% 
  ld2a(dim.order=c(3,1,2))

h5_file <- "data/shoemaker.h5"
h5createFile(h5_file)

h5createGroup(h5_file,"expression")
h5write(shoemaker_reformat, h5_file, "expression/data")
h5write(group_labels, h5_file, "expression/group_labels")
h5write(gene_labels, h5_file, "expression/gene_labels")
h5ls(h5_file)
