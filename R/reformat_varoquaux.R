library(tidyverse)
library(str2str)
library(stringr)
library(rhdf5)
library(timecoursedata)
library(imputeTS)

# pull in data and reshape
data(varoquaux2019leaf)
data(varoquaux2019root)

leaf_data <- t(varoquaux2019leaf$data)
root_data <- t(varoquaux2019root$data)
varoquaux <- data.frame(leaf_data) %>%
        rbind(data.frame(root_data))

# filter to genes within top 10% variance
gene_variance <- apply(varoquaux, 2, var)
cutoff <- quantile(gene_variance, 0.9)
data <- varoquaux[, gene_variance > cutoff]
data <- data %>%
  rownames_to_column(var = "id") %>%
  pivot_longer(names_to = "Gene", values_to = "Expression", cols = colnames(data))
# reformat metadata
meta <- varoquaux2019leaf$meta %>%
  rbind(varoquaux2019root$meta) %>%
  rownames_to_column(var = "id") %>%
  select(c("id", "Week", "Replicate", "Condition", "Sample.type", "Genotype"))


merged <- data %>%
  left_join(meta, by="id") %>%
  select(-id) %>%
  pivot_wider(names_from = "Week", values_from = "Expression") %>%
  select(Gene, Replicate, Condition, Sample.type, Genotype, as.character(2:17))

expression_data <- merged %>%
  select(as.character(2:17))

imputed_expression <- t(apply(expression_data, 1, na_interpolation))


labels <- merged %>%
  select(Gene, Condition, Sample.type, Genotype)
group_labels <- labels %>%
  unite(Group, c(Condition, Sample.type, Genotype))

h5_file <- "data/varoquaux.h5"
h5createFile(h5_file)

h5createGroup(h5_file,"expression")
h5write(imputed_expression, h5_file, "expression/data")
h5write(group_labels, h5_file, "expression/group_labels")
h5write(labels, h5_file, "expression/labels")
h5ls(h5_file)
