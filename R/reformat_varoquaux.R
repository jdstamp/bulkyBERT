library(tidyverse)
library(str2str)
library(stringr)
library(rhdf5)
library(timecoursedata)
library(imputeTS)

# pull in data and reshape
data(varoquaux2019leaf)
data(varoquaux2019root)

filter_and_save <- function (data, meta, filename) {
  paper_genes <- c("003G209800",
                   "003G356000",
                   "003G264400",
                   "003G364400",
                   "003G243400",
                   "006G026800",
                   "001G502000",
                   "006G026900")
  # gene_variance <- apply(data, 2, var)
  # cutoff <- quantile(gene_variance, 0.9)
  # filtered <- data[, gene_variance > cutoff]
  filtered <- data %>%
    rownames_to_column(var = "id") %>%
    pivot_longer(!id, names_to = "Gene", values_to = "Expression")
  filtered <- filtered %>%
    mutate(Gene = str_extract(Gene, "([0-9]+G[0-9]+)", group = 1)) %>%
    filter(Gene %in% paper_genes)

  meta <- meta %>%
    rownames_to_column(var = "id") %>%
    select(c("id", "Week", "Replicate", "Condition", "Sample.type", "Genotype"))
  merged <- filtered %>%
    left_join(meta, by="id") %>%
    select(-id)
  missing_postflowering <- merged %>%
    filter(Week < 9, Condition == "Control") %>%
    mutate(Condition = "Postflowering")
  merged <- merged %>%
    rbind(missing_postflowering) %>%
    distinct() %>%
    pivot_wider(names_from = "Week", values_from = "Expression") %>%
    select(Gene, Replicate, Condition, Sample.type, Genotype, as.character(2:17))
  expression_data <- merged %>%
    select(as.character(2:17))
  imputed_expression <- t(apply(expression_data, 1, na_interpolation))
  # mean_time <- apply(imputed_expression, 1, mean)
  # sd_time <- apply(imputed_expression, 1, sd)
  # cv <- sd_time / mean_time
  # cv_cutoff <- quantile(cv, 0.98, na.rm=TRUE)
  # imputed_expression_cv <- imputed_expression[cv > cv_cutoff, ]
  labels <- merged %>% #[cv > cv_cutoff,] %>%
    select(Gene, Condition, Sample.type, Genotype) %>%
    as.matrix()
  # labels <- labels[complete.cases(imputed_expression_cv),]
  # imputed_expression_cv <- imputed_expression_cv[complete.cases(imputed_expression_cv),]
  h5createFile(filename)
  h5createGroup(filename, "expression")
  h5write(imputed_expression, filename, "expression/data")
  h5write(labels, filename, "expression/labels")
  h5ls(filename)
}



filename_all <- "data/varoquaux.h5"
filename_root <- "data/varoquaux_root.h5"
filename_leaf <- "data/varoquaux_leaf.h5"
leaf_data <- data.frame(t(varoquaux2019leaf$data))
root_data <- data.frame(t(varoquaux2019root$data))
varoquaux <- leaf_data %>%
        rbind(root_data)
meta <- varoquaux2019leaf$meta %>%
  rbind(varoquaux2019root$meta)

filter_and_save(varoquaux, meta, filename_all)
filter_and_save(root_data, varoquaux2019root$meta, filename_root)
filter_and_save(leaf_data, varoquaux2019leaf$meta, filename_leaf)
