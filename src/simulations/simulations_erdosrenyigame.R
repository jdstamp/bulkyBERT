library(sgnesR)
library(abind)
library(rhdf5)
library(igraph)

generate_network <- function(n_genes, prob_edges) {
  # Generation of a random scale-free network with 20 nodes using an Erdos-Renyi network model. Time points: 15, genes:
  graph <- sample_gnp(n_genes, prob_edges, directed = TRUE)
  
  # Assigning initial values to the RNAs and protein products to each node randomly.
  V(graph)$Ppop <- (sample(100, vcount(graph), rep = TRUE))
  V(graph)$Rpop <- (sample(100, vcount(graph), rep = TRUE))
  
  ## Changes graphical structure a bit -
  # Assign -1 or +1 to each directed edge to represent that an interacting node is acting either as a
  # activator, if +1, or as a suppressor, if -1
  activation_weights <- sample(c(1,-1), ecount(graph), rep = TRUE, p = c(.8, .2))
  E(graph)$op <- activation_weights
  
  return(graph)
}


simulate_gene_expression <- function(graph, n_time_points) {

    end_time = n_time_points * 500 - 500
    # Specifying global reaction parameters. Defines the initial parameters which include “start time”, “stop time” and “read-out interval” for time series data
    rp <- new(
        "rsgns.param",
        time = 0,
        stop_time = end_time,
        readout_interval = 500
      )

    # Specifying the reaction rate constant vector for following reactions: (1) Translation rate, (2) RNA
    #degradation rate, (3) Protein degradation rate, (4) Protein binding rate, (5) unbinding rate, (6)
    #transcription rate.
    reaction_rate_constants <- c(0.002, 0.005, 0.005, 0.005, 0.01, 0.02)

    # Defines a data object for the input which includes the network topology and other parameters such as the initial populations of
    # RNA and protein molecules of each node/gene, rate constants, delay parameters and initial population parameters of different molecules.
    rsg <- new("rsgns.data", network = graph, rconst = reaction_rate_constants)
    #Call the R function for SGN simulator
    simulation_data <- rsgns.rn(rsg, rp, timeseries = FALSE, sample = n_time_points)
    return(simulation_data$expression)
}

n_genes <- 20
n_time_points <- 16
num_replications <- 100
num_networks <- 2
prob_edges_network1 <- 0.2
prob_edges_network2 <- c(0.1, 0.13)

stacked_expression <- NULL
stacked_labels <- NULL

for(p in 1:2) {
  network1 <- generate_network(n_genes, prob_edges_network1)
  for (i in 1:num_replications) {
    stacked_expression <-
      abind(stacked_expression, exp_network1, along = 3)
    stacked_labels <- rbind(stacked_labels, 0)
  }
  p_network2 <- prob_edges_network2[p]
  network2 <- generate_network(n_genes, p_network2)
  for (i in 1:num_replications) {
    exp_network2 <- simulate_gene_expression(network2, n_time_points)
    stacked_expression <-
      abind(stacked_expression, exp_network2, along = 3)
    stacked_labels <- rbind(stacked_labels, 1)
  }
  h5_file <- paste0("sim_gene_expression_",p,".h5")
  h5createFile(h5_file)
  h5createGroup(h5_file, "expression")
  
  h5write(stacked_expression, h5_file, "expression/data")
  h5write(stacked_labels, h5_file, "expression/labels")
}

## CODE TO READ IN FILE ##
#file <- h5read(file = "sim_erdosrenyi.h5",
#                 name = "expression/data")
#file <- h5read(file = "sim_erdosrenyi.h5",
#                 name = "expression/labels")
