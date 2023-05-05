library(sgnesR)
library(abind)
library(rhdf5)
library(igraph)

simulate_periodic_signal <- function(num_time_series, mean_periodicity, num_time_points) {
    data <- NULL
    x <- 1:num_time_points
    for (i in seq_len(num_time_series)) {
        phi = rnorm(1, mean_periodicity, 0.04)
        epsilon = rnorm(num_time_points, 0, 1) * 0.2
        eta = runif(1, -pi, pi)
        f = sinpi(phi*x + eta) + epsilon
        data <- rbind(data, f)
    }
    return(data)
}

n_genes <- 20
n_time_points <- 16
num_replications <- 100
mean_periodicity <- c(0.1, 0.2, 0.3)

stacked_expression <- NULL
stacked_labels <- NULL

for (p in 1:length(periods)) {
  for (i in 1:num_replications) {
    data <- simulate_periodic_signal(n_genes, mean_periodicity[p], n_time_points)
    stacked_expression <-
      rbind(stacked_expression, data)
    data_labels <- rep(p, n_genes)
    stacked_labels <-
      c(stacked_labels, (data_labels))
  }
}

h5_file <- "data/data_simulated/sim_periodic_signal.h5"
h5createFile(h5_file)
h5createGroup(h5_file, "expression")

h5write(stacked_expression, h5_file, "expression/data")
h5write(stacked_labels, h5_file, "expression/labels")

## CODE TO READ IN FILE ##
# file <- h5read(file = "sim_periodic_signal.h5",
#                name = "expression/data")
# labels <- h5read(file = "sim_periodic_signal.h5",
#                  name = "expression/labels")
