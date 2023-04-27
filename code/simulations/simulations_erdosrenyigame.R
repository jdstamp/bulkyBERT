library(sgnesR)
library(abind)
library(rhdf5)

network_generator <- function(n_genes){
                        ##Generation of a random scale-free network with 20 nodes using an Erdos-Renyi network model. Time points: 15, genes:
                        g <- erdos.renyi.game(n_genes,.15, directed=TRUE)

                        # Assigning initial values to the RNAs and protein products to each node randomly.
                        V(g)$Ppop <- (sample(100,vcount(g), rep=TRUE))
                        V(g)$Rpop <- (sample(100, vcount(g), rep=TRUE))
                        return(g)
 } #G is the graph structure and it is different for each erdo.renyi.game function

simulation <- function(g, timep) {## Changes graphical structure a bit -
                # Assign -1 or +1 to each directed edge to represent that an interacting node is acting either as a
                #activator, if +1, or as a suppressor, if -1
                sm <- sample(c(1,-1), ecount(g), rep=TRUE, p=c(.8,.2))
                E(g)$op <- sm

                end_time = timep * 500 - 500
                # Specifying global reaction parameters. Defines the initial parameters which include “start time”, “stop time” and “read-out interval” for time series data
                rp<-new("rsgns.param",time=0,stop_time=end_time,readout_interval=500)

                # Specifying the reaction rate constant vector for following reactions: (1) Translation rate, (2) RNA
                #degradation rate, (3) Protein degradation rate, (4) Protein binding rate, (5) unbinding rate, (6)
                #transcription rate.
                rc <- c(0.002, 0.005, 0.005, 0.005, 0.01, 0.02)

                # defines a data object for the input which includes the network topology and other parameters such as the initial populations of
                # RNA and protein molecules of each node/gene, rate constants, delay parameters and initial population parameters of different molecules.
                rsg <- new("rsgns.data",network=g, rconst=rc)
                #Call the R function for SGN simulator
                xx <- rsgns.rn(rsg, rp, timeseries=TRUE)
                return(xx)
 }

genes = 20
time_points = 15
network <- network_generator(genes)
l = 100
num_networks = 2

stacked_labels <- NULL
stacked_expression <- NULL

for (g in 1:num_networks){
  for (i in 1:l){
    exp <- simulation(network, time_points)
    stacked_expression <- abind(exp$expression, stacked_expression, along = 3)
    stacked_labels <- rbind(g, stacked_labels)
  }
}

h5_file <- "sim_erdosrenyi.h5"
h5createFile(h5_file)
h5createGroup(h5_file,"expression")

h5write(stacked_expression, h5_file, "expression/data")
h5write(stacked_labels, h5_file, "expression/labels")

## CODE TO READ IN FILE ##
#file <- h5read(file = "sim_erdosrenyi.h5",
#                 name = "expression/data")
#file <- h5read(file = "sim_erdosrenyi.h5",
#                 name = "expression/labels")

