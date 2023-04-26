################################ CODE TO RUN sgnesR using erdos.renyi.game ################################

## Load library
library(sgnesR)

network_generator <- function(ngen){
                        ##Generation of a random scale-free network with 20 nodes using an Erdos-Renyi network model. Time points: 15, genes:
                        g <- erdos.renyi.game(ngen,.15, directed=TRUE)
                        
                        # Assigning initial values to the RNAs and protein products to each node randomly.
                        V(g)$Ppop <- (sample(100,vcount(g), rep=TRUE))
                        V(g)$Rpop <- (sample(100, vcount(g), rep=TRUE))
                        return(g)} #G is the graph structure and it is different for each erdo.renyi.game function

simulation <- function(g, timep) {## Changes graphical structure a bit - 
                # Assign -1 or +1 to each directed edge to represent that an interacting node is acting either as a
                #activator, if +1, or as a suppressor, if -1
                sm <- sample(c(1,-1), ecount(g), rep=TRUE, p=c(.8,.2))
                E(g)$op <- sm
                
                # Specifying global reaction parameters. Defines the initial parameters which include “start time”, “stop time” and “read-out interval” for time series data
                rp<-new("rsgns.param",time=0,stop_time=1000,readout_interval=500)
                
                # Specifying the reaction rate constant vector for following reactions: (1) Translation rate, (2) RNA
                #degradation rate, (3) Protein degradation rate, (4) Protein binding rate, (5) unbinding rate, (6)
                #transcription rate.
                rc <- c(0.002, 0.005, 0.005, 0.005, 0.01, 0.02)
                
                # defines a data object for the input which includes the network topology and other parameters such as the initial populations of 
                # RNA and protein molecules of each node/gene, rate constants, delay parameters and initial population parameters of different molecules.
                rsg <- new("rsgns.data",network=g, rconst=rc)
                #Call the R function for SGN simulator
                xx <- rsgns.rn(rsg, rp, timeseries=FALSE, sample=timep)
                return(xx)}

genes = 20
time_points = 50
network <- network_generator(genes)
labels = 2

total_expression = data.frame(matrix(nrow = 1, ncol = time_points+1))

for (i in 1:labels){
  expression <- simulation(network, time_points)
  lab <- rep(i, dim(expression$expression)[1])
  expression <- cbind(expression$expression, lab)
  colnames(expression) <- colnames(total_expression)
  total_expression <- rbind(total_expression, expression)
}

total_expression <- total_expression[-1,]
