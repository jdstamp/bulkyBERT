library(dplyr)
library(tibble)
library(sgnesR)
library(igraph)
library(ggplot2)
library(reshape2)
library(abind)
library(rhdf5)

simulate_expression_data <- function(){
  g <-sample_gnm(10,10, directed=TRUE, loops = FALSE)
  plot(g)
  g <-sample_gnp(10,0.15, directed=TRUE, loops = FALSE)
  plot(g)
  V(g)$Ppop <- (sample(100,vcount(g), rep=TRUE))
  V(g)$Rpop <- (sample(100, vcount(g), rep=TRUE))
  sm <- sample(c(1,-1), ecount(g), rep=TRUE, p=c(.8,.2))
  E(g)$op <- sm
  rp<-new("rsgns.param",time=0,stop_time=1000,readout_interval=500)
  rc <- c(0.002, 0.005, 0.005, 0.005, 0.01, 0.02)
  rsg <- new("rsgns.data",network=g, rconst=rc)
  xx <- rsgns.rn(rsg, rp, timeseries=FALSE, sample=100)
  return(xx$expression)
}
X <- t(simulate_expression_data())
ggplot(data = melt(X), aes(x = Var1, y = Var2, fill=value)) +
  geom_tile()

stacked_array <- abind(X, X, X, X, X, X, along=3)
stacked_labels <- c(0, 1, 0, 0, 1, 0)

h5_file <- "myhdf5file.h5"
h5createFile(h5_file)
h5createGroup(h5_file,"expression")

h5write(stacked_array, h5_file, "expression/data")
h5write(stacked_labels, h5_file, "expression/labels")

h5ls(h5_file)
