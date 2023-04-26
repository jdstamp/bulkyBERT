library(dplyr)
library(sgnesR)
library(igraph)

simulate_expression_data <- function(){
  g <-sample_gnm(10,10, directed=TRUE, loops = FALSE)
  plot(g)
  g <-sample_gnp(10,0.15, directed=TRUE, loops = FALSE)
  plot(g)
  V(g)$Ppop <- (sample(100,vcount(g), rep=TRUE))
  V(g)$Rpop <- (sample(100, vcount(g), rep=TRUE))
  sm <- sample(c(1,-1), ecount(g), rep=TRUE, p=c(.8,.2))
  E(g)$op <- sm
  rp<-new("rsgns.param",time=0,stop_time=10,readout_interval=10)
  rc <- c(0.002, 0.005, 0.005, 0.005, 0.01, 0.02)
  rsg <- new("rsgns.data",network=g, rconst=rc)
  xx <- rsgns.rn(rsg, rp, timeseries=FALSE, sample=100)
  matplot(t(xx$expression), type = "l")
  return(xx$expression)
}
X <- simulate_expression_data()
expression_data <- data.frame(X) %>%
  mutate(label = 0)

expression_data <- data.frame(X) %>%
  mutate(label = 1) %>%
  bind_rows(expression_data)

expression_data <- data.frame(X) %>%
  mutate(label = 2) %>%
  bind_rows(expression_data)

expression_data <- data.frame(X) %>%
  mutate(label = 3) %>%
  bind_rows(expression_data)

expression_data <- data.frame(X) %>%
  mutate(label = 4) %>%
  bind_rows(expression_data)

write.csv(expression_data, file = gzfile("train_data.csv.gz"))
