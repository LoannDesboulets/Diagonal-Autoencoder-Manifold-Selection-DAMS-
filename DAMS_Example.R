## -----------------------------------------------------------
## Clear Memory
rm(list=ls())

## -----------------------------------------------------------
## Load the functions to simulate data and perform DAMS
source("Simulate_Manifold.R")
source("DAMS.R")

## -----------------------------------------------------------
## Simulate some data
X <- Simulate_Manifold()

## -----------------------------------------------------------
## Invoke the function
result <- DAMS(X)

## -----------------------------------------------------------
## Plot Paths
matplot(result$Paths,type="l",lwd=2)

## -----------------------------------------------------------
## Plot the Sparse Graphical Model
result <- DAMS(X,graph.model=TRUE) # Rerun the algorithm with graph.model
source("Plot_Graph.R")             # Customized function for plotting
G <- result$Graph[,,5]             # The graph with the halfway penalty
Plot_Graph(G)
