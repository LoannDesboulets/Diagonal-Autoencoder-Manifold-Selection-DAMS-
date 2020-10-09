Plot_Graph <- function(G                                   ,
                       names     = NULL                    ,
                       title     = "Sparse Graphical Model",
                       circ.size = 5                       ,
                       text.size = 1                       ){
  

    

  
# ---------------------------------------------------------------------------------------------------------------#
#                                                                                                                #
#                                                   Plot Graph                                                   #
#                                                                                                                #
#                                                                                                                #
#   This function plot a sparse graphical model from adjacency matrix G, provided by LLMS() or DAMS().           #                                                                             #
#                                                                                                                # 
# ---------------------------------------------------------------------------------------------------------------#

  
  
  
  
## -------------------------------------
## Check the input of the functions and
## throw error if needed
if(!is.matrix(G)){
  stop("ERROR: The graph 'G' is not a matrix.")
}
if(!isSymmetric(G)){
  stop("ERROR: The graph 'G' is not symmetric.")
}
  
  ## --------
  ## Settings
  require(viridis)
  p = nrow(G)
  
  ## ---------------------------
  ## Gives names to the vertices
  if(is.null(names)){
    names <- rownames(G)
  }
  if(is.null(names)){
    names <- paste0("X",1:p)
  }
  
  # Sequence of equally spaced points
  t <- ppoints(p)
  loc <- cbind(sin(2*pi*t),cos(2*pi*t))
  
  # Initialize the plot
  plot(0, 
       xaxt='n'    ,
       yaxt='n'    ,
       bty = 'n'   , 
       pch = ''    , 
       ylab = ''   , 
       xlab = ''   , 
       xlim=c(-1,1),
       ylim=c(-1,1),
       main=title)
  
  # Connect edges
  for(j1 in 1:p){
    for(j2 in 1:j1){
      if(j1!=j2){
          color <- rgb(255,0,0,max=255,alpha=abs(G[j1,j2])*255)
          x = c(loc[j1,1],loc[j2,1])
          y = c(loc[j1,2],loc[j2,2])
          lines(x,y,col=color,lwd=abs(G[j1,j2])*5)
      }
    }
  }
  
  # Plot the points over a circle
  color = viridis(1.5*p)
  points(loc,cex=circ.size,xaxt='n',yaxt='n',bty='n',xlab="",ylab="",type="p",pch=21,bg=color,lwd=2)
  for(j in 1:p){
      text(loc[j,1],loc[j,2],names[j],cex=text.size,col="white")
  }

}