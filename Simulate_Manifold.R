Simulate_Manifold <- function(n          = 1000,
                              p          = 20  ,
                              active.set = 5   ,
                              r          = 1   ,
                              sigma      = 0.0 ,
                              max.corr   = 0.0 ,
                              linear     = FALSE){
  

    

  
# ---------------------------------------------------------------------------------------------------------------#
#                                                                                                                #
#                                              Simulate Manifold                                                 #
#                                                                                                                #
#                                                                                                                #
#   This function simulates a manifold, as defined in the numerical simulations design.                          #
#   C.f. section 2.6.1 of the thesis                                                                             #
#                                                                                                                # 
# ---------------------------------------------------------------------------------------------------------------#

  
  
  
  
## -------------------------------------
## Check the input of the functions and
## throw error if needed
if(any(c(n,p,active.set,r,sigma,max.corr)<0)){
  stop("ERROR: No parameter can be negative.")
}
if(r>active.set){
  stop("ERROR: The low dimension of the manifold 'r' cannot be larger than the number of active variables.")
}
if(active.set>p){
  stop("ERROR: The number of active variables cannot be larger than the number of candidates variables.")
}
if(max.corr>0.5){
  stop("ERROR: The maximum correlation between the noise variables cannot exceed 0.5.")
}
if(!is.logical(linear)){
  message("ERROR: The parameter 'linear' is mispecified, using default FALSE instead.")
  linear = FALSE
}
  
  ## --------------------
  ## Non-linear functions
  Non_Linear_Functions <- function(x,n){
  switch(n,
           y <- x,
           y <- x^2,
           y <- 0.05*exp(-0.5*(x-mean(x))^2),
           y <- cos(1.5*pi*x),
           y <- (x+1)*(x>mean(x)) - (x-3)*(x<mean(x)),
           y <- sin(2*x),
           y <- x^3,
           y <- tanh(2*x),
           y <- 0.0001*x + x*(x<mean(x)),
           y <- 0.5*x + sin(2*x)
         )
  return(y)
  }
  
  ## ---------------------------
  ## Set Random Number Generator
  t <- as.numeric(Sys.time())
  seed <- 1e8 * (t - floor(t))
  set.seed(seed)
  
  ## ---------------------------
  ## Coordinates on the Manifold
  Z <- matrix(4*runif(n*r)-2,n,r)
  A <- matrix(4*runif(r*active.set)-2,r,active.set)
  A <- t(svd(A)$v)
  
  ## ----------------------------
  ## Linear & Nonlinear Manifolds
  if(linear == TRUE){
    X1 <- Z%*%A
  }else{
    X1 <- Z%*%A
    for(j in 1:active.set){
      X1[,j] <- Non_Linear_Functions(X1[,j],j)
      X1[,j] <- X1[,j]/norm(X1[,j],"2")
    }
  }
  
  ## -------------------------
  ## Uncorrelated Normal Noise
  X1 = scale(X1)
  Noise <- sigma*matrix(rnorm(n*active.set),n,active.set)
  X1 = X1 + Noise

  ## -----------------------------
  # Multivariate Normal Candidates
  if((p-active.set)>0){
    S <- matrix(1,(p-active.set),(p-active.set))
    for(j1 in 1:(p-active.set)){
      for(j2 in 1:(p-active.set)){
        if(j1!=j2){S[j1,j2] = ((-1)^j1)*max.corr^abs(j1-j2)}
      }
    }
    require(MASS)
    X2 = mvrnorm(n=n, mu=rep(0,p-active.set), Sigma=S, empirical=FALSE)
    X2 <- X2[,sample(p-active.set)]
    X1 = scale(X1)
    X2 = scale(X2)
    X = cbind(X1,X2)
  }else{
    X1 = scale(X1)
    X <- X1
  }
  
  ## ------
  ## Output
  return(X)

}