DAMS <- function(X                           ,
                 hidden.layers   = c(7,2,7)  ,
                 max.iter        = 2e3       ,
                 batch.size      = 0.2       ,
                 selection.start = 250       ,
                 learning.rate   = 0.05      ,
                 ensemble.size   = 100       ,
                 graph.model     = FALSE){

    

  
# ---------------------------------------------------------------------------------------------------------------#
#                                                                                                                #
#                           Diagonal Autoencoder Manifold Selection (DAMS)                                       #
#                                                                                                                #
#                                                                                                                #
#   This function implements the Diagonal Autoencoder Manifold Selection estimator, as well as the graphical     #
#   representation. In the first case, the output is a vector of selection probabilities, of the same length     #
#   as the number of variables. In the second case, the output is a matrix of selection probabilities, that      #
#   represents the edges of the graph.                                                                           #
#                                                                                                                # 
# ---------------------------------------------------------------------------------------------------------------#

  
  
  
  
## -------------------------------------
## Check the input of the functions and
## throw error if needed
X <- data.matrix(X)
index_not_numeric = which(is.na(as.numeric(X)))
if(length(index_not_numeric)>0){
  stop(paste("Input data X is not numeric at index position(s): ", index_not_numeric))
}
n = nrow(X)
p = ncol(X)
if(any(c(hidden.layers,max.iter,batch.size,selection.start,learning.rate,ensemble.size)<0)){
  stop("ERROR: No parameter can be negative.")
}
if(batch.size>1){
  stop("ERROR: The parameter 'batch.size' cannot be larger than 1.")
}
if(!is.logical(graph.model)){
  message("ERROR: The parameter 'graph.model' is mispecified, using default FALSE instead.")
  graph.model = FALSE
}

## ----------------------------------------------
## Change batch.size into its integer counterpart 
batch.size = floor(batch.size*n)

## ------------------------------------
## Group the input of the DAMS function
settings <- data.frame(max.iter        = max.iter         ,
                       every.iter      = floor(max.iter/5),
                       batch.size      = batch.size       ,
                       selection.start = selection.start  ,
                       lr              = learning.rate    )

## -----------------------------
## Function for a single network
singleNN <- function(x,h,settings){

    # ---------------------------------
    # -- Ensure data format as matrices
    y = x
    x <- as.matrix(x)
    y <- as.matrix(y)
    
    # -----------------
    # -- Get dimensions
    n <- dim(x)[1]
    k1 <- dim(x)[2]
    k2 <- dim(y)[2]
    
    # ----------------------------------------------------------------------------------------------
    # -- Store the full layer dimension (add the dimension of input to it for simplicity afterwards)
    h <- c(k1,h)
    nh <- length(h)

    # -------------------
    # -- Standardize data
    x = scale(x)
    y = scale(y)
    x[is.na(x)] = 0

    # -------------------------------------------------------------------------------
    # -- Maximum amount of explained variance (it simplifies because y is normalized)
    SSR_max = sum(k2*(n-1))

    # ---------------------
    # -- Weights boundaries
    a = -1/sqrt(max(h))
    b = 1/sqrt(max(h))

    # --------------------
    # -- Selection Weights
    gamma1 = rep(0,k1)
    Ws1 = diag(1/(1+exp(-100*gamma1)))
    Ws2 = Ws1

    # --------------------
    # -- Network Weights
    W_in = matrix((b-a)*runif((k1+1)*h[1])+ a, k1+1, h[1])
    W_layer = vector(mode="list",length=nh-1)
    for(j in 1:(nh-1)){
        W_layer[[j]] <- matrix((b-a)*runif((h[j]+1)*h[j+1])+a, h[j]+1, h[j+1])
    }
    W_out   = matrix((b-a)*runif( (h[nh]+1)*k2)+a, h[nh]+1, k2)

    # ---------------------------
    # -- Normalized Learning rate
    a = settings$lr/settings$batch.size

    # ----------------------------------------------------------
    # -- Pre-allocation of Neuron Layers and Weights Derivatives
    S_layer = vector(mode="list",length=nh-1)
    Z_layer = vector(mode="list",length=nh-1)
    D_layer = vector(mode="list",length=nh-1)
    d_layer = vector(mode="list",length=nh-1)
    
    # --------------------------
    # -- Pre-allocation of Paths
    Paths <- NULL

    # --------------------------------------------------------------------------------
    # -- Pre-allocation of iteration counter (iter) and sum of squared residuals (SSR)
    iter = 1
    SSR = 0

    # ----------------------------------------
    # -- Stochastic Gradient Descent Algorithm
    while(iter <= settings$max.iter){
      
        # -------------------------------------------------------------
        # -- Random batch with replacement (smaller sampling bootstrap)
        batch <- sample(1:n,settings$batch.size,replace=TRUE)
        
        # ------------------------------------------------------
        # -- Input Layer with Selection (no activation function)
        S_in = x[batch,]%*%Ws1
        
        # --------------
        # -- Deep Layers
        S_layer[[1]] = cbind(S_in,1)%*%W_in
        Z_layer[[1]] = tanh(S_layer[[1]])
        for(j in 2:nh){
            S_layer[[j]] = cbind(Z_layer[[j-1]],1)%*%W_layer[[j-1]]
            Z_layer[[j]] = tanh(S_layer[[j]])
        }
    
        # ------------------------------------------------------
        # -- Ouput Layer with Selection (no activation function)
        S_out = cbind(Z_layer[[nh]],1)%*%W_out
        y_est = S_out%*%Ws2
        
        # --------
        # -- Error
        E = (y_est - y[batch,])
        E[is.na(E)] = 0
        
        # ----------------------------------------------------------------
        # -- Approximate SSR of the new iteration (with correction factor)
        new_SSR = sum(colSums(E^2)*diag(Ws2)/sum(Ws2)*k2)*(n/settings$batch.size)
        crit = abs(new_SSR - SSR)
        SSR = new_SSR
        
        # ----------------------------------------
        # -- Record evolution of selection weights
        Paths <- cbind(Paths,diag(Ws1))
        
        # ---------------------------------------------------------
        # -- Progressive displays to keep track of the optimization
        if(iter %% settings$every.iter==0) {
    
          # ---------------------------------------------------
          # -- Error Rate = explained variance / total variance
          print(paste0('Epoch :',iter,', Error Rate: ', round(SSR/SSR_max,3)))
          
        }
        
        # -------------------------
        # -- Next counter iteration
        iter = iter+1
        
        # -------------------------------
        # -- Backpropagation Computations
        D_out = t(E%*%Ws2)
        D_layer[[nh]] = t(1-Z_layer[[nh]]^2)*(W_out[1:h[nh],]%*%D_out)
        for(j in (nh-1):1){
            D_layer[[j]] = t(1-Z_layer[[j]]^2)*(W_layer[[j]][1:h[j],]%*%D_layer[[j+1]])
        }
        d_out =  t(D_out%*%cbind(Z_layer[[nh]],1))
        for(j in 2:nh){
            d_layer[[j-1]] = t(D_layer[[j]]%*%cbind(Z_layer[[j-1]],1))
        }
        d_in = t(D_layer[[1]]%*%cbind(S_in,1))
        ds1 = (W_in[1:h[1],]%*%D_layer[[1]])%*%x[batch,]
        ds2 = t(E)%*%S_out
        d_gamma2 = diag(ds2)*diag(100*Ws2*(1-Ws2))
        d_gamma1 = diag(ds1)*diag(100*Ws1*(1-Ws1))
        
        # -----------------
        # -- Weights Update
        W_in = W_in - a*d_in
        for(j in 1:(nh-1)){
            W_layer[[j]] = W_layer[[j]] - a*d_layer[[j]]
        }
        W_out = W_out - a*d_out
        if(iter>settings$selection.start){
          gamma1 = gamma1 - a*(d_gamma1+d_gamma2)
          Ws1 = diag(1/(1+exp(-gamma1)))
          Ws2 = Ws1
        }
        
    ## ----------
    ## End of SGD
    }

## -------------------------------------------------------------------
## Reevaluate the network on the whole data to compute the derivatives
## Only useful for the graph representation.
if(graph.model){
  S_in = x%*%Ws1
  S_layer[[1]] = cbind(S_in,1)%*%W_in
  Z_layer[[1]] = tanh(S_layer[[1]])
  for(j in 2:nh){
      S_layer[[j]] = cbind(Z_layer[[j-1]],1)%*%W_layer[[j-1]]
      Z_layer[[j]] = tanh(S_layer[[j]])
  }
  dydx = array(0,c(n,k1,k1))
  for(j1 in 1:k1){
    for(j2 in 1:k1){
      J= matrix(1,1,n)
      D_layer[[nh]] = t(1-Z_layer[[nh]]^2)*(W_out[1:h[nh],j1]%*%J)
      for(j in (nh-1):1){
          D_layer[[j]] = t(1-Z_layer[[j]]^2)*(W_layer[[j]][1:h[j],]%*%D_layer[[j+1]])
      }
      dydx[,j1,j2] = (W_in[j2,]%*%D_layer[[1]])
    }
  }
}else{
  dydx = NULL
}

return(
    list(
          "Ws"    = diag(Ws1),
          "deriv" = dydx,
          "Paths" = t(Paths)
         )
       )

## ----------------------------------
## End of the single network function
}


## ------------------------------------------
## Preallocation for the ensemble of networks
Ws <- matrix(0,p,1)
Paths <- matrix(0,settings$max.iter,p)
if(graph.model){
  deriv <- array(0,c(n,p,p))
}

## ----------------------------------
## Construct the ensemble of networks
for(i in 1:ensemble.size){
  
  ## ----------------
  ## Display progress 
  print('-------------------------------')
  print(paste0("Network ",i,"/",ensemble.size))
  print('-------------------------------')
  
  ## ---------------------
  ## Call a single network
  model.nn <- singleNN(X,hidden.layers,settings)
  
  ## ------------------
  ## Ensemble averaging
  Ws    <- Ws    + (1/ensemble.size)*model.nn$Ws
  Paths <- Paths + (1/ensemble.size)*model.nn$Paths
  if(graph.model){
    deriv <- deriv + (1/ensemble.size)*model.nn$deriv
  }
  
}

## ---------------------------------------
## Symmetric matrix of partial derivatives
## Only useful for the graph representation.
if(graph.model){
  Q = apply(deriv,c(2,3),function(x)mean(abs(x)))
  Q = Q - diag(diag(Q))
  Q = Q + t(Q)
}

## -------------------
## Sequence of penalty
## Only useful for the graph representation.
if(graph.model){
  theta = seq(0.01,0.99,length.out=10)
}

## -------------------
## Construct the graph
if(graph.model){
  
## -------------
## Preallocation
W = Ws%*%t(Ws)
G <- array(0,c(p,p,length(theta)))

  ## --------------------------------
  ## One graph for each penalty value
  for(index_theta in 1:length(theta)){
    W[ W < theta[index_theta] ]=0
    G[,,index_theta] = Q*W
    G[,,index_theta] = G[,,index_theta]/max(G[,,index_theta])
  }
  
}else{
  
  G = NULL
  
}


## ----------------------
## Output of the function
return(
    list(
          "Ws"    = Ws,
          "Paths" = Paths,
          "Graph" = G
        )
      )

## ------------------------  
## End of the DAMS function  
}

