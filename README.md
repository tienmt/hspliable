# hspliable

Bayesian inference for the pliable lasso model in the presence of missing response values. 
The pliable lasso model is a sparse model with sparse interaction effects.
We also extend the pliable lasso to the case of GLMs.

This is based on the paper:  
**"Bayesian Pliable Lasso with Horseshoe Prior for Interaction Effects in GLMs with Missing Responses."**

## Installation

Install the package using:

```r
devtools::install_github('tienmt/hspliable')

library(hspliable)

# simulate data
ntest <- 50
n <- 200 
p <- 10 
q <- 2

xx <- matrix(rnorm((n + ntest) * p), (n + ntest), p)
X <- xx[1:n, ]
xtest <- xx[-(1:n), ]
zz <- matrix(rnorm((n + ntest) * q), (n + ntest), q)
Z <- zz[1:n, ]
ztest <- zz[-(1:n), ]

beta_true <- c( 2,-2, 2, 2 , rep(0, p-4))

theta_true <- matrix(0, p, q)
theta_true[1:3, ] <- matrix( c(rep(1,q), 
                               rep(-2,q), 
                               c(1:q) ) , 3, q, byrow = TRUE)
theta0_true = 0.5
beta0_true = -1
yy <- beta0_true + zz %*% rep(theta0_true, q) + 
  rowSums(sapply(1:p, function(j) xx[, j] * (beta_true[j] + zz %*% theta_true[j, ] ))) +
  rnorm(n + ntest)
y <- yy[1:n]
ytest <- yy[-(1:n)]


library(hspliable)

# Try with pliable Horseshoe function
fit_pHS <- pliable_HS(y, X, Z, n_iter = 1000, burn_in = 500)
(be_HS_2 <- colMeans(fit_pHS$beta) )  
beta_true
(theta_HS_2 <- apply(fit_pHS$theta, c(1, 2), mean) )
theta_true


