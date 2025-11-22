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
```
or install with vignettes "tutorial"
```r
devtools::install_github(
    "tienmt/hspliable", 
    build_vignettes = TRUE,
    build_opts = c("--no-resave-data", "--no-manual")
)
```

</details>

## Linear sparse pliable model example
<details>

<summary>Gaussian Linear sparse pliable model example</summary>

```r
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


```

## Logistic pliable model for sparse interaction effect
<details>

<summary>Logistic sparse pliable model example</summary>

```r
library(hspliable)
ntest <- 500
n <- 200 
p <- 10
q <- 1
xx <- matrix(rnorm((n + ntest) * p), (n + ntest), p)
X <- xx[1:n, ]
xtest <- xx[-(1:n), ]
zz <- matrix(rnorm((n + ntest) * q), (n + ntest), q)
Z <- zz[1:n, ]
ztest <- zz[-(1:n), ]
beta_true <- c( 2,-2, 2, 2 , rep(0, p-4))
theta_true <- matrix(0, p, q)
#theta_true[1:3, ] <- matrix( c(rep(1,q), 
#                               rep(-2,q), 
#                              c(1:q)) , 3, q, byrow = TRUE)
theta0_true = 0.5
beta0_true = 2
my_mu <- beta0_true + zz %*% rep(theta0_true, q) + 
  rowSums(sapply(1:p, function(j) xx[, j] * (beta_true[j] + zz %*% theta_true[j, ] )))
yy <- rbinom(n + ntest, size = 1, prob = boot::inv.logit(my_mu) )
y <- yy[1:n]

fit2_cpp <- pliable_HS_logistic(y, X, as.matrix(Z), eps = 1e-10,
                          n_iter = 10000, burn_in = 3000, clamp_min = 1e-15,verbose = TRUE)
b_pHS_cpp  <- colMeans(fit2_cpp$beta) ; 
( theta_pHS <- apply(fit2_cpp$theta, c(2,3), mean))
mean(fit2_cpp$beta0); 
colMeans(fit2_cpp$theta0)

cat('Mean squared error of beta =', sum( ( b_pHS_cpp - beta_true)^2 ) )

cat('Mean squared error of theta =', sum( ( theta_pHS- theta_true)^2 ))

```
</details>

## Poisson pliable model for sparse interaction effect
<details>

<summary>Poisson sparse pliable model example</summary>

```r

library(hspliable)
set.seed(1)
n <- 100
p <- 10
q <- 2
X <- matrix(rnorm(n * p), n, p)
Z <- matrix(rnorm(n * q), n, q)
beta_true <- c(2, -2, 0, 2, rep(0, p - 4)) / 4
theta_true <- matrix(0, p, q)
theta_true[1:2, ] <- matrix(c(rep(1, q), rep(-2, q), c(1:q)), 2, q, byrow = TRUE) / 4
theta0_true <- rep(0.5, q)
beta0_true <- 2
# linear predictor and counts
eta <- beta0_true + Z %*% theta0_true + rowSums(sapply(1:p, function(j) X[, j] * (beta_true[j] + Z %*% theta_true[j, ])))
y <- rpois(n, lambda = exp(eta))

fit <- pliable_HS_poisson(y, X, Z, n_iter = 5000L, burn_in = 1000L, verbose = TRUE)
colMeans(fit$beta) ; beta_true
apply(fit$theta, c(2, 3), mean)

```


</details>

