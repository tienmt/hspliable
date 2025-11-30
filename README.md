
# hspliable

Bayesian inference for the **pliable lasso** model in the presence of missing response values.
The pliable lasso is a *sparse regression model with sparse interaction effects*, and this package extends it to **generalized linear models (GLMs)** with a fully Bayesian formulation using **Horseshoe priors**.

This package accompanies the paper:

**“Bayesian Pliable Lasso with Horseshoe Prior for Interaction Effects in GLMs with Missing Responses.”**

---

## Installation

Install the package using:

```r
devtools::install_github("tienmt/hspliable")
```

To install with the tutorial vignettes:

```r
devtools::install_github(
    "tienmt/hspliable",
    build_vignettes = TRUE,
    build_opts = c("--no-resave-data", "--no-manual")
)
```

---

# Examples

### Linear Sparse Pliable Model (Gaussian)

<details>
<summary>Click to expand</summary>

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

beta_true <- c(2, -2, 2, 2, rep(0, p - 4))
theta_true <- matrix(0, p, q)
theta_true[1:3, ] <- matrix(c(rep(1, q), 
                              rep(-2, q), 
                              c(1:q)), 3, q, byrow = TRUE)

theta0_true <- 0.5
beta0_true <- -1

yy <- beta0_true + zz %*% rep(theta0_true, q) + 
  rowSums(sapply(1:p, function(j) xx[, j] * (beta_true[j] + zz %*% theta_true[j, ]))) +
  rnorm(n + ntest)

y <- yy[1:n]
ytest <- yy[-(1:n)]

# Fit pliable Horseshoe model
fit_pHS <- pliable_HS(y, X, Z, n_iter = 1000, burn_in = 500)

colMeans(fit_pHS$beta)
beta_true

apply(fit_pHS$theta, c(1, 2), mean)
theta_true
```

</details>

---

### Logistic Sparse Pliable Model

<details>
<summary>Click to expand</summary>

```r
library(hspliable)

ntest <- 500
n <- 200 
p <- 10
q <- 1

xx <- matrix(rnorm((n + ntest) * p), (n + ntest), p)
X <- xx[1:n, ]
zz <- matrix(rnorm((n + ntest) * q), (n + ntest), q)
Z <- zz[1:n, ]

beta_true <- c(2, -2, 2, 2, rep(0, p - 4))
theta_true <- matrix(0, p, q)

theta0_true <- 0.5
beta0_true <- 2

mu <- beta0_true + zz %*% rep(theta0_true, q) +
  rowSums(sapply(1:p, function(j) xx[, j] * (beta_true[j] + zz %*% theta_true[j, ])))

yy <- rbinom(n + ntest, size = 1, prob = boot::inv.logit(mu))
y <- yy[1:n]

fit <- pliable_HS_logistic(
  y, X, Z,
  eps = 1e-10,
  n_iter = 10000, burn_in = 3000,
  clamp_min = 1e-15,
  verbose = TRUE
)

b_est <- colMeans(fit$beta)
theta_est <- apply(fit$theta, c(2, 3), mean)

mean(fit$beta0)
colMeans(fit$theta0)

cat("MSE beta =", sum((b_est - beta_true)^2), "\n")
cat("MSE theta =", sum((theta_est - theta_true)^2), "\n")
```

</details>

---

### Poisson Sparse Pliable Model

<details>
<summary>Click to expand</summary>

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

eta <- beta0_true + Z %*% theta0_true +
  rowSums(sapply(1:p, function(j) X[, j] * (beta_true[j] + Z %*% theta_true[j, ])))

y <- rpois(n, lambda = exp(eta))

fit <- pliable_HS_poisson(
  y, X, Z,
  n_iter = 5000L, burn_in = 1000L,
  verbose = TRUE
)

colMeans(fit$beta)
beta_true
```

</details>

---

### Gamma Sparse Pliable Model

<details>
<summary>Click to expand</summary>

```r
library(hspliable)
ntest <- 500
n <- 100
p <- 110
q <- 2
xx <- matrix(rnorm((n + ntest) * p), (n + ntest), p)
X <- xx[1:n, ]
xtest <- xx[-(1:n), ]
zz <- matrix(rnorm((n + ntest) * q), (n + ntest), q)
Z <- zz[1:n, ]
ztest <- zz[-(1:n), ]
beta_true <- c( .5,-2, 2, .5 , rep(0, p-4))
theta_true <- matrix(0, p, q)
theta_true[1:3, ] <- matrix( c(rep(1,q),
                               rep(-2,q),
                               c(1:q)) , 3, q, byrow = TRUE)
theta0_true = 0.5
beta0_true = 2
my_mu <- beta0_true + zz %*% rep(theta0_true, q) +
  rowSums(sapply(1:p, function(j) xx[, j] * (beta_true[j] + zz %*% theta_true[j, ] )))
mu_true <- exp(my_mu)

k_true <- 2.0
yy <- rgamma(n, shape = k_true, scale = mu_true / k_true)
y <- yy[1:n]

out_gibbs_HS <- pliable_HS_gamma_reg (y, X, Z ,
                                              niter = 8000, burnin = 1000, thin = 2,
                                              b0 = rep(0, 1 + q), V0 = diag(10, 1 + q),
                                              a_tau = 1, b_tau = 0.01,
                                              verbose = T, seed = 123)
round( colMeans(out_gibbs_HS$beta)[1:5], 3)
summary( out_gibbs_HS$k_hat )
sum( (  colMeans(out_gibbs_HS$beta) - beta_true )^2)
round( apply(out_gibbs_HS$theta, c(2,3), mean )[1:5,], 3)
theta_true[1:3,]

# Fit with missing data
y_na = y
y_na[ sample(1:n, n*0.3) ] <- NA
library(tictoc); tic()
out_gibbs_HS <- pliable_HS_gamma_reg(y_na, X, Z ,
                                              niter = 5000, burnin = 1000, thin = 2,
                                              b0 = rep(0, 1 + q), V0 = diag(10, 1 + q),
                                              a_tau = 1, b_tau = 0.01,
                                              verbose = T, seed = 123)  ; toc()
round( colMeans(out_gibbs_HS$beta)[1:5], 3)
summary( out_gibbs_HS$k_hat )
sum( (  colMeans(out_gibbs_HS$beta) - beta_true )^2)
round( apply(out_gibbs_HS$theta, c(2,3), mean )[1:5,], 3)
theta_true[1:3,]



