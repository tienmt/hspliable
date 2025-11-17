#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// Gibbs sampler for Poisson pliable lasso with group horseshoe prior
// Modified from logistic version to handle Poisson counts and missing y
// Missing y are imputed each iteration as Poisson draws from current mean.
// Numerical stability fixes added: stronger clamping of mu/weights, symmetrize
// matrices before inversion, clamp hyperparameters, and added jitter.

using namespace Rcpp;
using namespace arma;

// draw inverse-gamma as in R code: 1 / rgamma(shape, rate)
inline double rinv_gamma(double shape, double rate) {
  double g = R::rgamma(shape, 1.0 / rate);
  return 1.0 / g;
}

// multivariate normal sampler: mean m (arma::vec), cov V (arma::mat)
inline arma::vec rmvnorm_arma(const arma::vec &m, const arma::mat &V) {
  // ensure V is symmetric
  arma::mat Vs = 0.5 * (V + V.t());
  arma::mat U = arma::chol(Vs, "lower");
  arma::vec z = arma::randn<arma::vec>(m.n_elem);
  return m + U * z;
}

// [[Rcpp::export]]
Rcpp::List gibbs_pliable_lasso_poisson_rcpp(
    Rcpp::NumericVector y_in,
    const arma::mat &X,
    const arma::mat &Z,
    int n_iter = 2000,
    int burn_in = 1000,
    double sigma0_sq = 1.0,
    double eps = 1e-6,
    double clamp_min = 1e-6,
    double clamp_max = 1e5,
    bool verbose = true
) {
  // check dimensions
  int n = X.n_rows;
  int p = X.n_cols;
  int q = Z.n_cols;
  if ((int) y_in.size() != n) stop("Length of y must equal nrow(X)");
  if ((int) Z.n_rows != n) stop("Z must have the same number of rows as X");
  
  // identify missing indices in y
  std::vector<int> miss_idx;
  arma::vec y = arma::zeros<arma::vec>(n);
  for (int i = 0; i < n; ++i) {
    if (Rcpp::NumericVector::is_na(y_in[i])) {
      miss_idx.push_back(i);
      y(i) = 0.0; // placeholder; will be sampled
    } else {
      y(i) = y_in[i];
    }
  }
  
  auto clamp = [&](double x)->double {
    if (!std::isfinite(x)) return clamp_min;
    if (x < clamp_min) return clamp_min;
    if (x > clamp_max) return clamp_max;
    return x;
  };
  
  int d = 1 + q;
  double beta0 = 0.0;
  arma::vec theta0 = arma::zeros<arma::vec>(q);
  arma::vec beta = arma::ones<arma::vec>(p) * 0.1;
  arma::mat theta = arma::ones<arma::mat>(p, q) * 0.1;
  
  arma::vec lambda2 = arma::ones<arma::vec>(p);
  arma::vec nu = arma::ones<arma::vec>(p);
  double tau2 = 1.0;
  double xi = 1.0;
  
  int keep = n_iter - burn_in;
  if (keep < 1) stop("Not enough iterations to save after burnin.");
  
  arma::vec beta0_store = arma::vec(keep);
  arma::mat theta0_store = arma::mat(keep, q);
  arma::mat beta_store = arma::mat(keep, p);
  Rcpp::NumericVector theta_store(Rcpp::Dimension(keep, p, q));
  arma::vec tau2_store = arma::vec(keep);
  arma::mat lambda2_store = arma::mat(keep, p);
  
  // Precompute W0 = [1 | Z]
  arma::mat W0(n, d);
  W0.col(0).ones();
  if (q > 0) W0.cols(1, d-1) = Z;
  
  // Prepare Wj_list (n x d per j)
  std::vector<arma::mat> Wj_list(p);
  for (int j = 0; j < p; ++j) {
    arma::vec xj = X.col(j);
    arma::mat Zj = Z;
    for (int i = 0; i < n; ++i) Zj.row(i) *= xj(i);
    arma::mat Wj(n, d);
    Wj.col(0) = xj;
    if (q > 0) Wj.cols(1, d-1) = Zj;
    Wj_list[j] = std::move(Wj);
  }
  
  // prior precision for intercept block
  arma::mat prior_prec0 = arma::eye<arma::mat>(d, d) * (1.0 / sigma0_sq);
  
  arma::vec eta = arma::zeros<arma::vec>(n);
  arma::vec eta_all = arma::zeros<arma::vec>(n);
  
  for (int iter = 1; iter <= n_iter; ++iter) {
    // compute current linear predictor
    eta_all.zeros();
    for (int j = 0; j < p; ++j) {
      arma::vec xj = X.col(j);
      arma::vec Ztheta_j = Z * theta.row(j).t();
      eta_all += xj * beta(j) + xj % Ztheta_j;
    }
    arma::vec base_lin = arma::ones<arma::vec>(n) * beta0 + Z * theta0;
    eta = base_lin + eta_all;
    
    // Impute missing y: y_i ~ Poisson(exp(eta_i))
    if (!miss_idx.empty()) {
      for (int idx = 0; idx < (int) miss_idx.size(); ++idx) {
        int i = miss_idx[idx];
        
        // 1. Clamp eta *before* exponentiating
        double eta_i = eta(i);
        // exp(15) is ~3.2e6, exp(20) is ~4.8e8. 
        // Let's use a reasonable bound like 15.0 to prevent mu_i from exploding.
        if (eta_i > 15.0) eta_i = 15.0;
        if (eta_i < -15.0) eta_i = -15.0;
        
        // 2. Now calculate mu_i
        double mu_i = std::exp(eta_i);
        
        // 3. Your original clamps are still good safety nets
        mu_i = clamp(mu_i); // This will apply your (hopefully smaller) clamp_max
        if (!std::isfinite(mu_i)) mu_i = clamp_max;
        
        double draw = R::rpois(mu_i);
        
        y(i) = draw;
        //y_in[i] = draw;
      }
    }
    
    // IRLS / Gaussian approximation for Poisson:
    // mu = exp(eta)
    arma::vec mu = arma::exp(eta);
    for (int i = 0; i < n; ++i) mu(i) = clamp(mu(i));
    
    // weights and pseudo-response; ensure no division by zero
    arma::vec w = mu;
    for (int i = 0; i < n; ++i) {
      if (!std::isfinite(w(i)) || w(i) <= 0) w(i) = clamp_min;
      if (w(i) > clamp_max) w(i) = clamp_max;
    }
    arma::vec z = arma::zeros<arma::vec>(n);
    for (int i = 0; i < n; ++i) {
      double mu_i = mu(i);
      double yi = y(i);
      double denom = (mu_i > 0) ? mu_i : clamp_min;
      z(i) = eta(i) + (yi - mu_i) / denom;
      if (!std::isfinite(z(i))) z(i) = eta(i);
    }
    
    // 2) update intercept block (beta0, theta0) using weighted least squares
    arma::mat W0_w = W0;
    W0_w.each_col() %= w;
    arma::mat XtWX0 = W0.t() * W0_w;
    arma::mat P0 = XtWX0 + prior_prec0;
    // symmetrize and add jitter
    P0 = 0.5 * (P0 + P0.t()) + arma::eye<arma::mat>(d,d) * eps;
    arma::mat V0;
    bool ok0 = true;
    try {
      V0 = arma::inv_sympd(P0);
    } catch(...) {
      // fallback: add larger jitter
      P0 += arma::eye<arma::mat>(d,d) * (10.0 * eps);
      V0 = arma::inv_sympd(P0);
      ok0 = false;
    }
    
    arma::vec r0 = w % (z - eta_all);
    arma::vec m0 = V0 * (W0.t() * r0);
    arma::vec par0 = rmvnorm_arma(m0, V0);
    beta0 = par0(0);
    if (q > 0) for (int t = 0; t < q; ++t) theta0(t) = par0(t+1);
    
    // update eta accordingly
    eta = arma::ones<arma::vec>(n) * beta0 + Z * theta0 + eta_all;
    
    // 3) update blocks j = 1..p (weighted Gaussian update)
    for (int j = 0; j < p; ++j) {
      arma::mat Wj = Wj_list[j];
      arma::vec contrib_j = X.col(j) * beta(j) + Wj.cols(1, d-1) * theta.row(j).t();
      arma::vec eta_minus_j = eta - contrib_j;
      
      arma::vec r_j = w % (z - eta_minus_j);
      
      arma::mat Wj_w = Wj;
      Wj_w.each_col() %= w;
      arma::mat XtWXj = Wj.t() * Wj_w;
      
      arma::mat prior_prec_j = arma::eye<arma::mat>(d, d) * (1.0 / (tau2 * lambda2(j)));
      arma::mat Pj = XtWXj + prior_prec_j + arma::eye<arma::mat>(d,d) * eps;
      Pj = 0.5 * (Pj + Pj.t()) + arma::eye<arma::mat>(d,d) * eps;
      
      arma::mat Vj;
      try {
        Vj = arma::inv_sympd(Pj);
      } catch(...) {
        Pj += arma::eye<arma::mat>(d,d) * (10.0 * eps);
        Vj = arma::inv_sympd(Pj);
      }
      
      arma::vec mj = Vj * (Wj.t() * r_j);
      
      arma::vec gamma_j = rmvnorm_arma(mj, Vj);
      beta(j) = gamma_j(0);
      if (q > 0) for (int t = 0; t < q; ++t) theta(j, t) = gamma_j(t+1);
      
      arma::vec new_contrib_j = X.col(j) * beta(j) + Wj.cols(1, d-1) * theta.row(j).t();
      eta = eta_minus_j + new_contrib_j;
      
      double g2 = beta(j) * beta(j) + arma::accu(arma::square(theta.row(j)));
      double rate_lam = (1.0 / nu(j)) + g2 / (2.0 * tau2);
      lambda2(j) = rinv_gamma( (d + 1.0) / 2.0, rate_lam );
      lambda2(j) = clamp(lambda2(j));
      // clamp lambda2 away from zero
      if (lambda2(j) < clamp_min) lambda2(j) = clamp_min;
      if (lambda2(j) > clamp_max) lambda2(j) = clamp_max;
      
      nu(j) = rinv_gamma(0.5, 1.0 + 1.0 / lambda2(j));
      nu(j) = clamp(nu(j));
      if (nu(j) < clamp_min) nu(j) = clamp_min;
    }
    
    // 5) global scale tau2 and xi
    double sum_g2_over_lam = 0.0;
    for (int j = 0; j < p; ++j) {
      double g2j = beta(j) * beta(j) + arma::accu(arma::square(theta.row(j)));
      sum_g2_over_lam += g2j / lambda2(j);
    }
    double shape_tau = (p * d + 1.0) / 2.0;
    double rate_tau = (1.0 / xi) + 0.5 * sum_g2_over_lam;
    tau2 = rinv_gamma(shape_tau, rate_tau);
    tau2 = clamp(tau2);
    if (tau2 < clamp_min) tau2 = clamp_min;
    if (tau2 > clamp_max) tau2 = clamp_max;
    xi = rinv_gamma(0.5, 1.0 + 1.0 / tau2);
    xi = clamp(xi);
    
    // 6) store
    if (iter > burn_in) {
      int k = iter - burn_in - 1;
      beta0_store(k) = beta0;
      for (int t = 0; t < q; ++t) theta0_store(k, t) = theta0(t);
      for (int j = 0; j < p; ++j) beta_store(k, j) = beta(j);
      for (int j = 0; j < p; ++j) for (int t = 0; t < q; ++t) theta_store[k + keep * j + keep * p * t] = theta(j, t);
      tau2_store(k) = tau2;
      for (int j = 0; j < p; ++j) lambda2_store(k, j) = lambda2(j);
    }
    
    if (verbose && (iter % 500 == 0)) {
      double minlam = arma::min(lambda2);
      Rcout << "iter " << iter << ": min(lambda2)=" << minlam << ", tau2=" << tau2 << "\n";
    }
  }
  
  Rcpp::IntegerVector dims = Rcpp::IntegerVector::create(keep, p, q);
  theta_store.attr("dim") = dims;
  
  return Rcpp::List::create(
    Rcpp::Named("beta0") = beta0_store,
    Rcpp::Named("theta0") = theta0_store,
    Rcpp::Named("beta") = beta_store,
    Rcpp::Named("theta") = theta_store,
    Rcpp::Named("tau2") = tau2_store,
    Rcpp::Named("lambda2") = lambda2_store,
    Rcpp::Named("config") = Rcpp::List::create(
      Rcpp::Named("n_iter") = n_iter,
      Rcpp::Named("burn_in") = burn_in,
      Rcpp::Named("sigma0_sq") = sigma0_sq,
      Rcpp::Named("eps") = eps,
      Rcpp::Named("clamp_min") = clamp_min,
      Rcpp::Named("clamp_max") = clamp_max
    )
  );
}
