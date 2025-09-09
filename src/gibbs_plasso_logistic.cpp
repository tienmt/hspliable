#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// Gibbs sampler for logistic pliable lasso with group horseshoe prior
// Rewrites the original R implementation into Rcpp + Armadillo.
// NOTE: This implementation calls BayesLogit::rpg from R via Rcpp. Make sure
// the BayesLogit package is installed in R before calling this function.

using namespace Rcpp;
using namespace arma;

// draw inverse-gamma as in R code: 1 / rgamma(shape, rate)
inline double rinv_gamma(double shape, double rate) {
  // R::rgamma uses scale parameter. scale = 1/rate
  double g = R::rgamma(shape, 1.0 / rate);
  return 1.0 / g;
}

// multivariate normal sampler: mean m (arma::vec), cov V (arma::mat)
inline arma::vec rmvnorm_arma(const arma::vec &m, const arma::mat &V) {
  arma::mat U = arma::chol(V, "lower"); // upper triangular so V = U.t()*U
  arma::vec z = arma::randn<arma::vec>(m.n_elem);
  return m + U * z;
}

// [[Rcpp::export]]
Rcpp::List gibbs_pliable_lasso_logistic2_rcpp(
                            const arma::vec &y,
                            const arma::mat &X,
                            const arma::mat &Z,
                            int n_iter = 2000,
                            int burn_in = 1000,
                            double sigma0_sq = 1.0,
                            double eps = 1e-6,
                            double clamp_min = 1e-10,
                            double clamp_max = 1e10,
                            bool verbose = true
) {
  // check dimensions
  int n = X.n_rows;
  int p = X.n_cols;
  int q = Z.n_cols;
  if ((int) y.n_elem != n) stop("Length of y must equal nrow(X)");
  if ((int) Z.n_rows != n) stop("Z must have the same number of rows as X");
  
  auto clamp = [&](double x)->double {
    if (x < clamp_min) return clamp_min;
    if (x > clamp_max) return clamp_max;
    return x;
  };
  
  int d = 1 + q;
  double beta0 = 0.0;
  arma::vec theta0 = arma::zeros<arma::vec>(q);
  arma::vec beta = arma::ones<arma::vec>(p) * 0.1;
  arma::mat theta = arma::ones<arma::mat>(p, q) * 0.1; // p x q (rows correspond to j)
  
  arma::vec lambda2 = arma::ones<arma::vec>(p);
  arma::vec nu = arma::ones<arma::vec>(p);
  double tau2 = 1.0;
  double xi = 1.0;
  
  int keep = n_iter - burn_in;
  if (keep < 1) stop("Not enough iterations to save after burnin.");
  
  arma::vec beta0_store = arma::vec(keep);
  arma::mat theta0_store = arma::mat(keep, q);
  arma::mat beta_store = arma::mat(keep, p);
  // theta_store as a 3D array stored in column-major order compatible with R's array
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
  
  // call BayesLogit::rpg via Rcpp
  Rcpp::Function rpg("rpg", Rcpp::Environment::namespace_env("BayesLogit"));
  
  // prior precision for intercept block
  arma::mat prior_prec0 = arma::eye<arma::mat>(d, d) * (1.0 / sigma0_sq);
  
  // Main Gibbs loop
  arma::vec eta = arma::zeros<arma::vec>(n);
  arma::vec eta_all = arma::zeros<arma::vec>(n);
  
  for (int iter = 1; iter <= n_iter; ++iter) {
    // compute current linear predictor
    eta_all.zeros();
    for (int j = 0; j < p; ++j) {
      // X[,j] * beta[j] + X[,j] * (Z %*% theta[j,])
      arma::vec xj = X.col(j);
      arma::vec Ztheta_j = Z * theta.row(j).t(); // n x 1
      eta_all += xj * beta(j) + xj % Ztheta_j;
    }
    arma::vec base_lin = arma::ones<arma::vec>(n) * beta0 + Z * theta0;
    eta = base_lin + eta_all;
    
    // 1) Polya-Gamma draws via BayesLogit::rpg
    NumericVector eta_r = wrap(eta);
    NumericVector rpg_out = rpg(n, _["h"] = 1.0, _["z"] = eta_r);
    arma::vec omega = as<arma::vec>(rpg_out);
    for (int i = 0; i < (int) omega.n_elem; ++i) if (omega(i) <= 0) omega(i) = 0.0;
    arma::vec kappa = y - 0.5;
    
    // 2) update intercept block (beta0, theta0)
    // XtWX0 = t(W0) * diag(omega) * W0  => use W0.each_col() % omega then crossprod
    arma::mat W0_omega = W0;
    W0_omega.each_col() %= omega; // elementwise multiply each col by omega
    arma::mat XtWX0 = W0.t() * W0_omega;
    arma::mat P0 = XtWX0 + prior_prec0;
    // add small ridge for numerical stability
    P0 += arma::eye<arma::mat>(d, d) * eps;
    arma::mat V0 = arma::inv_sympd(P0);
    arma::vec m0 = V0 * (W0.t() * (kappa - omega % eta_all));
    arma::vec par0 = rmvnorm_arma(m0, V0);
    beta0 = par0(0);
    if (q > 0) {
      for (int t = 0; t < q; ++t) theta0(t) = par0(t+1);
    }
    
    // 3) update blocks j = 1..p
    for (int j = 0; j < p; ++j) {
      arma::mat Wj = Wj_list[j];
      arma::vec contrib_j = X.col(j) * beta(j) + Wj.cols(1, d-1) * theta.row(j).t();
      arma::vec eta_minus_j = eta - contrib_j;
      arma::vec r_j = kappa - omega % eta_minus_j;
      
      arma::mat Wj_omega = Wj;
      Wj_omega.each_col() %= omega;
      arma::mat XtWXj = Wj.t() * Wj_omega;
      
      arma::mat prior_prec_j = arma::eye<arma::mat>(d, d) * (1.0 / (tau2 * lambda2(j)));
      arma::mat Pj = XtWXj + prior_prec_j + arma::eye<arma::mat>(d,d) * eps;
      arma::mat Vj = arma::inv_sympd(Pj);
      arma::vec mj = Vj * (Wj.t() * r_j);
      
      arma::vec gamma_j = rmvnorm_arma(mj, Vj);
      beta(j) = gamma_j(0);
      if (q > 0) {
        for (int t = 0; t < q; ++t) theta(j, t) = gamma_j(t+1);
      }
      
      arma::vec new_contrib_j = X.col(j) * beta(j) + Wj.cols(1, d-1) * theta.row(j).t();
      eta = eta_minus_j + new_contrib_j;
      
      double g2 = beta(j) * beta(j) + arma::accu(arma::square(theta.row(j)));
      double rate_lam = (1.0 / nu(j)) + g2 / (2.0 * tau2);
      lambda2(j) = rinv_gamma( (d + 1.0) / 2.0, rate_lam );
      lambda2(j) = clamp(lambda2(j));
      nu(j) = rinv_gamma(0.5, 1.0 + 1.0 / lambda2(j));
      nu(j) = clamp(nu(j));
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
    xi = rinv_gamma(0.5, 1.0 + 1.0 / tau2);
    xi = clamp(xi);
    
    // 6) store
    if (iter > burn_in) {
      int k = iter - burn_in - 1; // 0-based index for storage
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
  
  // prepare return list: convert theta_store to an R array with dims c(keep,p,q)
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
