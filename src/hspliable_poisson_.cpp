
// [[Rcpp::depends(RcppArmadillo)]]
#include <R.h>
#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// -------------------- Helpers --------------------

// Inverse Gamma sampler
inline double rinv_gamma(double shape, double rate) {
  return 1.0 / R::rgamma(shape, 1.0 / rate);
}

// Robust Sampler: Returns x ~ N(P^-1 * b, P^-1)
// Prevents crashes if P is singular
inline arma::vec sample_gaussian_precision(const arma::vec &b, arma::mat P) {
  arma::mat L;
  bool success = false;
  double jit = 1e-8;

  // Attempt 1-6: Increasing jitter
  for(int i=0; i<6; ++i) {
    success = arma::chol(L, P, "lower");
    if(success) break;
    P.diag() += arma::ones<arma::vec>(P.n_rows) * jit;
    jit *= 10.0;
  }

  // Attempt 7: Massive Ridge (Last Resort)
  if (!success) {
    P.diag() += arma::ones<arma::vec>(P.n_rows) * 1e-2;
    success = arma::chol(L, P, "lower");
  }

  // FAILSAFE: If matrix is still dead, return 0 (strong shrinkage) rather than crash
  if (!success) {
    return arma::zeros<arma::vec>(P.n_rows);
  }

  // Standard Cholesky solve
  arma::vec y = arma::solve(trimatl(L), b);
  arma::vec mean = arma::solve(trimatu(L.t()), y);
  arma::vec z = arma::randn<arma::vec>(P.n_rows);
  arma::vec v = arma::solve(trimatu(L.t()), z);

  return mean + v;
}

// Clamp helper
inline double clamp_double(double x, double lo, double hi) {
  if (!std::isfinite(x)) return (lo+hi)/2.0;
  return std::clamp(x, lo, hi);
}

// -------------------- Main sampler --------------------

// [[Rcpp::export]]
Rcpp::List gibbs_pliable_lasso_poisson_rcpp(
    Rcpp::NumericVector y_in,
    const arma::mat &X,
    const arma::mat &Z,
    int n_iter = 2000,
    int burn_in = 1000,
    double sigma0_sq = 10.0,
    bool verbose = true,
    double a0 = 1.0,
    double b0 = 1.0,
    double init_sigma_u2 = 1.0
) {

  // --- Dimensions ---
  int n = X.n_rows;
  int p = X.n_cols;
  int q = Z.n_cols;
  int d = 1 + q;
  if (p > (n/2 ) / (q+1) ){
    if ((int) y_in.size() != n) stop("Length of y must equal nrow(X)");

    // --- Handle missing y ---
    std::vector<int> miss_idx;
    arma::vec y = arma::zeros<arma::vec>(n);
    for (int i = 0; i < n; ++i) {
      if (Rcpp::NumericVector::is_na(y_in[i])) {
        miss_idx.push_back(i);
      } else {
        y(i) = y_in[i];
      }
    }

    // --- Initialization ---
    double beta0 = 0.0;
    arma::vec theta0 = arma::zeros<arma::vec>(q);
    arma::vec beta = arma::zeros<arma::vec>(p);
    arma::mat theta = arma::zeros<arma::mat>(p, q);

    arma::vec lambda2 = arma::ones<arma::vec>(p);
    arma::vec nu = arma::ones<arma::vec>(p);
    double tau2 = 1.0;
    double xi = 1.0;

    // Latent variables
    arma::vec u = arma::zeros<arma::vec>(n);
    double sigma_u2 = init_sigma_u2;

    // Storage
    int keep = n_iter - burn_in;
    arma::vec beta0_store(keep);
    arma::mat theta0_store(keep, q);
    arma::mat beta_store(keep, p);
    Rcpp::NumericVector theta_store(Rcpp::Dimension(keep, p, q));
    arma::vec tau2_store(keep);
    arma::mat lambda2_store(keep, p);
    arma::vec sigma_u2_store(keep);

    // Pre-calc W0
    arma::mat W0(n, d);
    W0.col(0).ones();
    if (q > 0) W0.cols(1, d-1) = Z;

    // Initialize eta
    arma::vec eta = arma::ones<arma::vec>(n) * beta0 + Z * theta0 + u;

    // Reusables
    arma::mat Wj(n, d);
    arma::mat Pj(d, d);

    // --- Gibbs Loop ---
    for (int iter = 1; iter <= n_iter; ++iter) {

      // Allow user to stop calculation
      if (iter % 100 == 0) Rcpp::checkUserInterrupt();

      // 1. Impute Missing
      if (!miss_idx.empty()) {
        for (int ii : miss_idx) {
          double eta_i = clamp_double(eta(ii), -20.0, 20.0);
          y(ii) = R::rpois(std::exp(eta_i));
        }
      }

      // 2. Update Latent u (Newton-Raphson + Gaussian approx)
      double prec_u = 1.0 / sigma_u2;
      for (int i = 0; i < n; ++i) {
        double eta_no_u = eta(i) - u(i);
        double u_new = u(i);

        // Newton steps to find mode of p(u_i | y, ...)
        for (int nit = 0; nit < 5; ++nit) {
          double mu_i = std::exp(clamp_double(eta_no_u + u_new, -20.0, 20.0));
          double g = y(i) - mu_i - u_new * prec_u;
          double H = -mu_i - prec_u;

          double step = g / H;
          if (!std::isfinite(step)) step = 0.0;
          // Limit step size
          step = clamp_double(step, -2.0, 2.0);

          u_new -= step;
        }

        // Gaussian approximation at mode
        double mu_mode = std::exp(clamp_double(eta_no_u + u_new, -20.0, 20.0));
        double var_u = 1.0 / (mu_mode + prec_u);
        double draw = u_new + std::sqrt(var_u) * R::rnorm(0.0, 1.0);

        u(i) = clamp_double(draw, -15.0, 15.0);
        eta(i) = eta_no_u + u(i);
      }

      // 3. Update Sigma_u2
      sigma_u2 = rinv_gamma(a0 + 0.5 * n, b0 + 0.5 * arma::dot(u, u));
      sigma_u2 = clamp_double(sigma_u2, 1e-6, 100.0);

      // 4. Prepare IRLS Parts
      arma::vec mu = arma::exp(arma::clamp(eta, -20.0, 20.0));
      arma::vec w = mu;
      arma::vec z_resp = arma::zeros<arma::vec>(n);

      for(int i=0; i<n; ++i) {
        double mu_val = (mu(i) < 1e-6) ? 1e-6 : mu(i);
        w(i) = mu_val;
        double resid = (y(i) - mu_val) / mu_val;
        resid = clamp_double(resid, -20.0, 20.0);
        z_resp(i) = eta(i) + resid;
      }

      // 5. Update Intercept Block
      arma::vec eta_preds = eta - (W0 * join_vert(arma::vec({beta0}), theta0));
      arma::vec r0 = z_resp - eta_preds;

      arma::mat W0_w = W0;
      W0_w.each_col() %= w;
      arma::mat P0 = W0.t() * W0_w;
      P0.diag() += 1e-6;

      arma::vec XtWr0 = W0_w.t() * (z_resp - eta_preds); // Simplification: W^T * W * r = X' * w * (z - eta)
      arma::vec par0 = sample_gaussian_precision(XtWr0, P0);

      beta0 = par0(0);
      if(q > 0) theta0 = par0.subvec(1, q);

      eta = eta_preds + W0 * par0;

      // 6. Update Predictor Blocks
      for (int j = 0; j < p; ++j) {
        // Build Design Matrix
        arma::vec xj = X.col(j);
        Wj.col(0) = xj;
        for(int k=0; k<q; ++k) Wj.col(k+1) = Z.col(k) % xj;

        // Remove current effect
        arma::vec coef_curr(d);
        coef_curr(0) = beta(j);
        if(q>0) coef_curr.subvec(1, q) = theta.row(j).t();

        arma::vec eta_no_j = eta - Wj * coef_curr;

        // Weighted Least Squares Stats
        arma::mat Wj_w = Wj;
        Wj_w.each_col() %= w;
        Pj = Wj.t() * Wj_w;

        // Prior
        double prior_prec = 1.0 / (tau2 * lambda2(j));
        Pj.diag() += prior_prec + 1e-8;

        arma::vec XtWr = Wj_w.t() * (z_resp - eta_no_j);

        // Sample
        arma::vec new_coefs = sample_gaussian_precision(XtWr, Pj);

        beta(j) = new_coefs(0);
        if(q>0) for(int k=0; k<q; ++k) theta(j, k) = new_coefs(k+1);

        // Update Eta
        eta = eta_no_j + Wj * new_coefs;

        // Update Local Shrinkage
        double g2 = arma::dot(new_coefs, new_coefs);
        double rate_lam = (1.0/nu(j)) + g2/(2.0*tau2);
        lambda2(j) = rinv_gamma((d+1.0)/2.0, rate_lam);
        // Strong clamp prevents singular matrices in next iter
        lambda2(j) = clamp_double(lambda2(j), 1e-10, 100.0);

        nu(j) = rinv_gamma(0.5, 1.0 + 1.0/lambda2(j));
      }

      // 7. Global Shrinkage
      double sum_g2 = 0.0;
      for(int j=0; j<p; ++j) {
        double g2 = beta(j)*beta(j) + arma::accu(arma::square(theta.row(j)));
        sum_g2 += g2 / lambda2(j);
      }
      tau2 = rinv_gamma((p*d + 1.0)/2.0, (1.0/xi) + 0.5*sum_g2);
      tau2 = clamp_double(tau2, 1e-10, 10.0);
      xi = rinv_gamma(0.5, 1.0 + 1.0/tau2);

      // 8. Store
      if (iter > burn_in) {
        int k = iter - burn_in - 1;
        beta0_store(k) = beta0;
        theta0_store.row(k) = theta0.t();
        beta_store.row(k) = beta.t();

        // Correct 3D array flattening for R
        // R stores array[iter, p, q]
        for(int j=0; j<p; ++j) {
          for(int t=0; t<q; ++t) {
            theta_store[k + keep * j + keep * p * t] = theta(j, t);
          }
        }

        tau2_store(k) = tau2;
        lambda2_store.row(k) = lambda2.t();
        sigma_u2_store(k) = sigma_u2;
      }

      // 9. Periodic Re-centering (Drift Correction)
      if (iter % 200 == 0) {
        eta = arma::ones<arma::vec>(n)*beta0 + Z*theta0 + u;
        for(int j=0; j<p; ++j) {
          arma::vec term = X.col(j) * beta(j);
          if(q>0) term += (Z * theta.row(j).t()) % X.col(j);
          eta += term;
        }
        if (verbose) Rcout << "Iter " << iter << " tau2: " << tau2 << "\n";
      }
    }

    // Set dimensions on the flat vector for R
    theta_store.attr("dim") = Rcpp::IntegerVector::create(keep, p, q);

    return Rcpp::List::create(
      Rcpp::Named("beta0") = beta0_store,
      Rcpp::Named("theta0") = theta0_store,
      Rcpp::Named("beta") = beta_store,
      Rcpp::Named("theta") = theta_store,
      Rcpp::Named("tau2") = tau2_store,
      Rcpp::Named("lambda2") = lambda2_store,
      Rcpp::Named("sigma_u2") = sigma_u2_store,
      Rcpp::Named("u_last") = u // useful for diagnostics
    );
  }else{
    if ((int) y_in.size() != n) stop("Length of y must equal nrow(X)");
    if ((int) Z.n_rows != n) stop("Z must have the same number of rows as X");

    // --- Handle Missing Y ---
    std::vector<int> miss_idx;
    arma::vec y = arma::zeros<arma::vec>(n);
    for (int i = 0; i < n; ++i) {
      if (Rcpp::NumericVector::is_na(y_in[i])) {
        miss_idx.push_back(i);
        y(i) = 0.0;
      } else {
        y(i) = y_in[i];
      }
    }

    // --- Initialization ---
    // Helper lambda for clamping values to safe ranges
    auto clamp_val = [](double x) -> double {
      if (!std::isfinite(x)) return 1e-6;
      if (x < 1e-8) return 1e-8;
      if (x > 1e8) return 1e6;
      return x;
    };

    double beta0 = 0.0;
    arma::vec theta0 = arma::zeros<arma::vec>(q);
    arma::vec beta = arma::zeros<arma::vec>(p);    // Initialize at 0 for sparsity
    arma::mat theta = arma::zeros<arma::mat>(p, q);

    arma::vec lambda2 = arma::ones<arma::vec>(p);
    arma::vec nu = arma::ones<arma::vec>(p);
    double tau2 = 0.1;
    double xi = 0.1;

    // Storage
    int keep = n_iter - burn_in;
    if (keep < 1) stop("burn_in must be less than n_iter");

    arma::vec beta0_store(keep);
    arma::mat theta0_store(keep, q);
    arma::mat beta_store(keep, p);
    // Flatten theta storage for memory efficiency in R (List of matrices is heavy)
    // We will return a 3D array concept, but store flat here or use cube if p is moderate.
    // For high-dim, better to store only betas usually, but we will keep structure.
    Rcpp::NumericVector theta_store(Rcpp::Dimension(keep, p, q));
    arma::vec tau2_store(keep);
    arma::mat lambda2_store(keep, p);

    // Pre-calculate W0 (Intercept Block)
    arma::mat W0(n, d);
    W0.col(0).ones();
    if (q > 0) W0.cols(1, d-1) = Z;

    // Initialize Eta
    arma::vec eta = arma::ones<arma::vec>(n) * beta0 + Z * theta0;
    // Note: X*beta is 0 initially

    // Pre-allocate reusable matrices to save alloc time in loop
    arma::mat Wj(n, d);
    arma::mat Pj(d, d);
    arma::vec r_j(n);
    arma::vec XtWr(d);

    // --- Gibbs Loop ---
    for (int iter = 1; iter <= n_iter; ++iter) {

      // 1. Impute Missing Data (if any)
      if (!miss_idx.empty()) {
        for (int i : miss_idx) {
          double eta_i = std::clamp(eta(i), -10.0, 10.0); // clamp for Poisson stability
          y(i) = R::rpois(std::exp(eta_i));
        }
      }

      // 2. Data Augmentation / Approximation (Poisson -> Weighted Least Squares)
      // We use the Taylor expansion approximation (IRLS step)
      // mu = exp(eta), z = eta + (y-mu)/mu, weights = mu
      arma::vec mu = arma::exp(eta);
      arma::vec w = mu;
      arma::vec z_resp = arma::zeros<arma::vec>(n);

      for(int i=0; i<n; ++i) {
        // Cap mu to prevent overflow
        double mu_val = (mu(i) > 1e5) ? 1e5 : ((mu(i) < 1e-4) ? 1e-4 : mu(i));
        w(i) = mu_val;

        // Working response z
        double resid = (y(i) - mu_val) / mu_val;
        // Cap residual to prevent exploding gradients
        if(resid > 20.0) resid = 20.0;
        if(resid < -20.0) resid = -20.0;

        z_resp(i) = eta(i) + resid;
      }

      // 3. Update Intercept Block (Beta0, Theta0)
      // We subtract the effect of X*Beta + W*Theta from z to get residual for intercept
      arma::vec eta_predictors = eta - (W0.col(0)*beta0 + W0.cols(1, d-1)*theta0);
      arma::vec r0 = z_resp - eta_predictors;

      // Weighted X'X and X'y
      // P0 = W0' * diag(w) * W0 + Prior
      arma::mat W0_weighted = W0;
      W0_weighted.each_col() %= w; // Element-wise multiply columns by weights
      arma::mat P0 = W0.t() * W0_weighted;
      P0.diag() += (1.0/sigma0_sq); // Prior precision

      arma::vec XtWr0 = W0_weighted.t() * r0; // Effectively W0' * W * r0

      // Sample
      arma::vec par0 = sample_gaussian_precision(XtWr0, P0);
      beta0 = par0(0);
      if (q > 0) theta0 = par0.subvec(1, q);

      // Update Eta with new intercept
      eta = eta_predictors + (W0 * par0);

      // 4. Update Predictor Blocks j=1...p
      for (int j = 0; j < p; ++j) {

        // --- A. Construct Design Matrix Wj on the fly (Save RAM) ---
        arma::vec xj = X.col(j);
        Wj.col(0) = xj;
        // Interaction terms: Z_k * X_j
        for(int k=0; k<q; ++k) {
          Wj.col(k+1) = Z.col(k) % xj;
        }

        // --- B. Remove current effect of j from eta ---
        arma::vec current_coefs(d);
        current_coefs(0) = beta(j);
        if(q>0) current_coefs.subvec(1, q) = theta.row(j).t();

        arma::vec contrib_j = Wj * current_coefs;
        arma::vec eta_no_j = eta - contrib_j;

        // --- C. Prepare Weighted LS parts ---
        // Residual for block j
        r_j = z_resp - eta_no_j;

        // Weighted Cross Products: X'WX
        // Optimization: (Wj.t() * diag(w) * Wj)
        // We do this manually or using Armadillo optimizations
        // Weighted cross-products and weighted RHS (explicit and stable)
        arma::vec wr = w % r_j;                  // elementwise multiply
        Pj = Wj.t() * (Wj.each_col() % w);      // or Pj = Wj.t() * (Wj % repmat(w, 1, d));
        XtWr = Wj.t() * wr;

        // Add horseshoe prior precision AND a small ridge for numerical stability
        double prior_prec_val = 1.0 / (tau2 * lambda2(j));
        Pj.diag() += prior_prec_val + 1e-8;      // proactively add tiny jitter

        // The "y" in weighted LS is z.
        // W * (z - eta_no_j) is correct for the residual form?
        // Actually simpler: XtWr = Wj.t() * (w % (z_resp - eta_no_j))
        // Which simplifies to Wj_w.t() * (z_resp - eta_no_j)

        // --- D. Sample New Coefficients ---
        arma::vec new_coefs = sample_gaussian_precision(XtWr, Pj);

        beta(j) = new_coefs(0);
        if (q > 0) {
          for(int k=0; k<q; ++k) theta(j, k) = new_coefs(k+1);
        }

        // --- E. Update Eta ---
        eta = eta_no_j + Wj * new_coefs;

        // --- F. Update Local Shrinkage (Lambda) ---
        double g2 = arma::dot(new_coefs, new_coefs); // beta^2 + sum(theta^2)
        double rate_lam = (1.0 / nu(j)) + g2 / (2.0 * tau2);
        lambda2(j) = rinv_gamma((d + 1.0) / 2.0, rate_lam);
        lambda2(j) = clamp_val(lambda2(j));

        nu(j) = rinv_gamma(0.5, 1.0 + 1.0 / lambda2(j));
        nu(j) = clamp_val(nu(j));
      }

      // 5. Global Scale Updates (Tau, Xi)
      double sum_g2_over_lam = 0.0;
      for (int j = 0; j < p; ++j) {
        double g2 = beta(j)*beta(j) + arma::accu(arma::square(theta.row(j)));
        sum_g2_over_lam += g2 / lambda2(j);
      }

      double rate_tau = (1.0 / xi) + 0.5 * sum_g2_over_lam;
      tau2 = rinv_gamma((p * d + 1.0) / 2.0, rate_tau);
      tau2 = clamp_val(tau2);

      xi = rinv_gamma(0.5, 1.0 + 1.0 / tau2);
      xi = clamp_val(xi);

      // 6. Numerical Hygiene: Recompute Eta periodically
      // This prevents floating point drift from accumulating over 50,000 updates
      if (iter % 100 == 0) {
        eta = arma::ones<arma::vec>(n) * beta0 + Z * theta0;
        for(int j=0; j<p; ++j) {
          arma::vec xj = X.col(j);
          arma::vec term = xj * beta(j); // main effect
          // interaction
          if(q > 0) {
            arma::vec Ztheta_j = Z * theta.row(j).t();
            term += xj % Ztheta_j;
          }
          eta += term;
        }
      }

      // 7. Store Samples
      if (iter > burn_in) {
        int k = iter - burn_in - 1;
        beta0_store(k) = beta0;
        theta0_store.row(k) = theta0.t();
        beta_store.row(k) = beta.t();

        // Flatten theta into the Rcpp vector
        for (int j = 0; j < p; ++j) {
          for (int t = 0; t < q; ++t) {
            // Indexing for 3D array: [iter, p, q]
            // R stores column major, so left-most index changes fastest
            // But Rcpp::Dimension helps us map it.
            // Let's just fill linearly based on how R expects arrays
            // dim = (keep, p, q)
            // index = k + keep * j + keep * p * t
            theta_store[k + keep * j + keep * p * t] = theta(j, t);
          }
        }
        tau2_store(k) = tau2;
        lambda2_store.row(k) = lambda2.t();
      }

      if (verbose && (iter % 500 == 0)) {
        Rcout << "Iter " << iter << " | tau2: " << tau2 << std::endl;
      }
    }

    return Rcpp::List::create(
      Rcpp::Named("beta0") = beta0_store,
      Rcpp::Named("theta0") = theta0_store,
      Rcpp::Named("beta") = beta_store,
      Rcpp::Named("theta") = theta_store,
      Rcpp::Named("tau2") = tau2_store,
      Rcpp::Named("lambda2") = lambda2_store
    );
  }
}
