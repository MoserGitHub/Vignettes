
data {
  int N; // number of obs
  int G; // number of groups
  int K; // number of predictors

  int y[N];
  int N_i[N];
  int g[N]; // map obs to groups
  row_vector[K] X[N];
}

parameters {
  real a_g[G];
  real intercept; // improper uniform prior (-inf, inf) 
  vector[K] beta; // improper uniform prior (-inf, inf) 
  real <lower=0> sigma; // improper uniform prior (-inf, inf) 
}

model {
    vector[N] p;
    for (i in 1:N) {
        p[i] = inv_logit(intercept+X[i]*beta+a_g[g[i]]);
    }
    y ~ binomial(N_i, p);
    a_g ~ normal(0, sigma);
    intercept ~ normal(0, 1000); // same as INLA
}

