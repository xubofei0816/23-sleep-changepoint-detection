data {
  int T;
  vector[T] y;
}
parameters {
  vector[2] alpha;
  vector<lower = 0>[2] sigma;
  real<lower = 0, upper = 1> xi1_init; 
  real y_tm1_init;
}
transformed parameters {
  matrix[T, 2] eta;
  matrix[T, 2] xi;
  vector[T] f;
  
  // fill in etas
  for(t in 1:T) {
    eta[t,1] = exp(normal_lpdf(y[t]| alpha[1], sigma[1]));
    if(t==1) {
      eta[t,2] = exp(cauchy_lpdf(y[t]| alpha[2], sigma[2]));
    } else {
      eta[t,2] = exp(cauchy_lpdf(y[t]| alpha[2] + y[t-1], sigma[2]));
    }
  }
  
  // work out likelihood contributions
  
  for(t in 1:T) {
    // for the first observation
    if(t==1) {
      f[t] = 0.999*xi1_init*eta[t,1] + // stay in state 1
             (1 - 0.999)*xi1_init*eta[t,2] + // transition from 1 to 2
             0.999*(1 - xi1_init)*eta[t,2] + // stay in state 2 
             (1 - 0.999)*(1 - xi1_init)*eta[t,1]; // transition from 2 to 1
      
      xi[t,1] = (0.999*xi1_init*eta[t,1] +(1 - 0.999)*(1 - xi1_init)*eta[t,1])/f[t];
      xi[t,2] = 1.0 - xi[t,1];
    
    } else {
    // and for the rest
      
      f[t] = 0.999*xi[t-1,1]*eta[t,1] + // stay in state 1
             (1 - 0.999)*xi[t-1,1]*eta[t,2] + // transition from 1 to 2
             0.999*xi[t-1,2]*eta[t,2] + // stay in state 2 
             (1 - 0.999)*xi[t-1,2]*eta[t,1]; // transition from 2 to 1
      
      // work out xi
      
      xi[t,1] = (0.999*xi[t-1,1]*eta[t,1] +(1 - 0.999)*xi[t-1,2]*eta[t,1])/f[t];
      
      // there are only two states so the probability of the other state is 1 - prob of the first
      xi[t,2] = 1.0 - xi[t,1];
    }
  }
  
}
model {
  // priors
  alpha ~ normal(0, .1);
  sigma ~ cauchy(0, 1);
  xi1_init ~ beta(2, 2);
  y_tm1_init ~ normal(0, .1);
  
  
  // likelihood is really easy here!
  target += sum(log(f));
}