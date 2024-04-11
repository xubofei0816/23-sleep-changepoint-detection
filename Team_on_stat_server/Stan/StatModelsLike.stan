data {
  int T;
  vector[T] y;
}
parameters {
  real<lower = 0, upper = 1> xi1_init; 
}
transformed parameters {
  matrix[T, 2] eta;
  matrix[T, 2] xi;
  vector[T] f;
  
  // fill in etas
  for(t in 1:T) {
    if(t==1) {
      eta[t,1] = exp(normal_lpdf(y[t]| 0, 10));
      eta[t,2] = exp(normal_lpdf(y[t]| 0, 10));
    } else {
      eta[t,1] = exp(normal_lpdf(y[t]| -3 + 0.9 * y[t-1], 15));
      eta[t,2] = exp(normal_lpdf(y[t]| y[t-1], 0.01));
    }
  }
  
  // work out likelihood contributions
  
  for(t in 1:T) {
    // for the first observation
    if(t==1) {
      f[t] = 0.99*xi1_init*eta[t,1] + // stay in state 1
             (1 - 0.99)*xi1_init*eta[t,2] + // transition from 1 to 2
             0.99*(1 - xi1_init)*eta[t,2] + // stay in state 2 
             (1 - 0.99)*(1 - xi1_init)*eta[t,1]; // transition from 2 to 1
      
      xi[t,1] = (0.99*xi1_init*eta[t,1] +(1 - 0.99)*(1 - xi1_init)*eta[t,1])/f[t];
      xi[t,2] = 1.0 - xi[t,1];
    
    } else {
    // and for the rest
      
      f[t] = 0.99*xi[t-1,1]*eta[t,1] + // stay in state 1
             (1 - 0.99)*xi[t-1,1]*eta[t,2] + // transition from 1 to 2
             0.99*xi[t-1,2]*eta[t,2] + // stay in state 2 
             (1 - 0.99)*xi[t-1,2]*eta[t,1]; // transition from 2 to 1
      
      // work out xi
      
      xi[t,1] = (0.99*xi[t-1,1]*eta[t,1] +(1 - 0.99)*xi[t-1,2]*eta[t,1])/f[t];
      
      // there are only two states so the probability of the other state is 1 - prob of the first
      xi[t,2] = 1.0 - xi[t,1];
    }
  }
  
}
model {
  // priors
  xi1_init ~ beta(2, 2);  
  
  // likelihood is really easy here!
  target += sum(log(f));
}