function [error_train, error_val scount] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val scount] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. Also returns
%       scount vector that records number of samples used for calculating
%       corresponding train and val errors. error_train and 
%       error_val are of the same size. error_train(i) contains the training 
%       error for scount(i) examples (and similarly for error_val(i)).
%

% Number of training examples
m  = size(X,   1);
mv = size(Xval,1);
intval = 100;
iter   = 10;

% You need to return these values correctly
error_train = zeros(length(100:intval:m), 1);
error_val   = zeros(length(100:intval:m), 1);
scount      = zeros(length(100:intval:m), 1);

% Random selection of samples
iii = 1;
for i = 100:intval:m;
  Et = 0; Ev = 0;
  %fprintf('\n\n=> Learning Curve Iteration %d (i=%d)\n', iii, i);
  for j = 1:iter
    sel = randperm(m);
    sel = sel(1:i);

    theta = trainLinearReg(X(sel, :), y(sel), lambda);
    [Et_, grad] = linearRegCostFunction(X(sel, :), y(sel), theta, 0);
    [Ev_, grad] = linearRegCostFunction(Xval, yval, theta, 0);
    Et = Et + Et_;
    Ev = Ev + Ev_;
    %fprintf('Etrain = %f, Eval = %f\n', Et, Ev);
  end
  error_train(iii) = Et/iter;
  error_val(iii)   = Ev/iter;
  scount(iii)      = i;
  fprintf('========================================================\n');
  fprintf('Average Etrain(%d) = %f, Eval(%d) = %f\n', ...
          iii, error_train(iii), iii, error_val(iii));
  iii = iii + 1;
end

end
