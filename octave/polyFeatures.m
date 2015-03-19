function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
m = size(X, 1);
n = size(X, 2); % number of columns in X
X_poly = zeros(m, p*n);

for r = 1:m
  for xc = 1:n % iterate through each column of X
    for c = 1:p
      X_poly(r, ((xc-1)*p+c)) = X(r, xc)^c;
    end
  end
end

end
