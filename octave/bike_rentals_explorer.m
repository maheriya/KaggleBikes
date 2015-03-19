%% Kaggle Bikes Rental ML challange
%
% X or Xtest:
% datetime,season,holiday,workingday,weather,temp,atemp,humidity,windspeed
% y, yr
% casual,registered
% Original data in train.csv and test.csv. Dumped in bikes_data.mat for
% speed.
%% Initialization
clear all; close all; clc
%warning('off', 'Octave:possible-matlab-short-circuit-operator');

%% =========== Part 1: Loading and Visualizing Data =============
% Load Training Data
% fprintf('Loading training data...\n');
load('bikes_data.mat'); % X, y, yr and Xchallenge will be populated.
total = y + yr; % total count 

%Xorg = X(:,1); % For now, just use timestamp
%Xorg = [X(:,1) X(:,7)]; % For now, just use timestamp and atemp
Xorg = X;
yorg = y;
rndi = randperm(size(X,1));
mtrain = round(size(X,1)*0.6);
mval   = round(size(X,1)*0.2);
mtest  = size(X,1) - mtrain - mval;
X     = Xorg(rndi(1:mtrain), :);
y     = yorg(rndi(1:mtrain));
Xval  = Xorg(rndi(mtrain+1:mtrain+mval), :);
yval  = yorg(rndi(mtrain+1:mtrain+mval), :);
Xtest = Xorg(rndi(mtrain+mval+1:end), :);
ytest = yorg(rndi(mtrain+mval+1:end), :);
%X     = [X(:,1:6) X(:,8:N)]; % Remove atemp column
%Xtest = [Xtest(:,1:6) Xtest(:,8:N)]; % Remove atemp column
m     = size(X, 1);


% Plot training data
plot(X(:,1), y, 'rx', 'MarkerSize', 5, 'LineWidth', 0.7);
xlabel('Time (x)');
ylabel('Casual Rentals (y)');

fprintf('Program paused. Press enter to continue.\n');
pause;

% % %% =========== Part 4: Train Linear Regression =============
% % %  trainLinearReg function will use linearRegCostFunction to train 
% % %  regularized linear regression.
% % % 
% % %  Write Up Note: The data is non-linear, so this will not give a great 
% % %                 fit.
% % %
% % 
% % %  Train linear regression with lambda = 0
% % lambda = 0;
% % [theta] = trainLinearReg([ones(m, 1) X], y, lambda);
% % 
% % %  Plot fit over the data
% % plot(X, y, 'rx', 'MarkerSize', 5, 'LineWidth', 0.7);
% % xlabel('Time (x)');
% % ylabel('Bike Rentals (y)');
% % hold on;
% % plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
% % hold off;
% % 
% % fprintf('Program paused. Press enter to continue.\n');
% % pause;
% % 
% % 
% % %% =========== Part 5: Learning Curve for Linear Regression =============
% % %  Write Up Note: Since the model is underfitting the data, we expect to
% % %                 see a graph with "high bias" -- slide 8 in ML-advice.pdf 
% % %
% % 
% % lambda = 0;
% % [error_train, error_val scount] = ...
% %     learningCurve([ones(m, 1) X], y, ...
% %                   [ones(size(Xval, 1), 1) Xval], yval, ...
% %                   lambda);
% % figure;
% % plot(scount, error_train, scount, error_val);
% % title('Learning curve for linear regression')
% % legend('Train', 'Cross Validation')
% % xlabel('Number of training examples')
% % ylabel('Error')
% % %axis([0 max(scount) 0 max(max([error_train error_val])
% % 
% % fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
% % for i = 1:length(scount)
% %     fprintf('  \t%d\t\t%f\t%f\n', scount(i), error_train(i), error_val(i));
% % end
% % 
% % %fprintf('Program paused. Press enter to continue.\n');
% % %pause;

%% =========== Part 6: Feature Mapping for Polynomial Regression =============
%  One solution to this is to use polynomial regression. You should now
%  complete polyFeatures to map each example into its powers
%

p = 8;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

fprintf('Normalized Training Example 1:\n');
fprintf(' %f', X_poly(1, :));
fprintf('\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;



%% =========== Part 7: Learning Curve for Polynomial Regression =============
%  Now, you will get to experiment with polynomial regression with multiple
%  values of lambda. The code below runs polynomial regression with 
%  lambda = 0. You should try running the code with different values of
%  lambda to see how the fit and learning curve change.
%

lambda = 0.1;
[theta] = trainLinearReg(X_poly, y, lambda);

% % % Plot training data and fit
% % figure(1);
% % plot(X, y, 'rx', 'MarkerSize', 5, 'LineWidth', 0.7);
% % plotFit(min(X(:,1)), max(X(:,1)), mu, sigma, theta, p);
% % xlabel('Time (x)');
% % ylabel('Rentals (y)');
% % title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));


figure(2);
[error_train, error_val scount] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(scount, error_train, scount, error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
%axis([0 13 0 100])
legend('Train', 'Cross Validation')

fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:length(scount)
    fprintf('  \t%d\t\t%f\t%f\n', scount(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 8: Validation for Selecting Lambda =============
%  You will now implement validationCurve to test various values of 
%  lambda on a validation set. You will then use this to select the
%  "best" lambda value.
%

[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;
