%%
%% train.csv:
%% datetime,season,holiday,workingday,weather,temp,atemp,humidity,windspeed,casual,registered,count
%% test.csv:
%% datetime,season,holiday,workingday,weather,temp,atemp,humidity,windspeed
%% datetime is modified from original CSV as follows:
%% A*24, format and save as number.
%% In other words, to reproduce the original timestamp in date and time format,
%% simply divide the number by 24, and format as 'date time' format in Excel


%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
n = 9;  % number of features

% Load Training Data
fprintf('Loading training data...\n');

data = csvread('train.csv'); % training data
X = data(:,1:n); % first n columns are features
y = data(:,n+1); % usage by casual users
%yr = data(:,n+2); % usage by regular users
total = data(:,n+3); % total count 

X = [X(:,1:6) X(:,8:n)]; % Remove atemp column
n = n - 1;
m = size(X, 1);

Xval = csvread('test.csv'); % test data

% Print out some data points
%fprintf('First 10 examples from the dataset X: \n');
%fprintf(' x = [%.0f %.0f %.0f %.0f %.0f %3.0f %3.0f %3.0f %2.0f], y = %.0f \n', ...
%        [X(1:10,1:n) y(1:10,:)]');
%fprintf('First 10 examples from the dataset Xval: \n');
%fprintf(' x = [%.0f %.0f %.0f %.0f %.0f %3.0f %3.0f %3.0f %2.0f] \n', [Xval(1:10,1:n)]');
%
%fprintf('Program paused. Press enter to continue.\n');


fprintf('First 10 examples from the dataset X: \n');
fprintf(' x = [%.0f %.0f %.0f %.0f %.0f %3.0f %3.0f %2.0f], y = %.0f \n', ...
        [X(1:10,1:n) y(1:10,:)]');
fprintf('First 10 examples from the dataset Xval: \n');
fprintf(' x = [%.0f %.0f %.0f %.0f %.0f %3.0f %3.0f %2.0f] \n', [Xval(1:10,1:n)]');

