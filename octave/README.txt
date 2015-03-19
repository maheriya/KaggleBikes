The timestamp in CSV files is modified as follows:
A*24 - 973055. This results in the timestamp in number of hours starting from 1 for timestamp 1/1/11 12:00 AM


To get the original date back, reverse operation should be done:

(A+973055)/24 and format as "date time".


- The main script to run is 'bikes_rentals.m'. This script currently converts X into 9th degree
  polynomial using polyFeatures.m (x, x^2, x^3, etc. No cross terms between features).

- 'bikes_results.mat' contains a dump of results (all variables) of one experiment.