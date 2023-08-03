%% Bivariate Normal Distribution pdf
% Compute and plot the pdf of a bivariate normal distribution with parameters mu = [0 0] and Sigma = [0.25 0.3; 0.3 1].
% Define the parameters mu and Sigma.
mu = [1 0];
Sigma = [1, 1.732; 1.732, 4];
% Create a grid of evenly spaced points in two-dimensional space.
x1 = -5:0.2:5;
x2 = -5:0.2:5;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
% Evaluate the pdf of the normal distribution at the grid points.
y = mvnpdf(X,mu,Sigma);
y = reshape(y,length(x2),length(x1));

% Plot the pdf values.
figure
surf(x1,x2,y)
caxis([min(y(:))-0.5*range(y(:)),max(y(:))])
axis([-5 5 -5 5 0 0.4])
xlabel('x1')
ylabel('x2')
zlabel('Probability Density')
title('pdf')
% Copyright 2015 The MathWorks, Inc.

% Calculate CDF
p = mvncdf(X,mu,Sigma);
figure
Z = reshape(p,51,51);
surf(X1,X2,Z)
title('cdf')