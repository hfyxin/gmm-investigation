%% 3-Variate Normal Distribution pdf
clear;
mu = [0 0 0];
Sigma = [9 1.732 0; 
         1.732 1 0; 
         0 0 1];

% Create a grid of evenly spaced points in two-dimensional space.
spacing = 0.4;
x1 = -10:spacing:10;
x2 = -5:spacing:5;
x3 = -5:spacing:5;

[X1,X2,X3] = meshgrid(x1,x2,x3);
X = [X1(:) X2(:) X3(:)];

% Evaluate the pdf of the normal distribution at the grid points.
y = mvnpdf(X,mu,Sigma);

% remove points that are below threshold
pdfThres = 0.003;   % arbitrary
xyPlot = [X, y];
xyPlot(xyPlot(:,4)<pdfThres, :) = [];


% Plot the pdf values.
figure
scatter3(xyPlot(:,1), xyPlot(:,2), xyPlot(:,3),25,'filled');
%caxis([min(y(:))-0.5*range(y(:)),max(y(:))])
axis([-5 5 -5 5 -5 5])
xlabel('x1')
ylabel('x2')
zlabel('x3')
title(['pdf > ', num2str(pdfThres)])

% show axis
axis([0 1 0 1 0 1])
hold all
quiver3(0,0,0,0,0,2*max(zlim),'k','LineWidth',3)
quiver3(0,0,0,0,2*max(ylim),0,'k','LineWidth',3)
quiver3(0,0,0,2*max(xlim),0,0,'k','LineWidth',3)
text(0,0,max(zlim),'X3','Color','k')
text(0,max(ylim),0,'X2','Color','k')
text(max(xlim),0,0,'X1','Color','k')
axis equal

