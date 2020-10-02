function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%J = (1/(2*m))*sum(((X * theta)-y).^2)
%theta(1)=0

%thetasum = ((theta' * X')-y')*X
%theta = theta - ((alpha * (1/m)) .* thetasum')


%theta(1)=0
%thetaSquared = sum(theta.^2)
%thetaScaled = (lambda/2*m).*theta
%Jreg = (1/(2*m))*sum(((X * theta)-y).^2) + thetaScaled
Jn = (1/(2*m))*sum(((X * theta)-y).^2)
h=(X * theta)
err = h-y
gradPre = ((1/m)*(X'*err))

theta(1)=0

%gradn2=(1/m)*sum((X(2:end) * theta(2:end))*X(2:end)')
thetaSquared = sum(theta.^2)
thetaScaled = (lambda/(2*m))*thetaSquared
J=Jn + thetaScaled
lamthet = (lambda/m)*theta
grad = gradPre + lamthet 
% =========================================================================

grad = grad(:);

end
