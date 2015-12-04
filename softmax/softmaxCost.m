function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

%K * M
theta_x = theta * data;
theta_x = bsxfun(@minus, theta_x, max(theta_x, [], 1) );
exp_theta_x = exp( theta_x);

% 1*M
cost = log( bsxfun(@rdivide , exp_theta_x, sum( exp_theta_x  ) ) ).* groundTruth;

cost = -1/numCases * sum(cost(:) );

cost = cost + lambda/2 * sum( theta(:) .^ 2 ) ;

%%%% gradient



thetagrad = ( groundTruth - ( bsxfun(@rdivide , exp_theta_x,  sum(exp_theta_x)  ) ) )* data';
thetagrad = -1/numCases * thetagrad + lambda * theta;




% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

