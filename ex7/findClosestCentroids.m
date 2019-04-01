function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K, number of clusters
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);   % idx = 300 x 1, indicate the cluster of each data

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% X = 300 x 2, 300 samples; K = 3, 3 clusters
##ini_dis = 1000;
##for i = 1 : size(X,1)
##  for j = 1 : K
##    dis = (X(i,:) - centroids(j,:)) * (X(i,:) - centroids(j,:))';
##    if dis < ini_dis
##      idx(i) = j;
##    end
##    ini_dis = dis;
##  end
##end

for i = 1 : size(X,1)   % 300 samples
  dis = zeros(K,1);   % dis = 3 x 1
  for j = 1 : K   % For each 3 clusters compute distance with the smaple
    dis(j) = ( X(i,:) - centroids(j,:) ) * ( X(i,:) - centroids(j,:) )';
  end
  [val, idx(i)] = min(dis);   % val = minimun distance, idx = index of minimum distance
end

% =============================================================

end

