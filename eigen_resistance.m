%% Eigenresistance Matlab Code

%@authors: Thomas Jagielski and Sparsh Bansal

%% Load images
train = zeros(250,600,3,80);
bw_train = zeros(250,600,80);
for k=0:79
    image_train = imread(strcat('./new_train_init/', int2str(k), '.png'));
    train(:,:,:,k+1) = image_train;
    bw_train(:,:,k+1) = rgb2gray(image_train);
end

test = zeros(250,600,3,40);
bw_test = zeros(250,600,40);
for k=0:39
    image_test = imread(strcat('./new_test_init/', int2str(k), '.png'));
    test(:,:,:,k+1) = image_test;
    bw_test(:,:,k+1) = rgb2gray(image_test);
end

%% Convert images to black and white
%figure()
%imagesc(bw_train(:,:,3))
%colormap 'gray'

%% Initialize system
% Reshape train images to form "vectors"
train_reshape = reshape(bw_train, size(bw_train,1) * size(bw_train,2), size(bw_train,3));

% Find SVD of the vector representations of the images
[U,S,V] = svd(train_reshape, 'econ');

% Project faces into eigenspace and find a matrix of weights
train_weights = U' * train_reshape;

%% Test New Image
% Reshape the test images for matrix calculations
test_reshape = reshape(bw_test, size(bw_test,1) * size(bw_test,2), size(bw_test,3));
% Find the weights of the eigenvectors for each face as a representation in
% the eigenspace
test_weights = U' * test_reshape;

%% Load Labels
labels = xlsread('values_final.xlsx');
train_labels = labels(1:80,2);
test_labels = labels(1:40,3);

%% For loop to run through images
% Initialize a vector of zeros to represent averages
% By preallocating space, we are saving in runtime
accuracy = zeros(40,1);
% For loop to compare the distance of each column of weights to the
% training image weights
for num = 1:40
    % Find the index of the minimum distance between two vectors
    [Y,I] = min(vecnorm(test_weights(:,num) - train_weights));

    if train_labels(I) == test_labels(num)
        accuracy(num) = 1;
    else
        accuracy(num) = 0;
    end
end
percent_correct = mean(accuracy)