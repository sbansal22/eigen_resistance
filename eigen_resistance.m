%% Eigenresistance Matlab Code

%@authors: Thomas Jagielski and Sparsh Bansal

%% Load images
train = zeros(250,600,3,78);
bw_train = zeros(250,600,78);
for k=0:78
    image_train = imread(strcat('./initialized_train/', int2str(k), '.png'));
    train(:,:,:,k+1) = image_train;
    bw_train(:,:,k+1) = rgb2gray(image_train);
end

test = zeros(250,600,3,100);
bw_test = zeros(250,600,100);
for k=0:99
    image_test = imread(strcat('./initialized_test/', int2str(k), '.png'));
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
%train_reshape = train_reshape - mean(train_reshape,2);

% Find SVD of the vector representations of the images
[U,S,V] = svd(train_reshape, 'econ');

% Project faces into eigenspace and find a matrix of weights
train_weights = U' * train_reshape;

% Find the average for each person
% Every person has two images next to each other in the training set
% Isolate the odd columns
%odd = train_weights(:,1:2:end);
% Isolate the even columns
%even = train_weights(:,2:2:end);
% Average the odd and even columns to find an average weight vector for
% each person
%avg_weight_face = (odd + even)/2;

%% Test New Image
% Reshape the test images for matrix calculations
test_reshape = reshape(bw_test, size(bw_test,1) * size(bw_test,2), size(bw_test,3));
% Find the weights of the eigenvectors for each face as a representation in
% the eigenspace
test_weights = U' * test_reshape;

%% Load Labels
labels = xlsread('values.xlsx');
train_labels = labels(1:80,2);
test_labels = labels(:,3);

%% For loop to run through images
% Initialize a vector of zeros to represent averages
% By preallocating space, we are saving in runtime
accuracy = zeros(test_weights(2));
% For loop to compare the distance of each column of weights to the
% training image weights
for num = 1:100
    % Find the index of the minimum distance between two vectors
    [Y,I] = min(vecnorm(test_weights(:,num) - train_weights));
    
    if train_labels(I) == test_labels(num)
        accuracy(num) = 1;
    else
        accuracy(num) = 0;
    end
    %figure()
    %subplot(2,1,1)
    %imagesc(bw_test(:,:,num))
    %colormap 'gray'
    %subplot(2,1,2)
    %imagesc(bw_train(:,:,I))
    %colormap 'gray'  
end
percent_correct = mean(accuracy)