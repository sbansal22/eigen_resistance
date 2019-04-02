%% Eigenresistance Matlab Code

%@authors: Thomas Jagielski and Sparsh Bansal
clear

%% Load images
red_weights = 1;
green_weights = 1;
blue_weights = 1;

test = zeros(250,600,3,100);
test_red = zeros(250*600,100);
test_green = zeros(250*600,100);
test_blue = zeros(250*600,100);
test_set = zeros(250*600*3,100);
for k=0:99
    image_test = imread(strcat('./initialized_test/', int2str(k), '.png'));
    test(:,:,:,k+1) = image_test;
    r_reshaped = reshape(test(:,:,1,k+1),[250*600, 1]);
    test_red(:,k+1) = r_reshaped;   
    g_reshaped = reshape(test(:,:,2,k+1),[250*600, 1]);
    test_green(:,k+1) = g_reshaped;   
    b_reshaped = reshape(test(:,:,2,k+1),[250*600, 1]);
    test_red(:,k+1) = b_reshaped;
    
    test_set(1:(250*600),k+1) = test_red(:,k+1);
    test_set((250*600)+1:2*(250*600),k+1) = test_green(:,k+1);
    test_set(2*(250*600)+1:3*(250*600),k+1) = test_blue(:,k+1);
end

train = zeros(250,600,3,78);
train_red = zeros(250*600,1);
train_green = zeros(250*600,1);
train_blue = zeros(250*600,1);
train_set = zeros(250*600*3,79*length(red_weights)^3);
j=1;
for k=0:78
    image_train = imread(strcat('./initialized_train/', int2str(k), '.png'));
    test(:,:,:,k+1) = image_train;
    r_reshaped = reshape(test(:,:,1,k+1),[250*600, 1]);
    train_red(:,k+1) = red_weights * r_reshaped;   
    g_reshaped = reshape(test(:,:,2,k+1),[250*600, 1]);
    train_green(:,k+1) = green_weights * g_reshaped;   
    b_reshaped = reshape(test(:,:,2,k+1),[250*600, 1]);
    train_blue(:,k+1) = blue_weights * b_reshaped;  
end

%% Initialize system
% Reshape train images to form "vectors"
%train_reshape = reshape(train_set, size(train_set,1) * size(train_set,2), size(train_set,3));

% Find SVD of the vector representations of the images
[U,S,V] = svd(train_set, 'econ');

% Project faces into eigenspace and find a matrix of weights
train_weights = U' * train_set;

%% Test New Image
% Reshape the test images for matrix calculations
%test_reshape = reshape(test_set, size(test_set,1) * size(test_set,2), size(test_set,3));
% Find the weights of the eigenvectors for each face as a representation in
% the eigenspace
test_weights = U' * test_set;

%% Load Labels
labels = xlsread('values.xlsx');
train_labels = labels(1:79,2);
test_labels = labels(:,3);

%% For loop to run through images
% Initialize a vector of zeros to represent averages
% By preallocating space, we are saving in runtime
accuracy = zeros(size(test_weights,2),1);
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
end
percent_correct = mean(accuracy)