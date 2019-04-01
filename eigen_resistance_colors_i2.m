%% Eigenresistance Matlab Code

%@authors: Thomas Jagielski and Sparsh Bansal
clear

%% Load images
train = zeros(250,600,3,78);
train_red = zeros(250*600,79);
train_green = zeros(250*600,79);
train_blue = zeros(250*600,79);
for k=0:78
    image_train = imread(strcat('./initialized_train/', int2str(k), '.png'));
    train(:,:,:,k+1) = image_train;
    r_reshaped = reshape(train(:,:,1,k+1),[250*600, 1]);
    train_red(:,k+1) = r_reshaped;   
    g_reshaped = reshape(train(:,:,2,k+1),[250*600, 1]);
    train_red(:,k+1) = g_reshaped;   
    b_reshaped = reshape(train(:,:,2,k+1),[250*600, 1]);
    train_red(:,k+1) = b_reshaped;   
end

test = zeros(250,600,3,100);
test_red = zeros(250*600,100);
test_green = zeros(250*600,100);
test_blue = zeros(250*600,100);
for k=0:99
    image_test = imread(strcat('./initialized_test/', int2str(k), '.png'));
    test(:,:,:,k+1) = image_test;
    r_reshaped = reshape(test(:,:,1,k+1),[250*600, 1]);
    test_red(:,k+1) = r_reshaped;   
    g_reshaped = reshape(test(:,:,2,k+1),[250*600, 1]);
    test_red(:,k+1) = g_reshaped;   
    b_reshaped = reshape(test(:,:,2,k+1),[250*600, 1]);
    test_red(:,k+1) = b_reshaped;  
end

%% Initialize system
% Find SVD of the vector representations of the images
[U_red,S,V] = svd(train_red, 'econ');
[U_green,S,V] = svd(train_green, 'econ');
[U_blue,S,V] = svd(train_blue, 'econ');

% Project faces into eigenspace and find a matrix of weights
red_train_weights = U_red' * train_red;
green_train_weights = U_green' * train_green;
blue_train_weights = U_blue' * train_blue;

%% Test New Image
% Find the weights of the eigenvectors for each face as a representation in
% the eigenspace
red_test_weights = U_red' * test_red;
green_test_weights = U_green' * test_green;
blue_test_weights = U_blue' * test_blue;

%% Load Labels
labels = xlsread('values.xlsx');
train_labels = labels(1:79,2);
test_labels = labels(:,3);

%% For loop to run through images
% Initialize a vector of zeros to represent averages
% By preallocating space, we are saving in runtime
accuracy = zeros(red_test_weights(2));
% For loop to compare the distance of each column of weights to the
% training image weights
for num = 1:100
    % Find the index of the minimum distance between two vectors
    [Y,I] = min(vecnorm(red_test_weights(:,num) - red_train_weights));
    
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