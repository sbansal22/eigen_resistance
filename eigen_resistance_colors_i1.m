%% Eigenresistance Matlab Code

%@authors: Thomas Jagielski and Sparsh Bansal
clear

%% Load images
train = zeros(250,600,3,78);
train_set = zeros(250*600*3,79);
for k=0:78
    image_train = imread(strcat('./initialized_train/', int2str(k), '.png'));
    train(:,:,:,k+1) = image_train;
    for a=1:3
        rgb_reshaped = reshape(train(:,:,a,k+1),[250*600, 1]);
        train_set((a-1)*(250*600)+1:a*(250*600),k+1) = rgb_reshaped;
    end
end

test = zeros(250,600,3,100);
test_set = zeros(250*600*3,100);
for k=0:99
    image_test = imread(strcat('./initialized_test/', int2str(k), '.png'));
    test(:,:,:,k+1) = image_test;
    for a=1:3
        rgb_reshaped = reshape(test(:,:,a,k+1),[250*600, 1]);
        test_set((a-1)*(250*600)+1:a*(250*600),k+1) = rgb_reshaped;
    end
end

%% Convert images to black and white
%figure()
%imagesc(bw_train(:,:,3))
%colormap 'gray'

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