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


%% To generate a plot of accuracies compared to the number of eigenfaces used
accuracy = zeros(2,size(test_weights,2));
for n=1:79
    % Initialize a vector of zeros to represent averages
    % By preallcating space, we are saving in runtime
    count = zeros(test_weights(2));
    % Calculate the new training weights based off the new set of eigen
    % vectors
    train_weights = U(:,1:n)' * train_reshape;
    % Calculate the new testing weights based off the new set of eigen
    % vectors
    test_weights = U(:,1:n)' * test_reshape;

    % For loop to compare the distance of each column of weights to the
    % training image weights
    for num = 1:size(test, 3)
        % Find the index of the minimum distance between two vectors
        [Y,I] = min(vecnorm(test_weights(:,num) - train_weights));
        % Compare if the names are equal for the images to find accuracy
        if train_labels(I) == test_labels(num)
            accuracy(num) = 1;
        else
            accuracy(num) = 0;
        end
    end
    % Average the count matrix of ones and zeros
    % One: correctly identifies face
    % Zero: incorrectly identifies face
    percent_correct = mean(count);
    accuracy(1,n) = n;
    accuracy(2,n) = percent_correct;    
end
[max_accuracy_without_avg, num_eigen_faces] = max(accuracy(2,:))
figure()
plot(accuracy(1,:),accuracy(2,:),'*')
xlabel('Number of Eigenvectors Used')
ylabel('Percent Correct (in decimal form)')
title('Percent Correct VS The Number of Eigenfaces Used')