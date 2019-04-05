%% Eigenresistance Matlab Code

%@authors: Thomas Jagielski and Sparsh Bansal
clear
close all

%% Load images
%% Load Labels
labels = xlsread('values_final.xlsx');
train_labels = labels(1:80,2);
test_labels = labels(1:40,3);

%all_weights = [0.1,0,1;
 %              0.1,0.1,1;
 %              0.5,0,0.7;
 %              0.05,0,1;
 %              0.2,0,1;
 %              0.25,0,1;
 %              0.3,0,1;
 %              0.35,0,1;
 %              0.4,0,1;
 %              0.45,0,1;
 %              0.1,0,0.9;
 %              0.1,0.1,0.7;
 %              0.5,0,0.85;
 %              0.05,0,0.7;
 %              0.2,0,0.88;
 %              0.25,0,0.76;
 %              0.3,0,0.88;
 %              0.35,0,0.94;
 %              0.4,0,0.9;
 %              0.45,0,0.75;
 %              1,1,1;
 %              0.05,0,1];
all_weights = [0.5,0.5,1]
           
accuracy_values = all_weights;
accuracy_values = [accuracy_values, zeros(size(accuracy_values,1),1)];
 %%          
for m=1:size(all_weights,1)
    weights = all_weights(m,:);
    train = zeros(250,600,3,80);
    train_set = zeros(250*600*3,80);
    for k=0:79
        image_train = imread(strcat('./new_train_init/', int2str(k), '.png'));
        train(:,:,:,k+1) = image_train;
        for a=1:3
            rgb_reshaped = reshape(train(:,:,a,k+1),[250*600, 1]);
            train_set((a-1)*(250*600)+1:a*(250*600),k+1) = weights(a) * rgb_reshaped;
        end
    end

    test = zeros(250,600,3,40);
    test_set = zeros(250*600*3,40);
    for k=0:39
        image_test = imread(strcat('./new_test_init/', int2str(k), '.png'));
        test(:,:,:,k+1) = image_test;
        for a=1:3
            rgb_reshaped = reshape(test(:,:,a,k+1),[250*600, 1]);
            test_set((a-1)*(250*600)+1:a*(250*600),k+1) = weights(a) * rgb_reshaped;
        end
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
    percent_correct = mean(accuracy);
    accuracy_values(m,end) = percent_correct;
end
%%
figure()
x = accuracy_values(:,1);
y = accuracy_values(:,3);
z = accuracy_values(:,4);
pointsize = 20;
scatter(x,y,pointsize,z)
c = colorbar;
colormap(jet);
ylabel(c, 'Percent Correct')
xlabel('Red Scale Factor')
ylabel('Blue Scale Factor')
title('Red and Blue Weight Multipliers')
axis([-0.05 0.55 0.65 1.05])

%%
figure()
eigen = U .* train_set;
eigenresistor = reshape(eigen(:,80),250,600,3);
imshow(eigenresistor)

%%

accuracy = zeros(2,40);
for n=1:40
    % Initialize a vector of zeros to represent averages
    % By preallcating space, we are saving in runtime
    count = zeros(1,40);
    % Calculate the new training weights based off the new set of eigen
    % vectors
    train_weights = U(:,1:n)' * train_set;
    % Calculate the new testing weights based off the new set of eigen
    % vectors
    test_weights = U(:,1:n)' * test_set;

    % For loop to compare the distance of each column of weights to the
    % training image weights
    for num = 1:40
        % Find the index of the minimum distance between two vectors
        [Y,I] = min(vecnorm(test_weights(:,num) - train_weights));

        if train_labels(I) == test_labels(num)
            count(num) = 1;
        else
            count(num) = 0;
        end
    end
    % Average the count matrix of ones and zeros
    % One: correctly identifies face
    % Zero: incorrectly identifies face
    percent_correct = mean(count);
    accuracy(1,n) = n;
    accuracy(2,n) = percent_correct;    
end
[max_accuracy_without_avg, num_eigen_faces] = max(accuracy(2,:));
figure()
plot(accuracy(1,:),accuracy(2,:),'*')
xlabel('Number of Eigenvectors Used')
ylabel('Percent Correct (in decimal form)')
title('Percent Correct VS The Number of Eigenvectors Used')