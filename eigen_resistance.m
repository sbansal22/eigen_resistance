%% Eigenresistance Matlab Code

%@authors: Thomas Jagielski and Sparsh Bansal

%% Load images
train = zeros(250,600,3,78);
for k=0:78
    train(:,:,:,k+1) = imread(strcat('./initialized_train/', int2str(k), '.png'));
end

test = zeros(250,600,3,100);
for k=0:99
    test(:,:,:,k+1) = imread(strcat('./initialized_test/', int2str(k), '.png'));
end

%% Convert images to black and white
bw_train = zeros(250,600,size(train, 4));
for l=size(train,4)
    bw_train(:,:,l) = rgb2gray(train(:,:,:,l));
end

bw_test = zeros(250,600,size(test, 4));
for l=size(test, 4)
    bw_train(:,:,l) = rgb2gray(train(:,:,:,l));
end

