addpath('functions');
% Deep Learning for Medical Imaging: Malaria Detection
% article source: https://blogs.mathworks.com/deep-learning/2019/11/14/deep-learning-for-medical-imaging-malaria-detection/
% dataset source: https://lhncbc.nlm.nih.gov/publication/pub9932

%% 1. Load the Database
% Images Datapath
datapath='..\dataset_NIH';

% Image Datastore
imds=imageDatastore(datapath, 'IncludeSubfolders',true, 'LabelSource','foldernames');

% Determine the split up
total_split = countEachLabel(imds);

%% 2. Visualize the images

%perm = visualize_images(imds);


%% 3. Preprocessing
% You can observe that most of the parasitized images contain a ‘red spot’ which indicates the presence of plasmodium. 
% However, there are some images which are difficult to distinguish. You could dig into the dataset to check some 
% tough examples. Also, there is a spectrum of image colors as these images are captured using different microscopes
% at different resolutions. We will address these by preprocessing the images in the subsequent section.

%visualize_preprocessed_images(imds, perm);


%% 4. Training, Testing and Validation
% Split the Training and Testing Dataset
train_percent = 0.8;
[imdsTrain, imdsTest] = splitEachLabel(imds, train_percent, 'randomize');

% Split the Training and Validation
valid_percent = 0.1;
[imdsValid, imdsTrain] = splitEachLabel(imdsTrain,valid_percent,'randomize');

train_split = countEachLabel(imdsTrain);

%% 5. Deep Learning Approach
% Load AlexNet
net = alexnet;
 
% Transfer the layers except the last 3 layers
layersTransfer = net.Layers(1:end-3);
 
% Clear the existing alexnet architecture
clear net;
 
% Define the new layers 
numClasses = numel(categories(imdsTrain.Labels));
 
% New layers 
layers=[
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor',20, 'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%% 6. Preprocess Training and Validation Dataset
imdsTrain.ReadFcn = @(filename)preprocess_malaria_images(filename, [layers(1).InputSize(1), layers(1).InputSize(2)]);
imdsValid.ReadFcn = @(filename)preprocess_malaria_images(filename, [layers(1).InputSize(1), layers(1).InputSize(2)]);

%% 7. Train the network
options = trainingOptions('adam', ...
    'MiniBatchSize',128, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',imdsValid, ...
    'ValidationFrequency',50,'ValidationPatience',4, ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment', 'gpu');
% Train the network
netTransfer = trainNetwork(imdsTrain, layers, options);

%% 8. Testing
% Preprocess the test cases similar to the training
imdsTest.ReadFcn=@(filename)preprocess_malaria_images(filename,[layers(1).InputSize(1), layers(1).InputSize(2)]);

% Predict Test Labels using Classify command
[predicted_labels,posterior]=classify(netTransfer, imdsTest);


%% 9. Performance Study
% Actual Labels
actual_labels=imdsTest.Labels;

% Confusion Matrix
figure;
plotconfusion(actual_labels,predicted_labels)
title('Confusion Matrix: AlexNet');

%% 10. ROC Curve
test_labels=double(nominal(imdsTest.Labels));

% ROC Curve - Our target class is the first class in this scenario.
[fp_rate,tp_rate,T,AUC]=perfcurve(test_labels,posterior(:,1),1);
figure;
plot(fp_rate,tp_rate,'b-');hold on;
grid on;
xlabel('False Positive Rate');
ylabel('Detection Rate');

% Area under the ROC value
AUC