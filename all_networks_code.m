addpath('functions');
% Deep Learning for Medical Imaging: Malaria Detection
% article source: https://blogs.mathworks.com/deep-learning/2019/11/14/deep-learning-for-medical-imaging-malaria-detection/
% dataset source: https://lhncbc.nlm.nih.gov/publication/pub9932

%% 1. Load the Database
% Images Datapath
datapath='..\dataset_NIH';

% Image Datastore
imds=imageDatastore(datapath, 'IncludeSubfolders', true, 'LabelSource','foldernames');

% Determine the split up
total_split = countEachLabel(imds);

%% 2. Visualize the images

%visualize_images(imds);


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


%% 5. Deep Learning Approach: 
%% a. Load pretrained network
% Load a pretrained net
net = alexnet;
%analyzeNetwork(net)


% Obtain input size dimension
net.Layers(1)
inputSize = net.Layers(1).InputSize;

%% b. Replace final layers
% The convolutional layers of the network extract image features that the 
% last learnable layer and the final classification layer use to classify 
% the input image. These two layers, 'loss3-classifier' and 'output' in 
% GoogLeNet, contain information on how to combine the features that the 
% network extracts into class probabilities, a loss value, 
% and predicted labels. To retrain a pretrained network to classify new images, 
% replace these two layers with new layers adapted to the new data set.
% Extract the layer graph from the trained network. 


% If the network is a SeriesNetwork object, such as AlexNet, VGG-16, or VGG-19, 
% then convert the list of layers in net.Layers to a layer graph.
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

% Find the names of the two layers to replace. You can do this manually or 
% you can use the supporting function findLayersToReplace to find these layers automatically.
[learnableLayer, classLayer] = findLayersToReplace(lgraph);

% Define the new layers 
numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name', 'new_fc', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name', 'new_conv', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

% The classification layer specifies the output classes of the network. 
% Replace the classification layer with a new one without class labels. 
% trainNetwork automatically sets the output classes of the layer at training time. 
newClassLayer = classificationLayer('Name', 'new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% To check that the new layers are connected correctly, plot the new layer 
% graph and zoom in on the last layers of the network.
figure('Units', 'normalized', 'Position', [0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

%% c. Freeze initial layers
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);


%% 6. Preprocess Training and Validation Dataset
imdsTrain.ReadFcn = @(filename)preprocess_malaria_images(filename, [layers(1).InputSize(1), layers(1).InputSize(2)]);
imdsValid.ReadFcn = @(filename)preprocess_malaria_images(filename, [layers(1).InputSize(1), layers(1).InputSize(2)]);

%% 7. Train the network
% malaria example params
solverName = 'adam';
miniBatchSize = 128;
valFrequency = 50;
MaxEpochs = 10;
InitialLearnRate = 1e-4;
ValidationPatience = 4;

% TransferLearningUsingGoogLeNetExample params
solverName = 'sgdm';
miniBatchSize = 2;
valFrequency = floor(numel(imdsTrain.Files)/miniBatchSize);
MaxEpochs = 6;
InitialLearnRate = 3e-4;
ValidationPatience = Inf;

options = trainingOptions(solverName, ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', MaxEpochs, ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate', InitialLearnRate, ...
    'ValidationData', imdsValid, ...
    'ValidationFrequency', valFrequency, ...
    'ValidationPatience', ValidationPatience, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
	'OutputFcn', @(info)stopIfAccuracyNotImproving(info,3));

% Train the network
netTransfer = trainNetwork(imdsTrain, lgraph, options);

%% 8. Testing
% Preprocess the test cases similar to the training
imdsTest.ReadFcn=@(filename)preprocess_malaria_images(filename, [layers(1).InputSize(1), layers(1).InputSize(2)]);

% Predict Test Labels using Classify command
[predicted_labels, posterior] = classify(netTransfer, imdsTest);


%% 9. Performance Study
% Actual Labels
actual_labels = imdsTest.Labels;

% Confusion Matrix
figure;
plotconfusion(actual_labels, predicted_labels)
title('Confusion Matrix: AlexNet');

%% 10. ROC Curve
test_labels = double(nominal(imdsTest.Labels));

% ROC Curve - Our target class is the first class in this scenario.
[fp_rate,tp_rate,T,AUC]=perfcurve(test_labels,posterior(:,1),1);
figure;
plot(fp_rate,tp_rate,'b-');hold on;
grid on;
xlabel('False Positive Rate');
ylabel('Detection Rate');

% Area under the ROC value
AUC


