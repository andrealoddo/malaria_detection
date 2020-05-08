%% 1. Load the Database
% Images and GT Labels Datapath - Local
impath = 'C:\Users\loand\Documents\GitHub\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\Falciparum\img';
labpath = 'C:\Users\loand\Documents\GitHub\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\Falciparum\gt';

% Images and GT Labels Datapath - Server
%impath = '/home/server/MATLAB/dataset/MP-IDB/Falciparum/img';
%labpath = '/home/server/MATLAB/dataset/MP-IDB/Falciparum/gt';

 
% Images and Labels Datastore
imds = imageDatastore(impath);
lds = imageDatastore(labpath);

% Determine the split up
%total_split = countEachLabel(imds);

%% Network loading 
if isfile('models/TrainingOnNihDataset/AlexNet.mat')
    load('models/TrainingOnNihDataset/AlexNet.mat');
else
    net = alexnet;
end
 
layers = net.Layers;
inputSize =  net.Layers(1).InputSize;

if isa(net, 'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

% Find the names of the two layers to replace. You can do this manually or 
% you can use the supporting function findLayersToReplace to find these layers automatically.
[learnableLayer, classLayer] = findLayersToReplace(lgraph);

% Define the new layers 
numClasses = 2; % Parasite vs Not Parasite

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


%% Labels pre-processing (resize)
lds.ReadFcn = @(filename)preprocess_mpidb_images(filename, [layers(1).InputSize(1), layers(1).InputSize(2)]);


% The data is stored in a two-column table, where the first column 
% contains the image file paths and the second column contains the vehicle bounding boxes.
% Split the dataset into training, validation, and test sets. Select 60% of the data for training, 
% 10% for validation, and the rest for testing the trained detector.
malariaDataset = table;
malariaDataset.imageFilename = imds.Files(:);


%% 2 semel. Conversion of BW ground-truths to rectangular bounding boxes to train the detector 
for i=1:numel(imds.Files)
    
    I = imread(imds.Files{i});
    L = imread(lds.Files{i});

    % Obtain Bounding Boxes --- TODO transform to function
    L_labels = bwlabel(L);
    L_props = regionprops(L_labels, 'BoundingBox'); % for Object Detection
    %L_props2 = regionprops(L_labels, 'PixelList'); % for Semantic Segmentation

    bboxNumber = max(size(L_props));
    %figure; imshow(I);
    
    parasites = zeros(bboxNumber, 4);
    for k = 1:bboxNumber
        box = L_props(k).BoundingBox;
        %rectangle('Position', [box(1), box(2), box(3), box(4)], 'EdgeColor', 'r', 'LineWidth', 2)
        parasites(k, 1:4) = [ round(box(1)), round(box(2)), round(box(3)), round(box(4))];
    end
    
    malariaDataset.parasite{i} = parasites;
    
end

%% 3. Train the detector
% The training data is stored in a table. 
% The first column contains the path to the image files. 
% The remaining columns contain the ROI labels for objectes. 

% Split the dataset into training, validation, and test sets. 
% Select the data for training, for validation, and for testing the trained detector.
rng(0)
shuffledIndices = randperm(height(malariaDataset));
train_perc = 0.8;
idx = floor(train_perc * height(malariaDataset));
trainingIdx = 1:idx;
trainingDataTbl = malariaDataset(shuffledIndices(trainingIdx),:);

trainingDataAugTbl = repelem(trainingDataTbl, 5, 1); 

valid_perc = 0.0;
validationIdx = idx+1 : idx + 1 + floor(valid_perc * length(shuffledIndices) );
validationDataTbl = malariaDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTb1 = malariaDataset(shuffledIndices(testIdx),:);


%% 4. Splitting into train/validation/test?
% Use imageDatastore and boxLabelDatastore to create datastores 
% or loading the image and label data during training and evaluation.

imdsTrain = imageDatastore(trainingDataTbl{:, 'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:, 'parasite'));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'parasite'));

imdsTest = imageDatastore(testDataTb1{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTb1(:,'parasite'));


%% 5. Preprocess Training and Validation Dataset
imdsTrain.ReadFcn = @(filename)preprocess_mpidb_images(filename, [layers(1).InputSize(1), layers(1).InputSize(2)]);
imdsValid.ReadFcn = @(filename)preprocess_mpidb_images(filename, [layers(1).InputSize(1), layers(1).InputSize(2)]);


%% X. Data augmentation
numRep = 5;
imTrainAug = repelem(imdsTrain.Files, numRep, 1); 
imdsTrainAug = imageDatastore(imTrainAug);

blTrainAug = repelem(bldsTrain.LabelData, numRep, 1); 
bldsTrainAug = boxLabelDatastore(cell2table(blTrainAug));

trainingDataAug = combine(imdsTrainAug, bldsTrainAug);
trainingDataAug = transform(trainingDataAug, @augmentData);


%% 6. Combination 
% Combine image and box label datastores.
trainingData = combine(imdsTrain, bldsTrain);
validationData = combine(imdsValidation, bldsValidation);
testData = combine(imdsTest, bldsTest);


%% 7. Train the network
% original parameters
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-6, ...
    'MaxEpochs', 30);

detector = trainRCNNObjectDetector(trainingDataAugTbl, lgraph, options, 'NegativeOverlapRange', [0 0.3]);


%% 8. Test the network
I = imread(testDataTb1.imageFilename{1});
[bboxes, scores, label] = detect(detector,I);

% Display the results.
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure, imshow(I); 


