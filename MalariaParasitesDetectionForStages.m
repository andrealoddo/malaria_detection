%% Object Detection Using Faster R-CNN Deep Learning
% source: https://www.mathworks.com/help/vision/examples/object-detection-using-faster-r-cnn-deep-learning.html

%% 0. Show figures
showFigures = false;
if ~showFigures
    set(0,'DefaultFigureVisible','off')
end

%% 0. Settings
if ispc
    datasetpath = 'C:\Users\loand\Documents\GitHub\Datasets\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\Falciparum\';
    impath = strcat(datasetpath, 'img');
    labelpath = strcat(datasetpath, 'gt');
    labelfile = strcat(datasetpath, 'mp-idb-falciparum.csv');
else
    datasetpath = '/home/server/MATLAB/dataset/MP-IDB/Falciparum/';
    impath = strcat(datasetpath, 'img');
    labelpath = strcat(datasetpath, 'gt');
    labelfile = strcat(datasetpath, 'mp-idb-falciparum.csv');
end

%% 0. Train the detector or load a pre-trained network
doTrainingAndEval = true;
if ~doTrainingAndEval && exist('models/detectors/fasterRCNN-ResNet50.mat', 'file')
    load('models/detectors/fasterRCNN-ResNet50.mat');
end

%% 0. Load pre-trained model for RPN
if exist('models/resnet18_malaria_NIH_MPIDB.mat', 'file')
    load('models/resnet18_malaria_NIH_MPIDB.mat');
end

%% 1. Load Data Set
if exist('data/mp-idb-falciparum.mat', 'file')
    load('data/mp-idb-falciparum.mat');
else
    
    % Creation of Images and Labels Datastore
    imds = imageDatastore(impath);
    lds = imageDatastore(labelpath);

    % Image and Labels pre-processing
    inputSize = [500, 375];
    imds.ReadFcn = @(filename)preprocess_mpidb_images(filename, [inputSize(1), inputSize(2)]);
    lds.ReadFcn = @(filename)preprocess_mpidb_label_images(filename, [inputSize(1), inputSize(2)]);

    % The data is stored in a two-column table, where the first column 
    % contains the image file paths and the second column contains the bounding boxes.
    malariaDataset = table;
    malariaDataset.imageFilename = imds.Files(:);

    %% 2. Conversion of ground-truths to rectangular bounding boxes
    for i=1:numel(imds.Files)
        
        I = imread(imds.Files{i});
        figure; imshow(I);

        imageParasites = readtable(labelfile);
        
        countRing = 1;
        countTro = 1;
        countSchi = 1;
        countGame = 1;
        ring = [];
        tro = [];
        schi = [];
        game = [];
        
        [filepath, filename, ext] = fileparts(imds.Files{i}); 
        currentImageParasites = imageParasites(strcmp(imageParasites.filename, strcat(filename, ext)), :);
        
        for k = 1:height(currentImageParasites)
            if(currentImageParasites.ymin(k) * currentImageParasites.ymax(k) < 20)
                
                disp(strcat('error in image: ', currentImageParasite.filename));
                
            end
            
            currentBoundingBox = [ currentImageParasites.xmin(k), currentImageParasites.xmax(k), currentImageParasites.ymin(k), currentImageParasites.ymax(k) ];
            
            if(strcmp( currentImageParasites.parasite_type(k), 'ring' ))
                ring(countRing, 1:4) = currentBoundingBox;
                countRing = countRing + 1;
            elseif(strcmp( currentImageParasites.parasite_type(k), 'tro' ))
                tro(countTro, 1:4) = currentBoundingBox;
                countTro = countTro + 1;
            elseif(strcmp( currentImageParasites.parasite_type(k), 'schi' ))
                schi(countSchi, 1:4) = currentBoundingBox;
                countSchi = countSchi + 1;
            elseif(strcmp( currentImageParasites.parasite_type(k), 'game' ))
                game(countGame, 1:4) = currentBoundingBox;[ currentImageParasites.xmin, currentImageParasites.xmax, currentImageParasites.ymin, currentImageParasites.ymax ];
                countGame = countGame + 1;
            end

            rectangle('Position', currentBoundingBox, 'EdgeColor', 'r', 'LineWidth', 2)

        end

        malariaDataset.ring{i} = ring;
        malariaDataset.tro{i} = tro;
        malariaDataset.schi{i} = schi;
        malariaDataset.game{i} = game;
        
    end
    
    save('data/mp-idb-falciparum.mat', 'malariaDataset');
    
end


%% 3. Split the dataset into training, validation, and test sets
% The training data is stored in a table. 
% The first column contains the path to the image files. 
% The remaining columns contain the ROI labels for objectes. 

% Split the dataset into training, validation, and test sets. 
% Select 80% of the data for training, 10% for validation, 
% and the rest for testing the trained detector.
rng(0)
shuffledIndices = randperm(height(malariaDataset));
idx = floor(0.6 * height(malariaDataset));

trainingIdx = 1:idx;
trainingDataTbl = malariaDataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = malariaDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = malariaDataset(shuffledIndices(testIdx),:);

% Use imageDatastore and boxLabelDatastore to create datastores 
% for loading the image and label data during training and evaluation.
imdsTrain = imageDatastore(trainingDataTbl{:, 'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:, 2:5));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:, 2:5));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:, 2:5));


%% 4. Combine image and box label datastores 
% Combine image and box label datastores.
trainingData = combine(imdsTrain, bldsTrain);
validationData = combine(imdsValidation, bldsValidation);
testData = combine(imdsTest, bldsTest);


%% 5. Display one of the training images and box labels.
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)


%% 6. Create Faster R-CNN Detection Network
inputSize = [224 224 3];

preprocessedTrainingData = transform(trainingData, @(data)preprocessData(data, inputSize));
numAnchors = 12; 
maxNumAnchors = 20;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(preprocessedTrainingData, numAnchors);
%estimateOptimalNumberOfAnchors(preprocessedTrainingData, maxNumAnchors);
featureExtractionNetwork = malarianet; %resnet50
%featureLayer = 'activation_40_relu';
featureLayer = 'conv1_relu';
numClasses = width(malariaDataset)-1;
lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);


%% 7. Display one of the training images and box labels.
data = read(preprocessedTrainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure; imshow(annotatedImage)


%% 8. Data Augmentation
augmentedTrainingData = transform(trainingData,@augmentData);

augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
figure; montage(augmentedData,'BorderSize',10)


%% 9. Preprocess Training Data
trainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
validationData = transform(validationData,@(data)preprocessData(data,inputSize));

data = read(trainingData);

I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure; imshow(annotatedImage)


%% 10. Train Faster R-CNN
% options = trainingOptions('sgdm',...
%    'MaxEpochs',30,...
%    'MiniBatchSize',4,...
%    'InitialLearnRate',1e-5,...
%    'CheckpointPath',tempdir,...
%    'ValidationData',validationData);

options = trainingOptions('adam', ...
    'MaxEpochs', 30,...
    'MiniBatchSize', 2,...
    'InitialLearnRate', 3e-4,...
    'Shuffle', 'every-epoch', ...
    'CheckpointPath', 'temp', ...
    'ValidationData', validationData);

    
if doTrainingAndEval
    % Train the Faster R-CNN detector.
    % * Adjust NegativeOverlapRange and PositiveOverlapRange to ensure
    %   that training samples tightly overlap with ground truth.
    %[detector, info] = trainFasterRCNNObjectDetector(trainingData,lgraph,options, ...
    %    'NegativeOverlapRange',[0 0.3], ...
    %    'PositiveOverlapRange',[0.6 1]);    
    [detector, info] = trainFasterRCNNObjectDetector(trainingData,lgraph,options);
else
    % Load pretrained detector for the example.
    pretrained = load('models/detectors/fasterRCNN-ResNet50.mat');
    detector = pretrained.detector;
end


%% 11. Test detector
I = imread(testDataTbl.imageFilename{1});
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure; imshow(I)