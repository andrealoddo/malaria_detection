%% 1. Load Data Set
% Images and GT Labels Datapath - Local
impath = 'C:\Users\loand\Documents\GitHub\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\Falciparum\img';
labpath = 'C:\Users\loand\Documents\GitHub\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\Falciparum\gt';

% Images and GT Labels Datapath - Server
%impath = '/home/server/MATLAB/dataset/MP-IDB/Falciparum/img';
%labpath = '/home/server/MATLAB/dataset/MP-IDB/Falciparum/gt';

% Images and Labels Datastore
imds = imageDatastore(impath);
lds = imageDatastore(labpath);

malariaDataset = table;
malariaDataset.imageFilename = imds.Files(:);

csvTable = table;

%% 2. Conversion of BW ground-truths to rectangular bounding boxes to train the detector 
for i=1:numel(imds.Files) 
    
    I = imread(imds.Files{i});
    L = imread(lds.Files{i});
    [width, height] = size(L);

    % Obtain Bounding Boxes --- TODO transform to function
    L_labels = logical(L);
    L_props = regionprops(L_labels, 'BoundingBox'); % for Object Detection
    %L_props2 = regionprops(L_labels, 'PixelList'); % for Semantic Segmentation

    bboxNumber = max(size(L_props));
    %figure; imshow(I);
    
    parasites = zeros(bboxNumber, 4);
    for k = 1:bboxNumber
        box = L_props(k).BoundingBox;
        %rectangle('Position', [box(1), box(2), box(3), box(4)], 'EdgeColor', 'r', 'LineWidth', 2)
        parasites(k, 1:4) = [ round(box(1)), round(box(2)), round(box(3)), round(box(4)) ];
        
        [filepath, name, ext] = fileparts(imds.Files{i});
        
        csvTable.filename{i+k-1} = strcat(name, ext);
        csvTable.width{i+k-1} = width;
        csvTable.height{i+k-1} = height;
        csvTable.class{i+k-1} = 'parasite';
        csvTable.xmin{i+k-1} = round(box(1));
        csvTable.ymin{i+k-1} = round(box(2));
        csvTable.xmax{i+k-1} = round(box(3));
        csvTable.ymax{i+k-1} = round(box(4));
                
    end
    
    malariaDataset.parasite{i} = parasites;
       
end

writetable(csvTable, 'data\mp-idb-falciparum.csv'); 