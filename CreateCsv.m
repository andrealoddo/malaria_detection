%% 1. Load Data Set
if exist('data/mp-idb-falciparum1.mat', 'file')
    load('data/mp-idb-falciparum.mat');
else
    % Images and GT Labels Datapath - Local
    impath = 'C:\Users\loand\Documents\GitHub\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\Falciparum\img';
    labpath = 'C:\Users\loand\Documents\GitHub\MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis\Falciparum\gt';

    % Images and GT Labels Datapath - Server
    %impath = '/home/server/MATLAB/dataset/MP-IDB/Falciparum/img';
    %labpath = '/home/server/MATLAB/dataset/MP-IDB/Falciparum/gt';

    % Images and Labels Datastore
    imds = imageDatastore(impath);
    lds = imageDatastore(labpath);
    
    % The data is stored in a two-column table, where the first column 
    % contains the image file paths and the second column contains the bounding boxes.
    malariaDataset = table;
    malariaDataset.imageFilename = imds.Files(:);

csvTable = [];
%% 2. Conversion of BW ground-truths to rectangular bounding boxes to train the detector 
    for i=1:numel(imds.Files)
        row = 1;
        I = imread(imds.Files{i});
        L = imread(lds.Files{i});

        % Obtain Bounding Boxes --- TODO transform to function
        L_labels = imbinarize(L);
        L_props = regionprops(L_labels, 'BoundingBox'); % for Object Detection
        L_props2 = regionprops(L_labels, 'PixelList'); % for Semantic Segmentation

        bboxNumber = max(size(L_props));
        %figure; imshow(I);

        %parasites = zeros(bboxNumber, 4);
        parasites = [];
        for k = 1:bboxNumber
            box = L_props(k).BoundingBox;

            if(ceil(box(3)) * ceil(box(4)) > 20)
                parasites(row, 1:4) = [ ceil(box(1)), ceil(box(2)), ceil(box(3)), ceil(box(4)) ];
                row = row + 1;
                %rectangle('Position', [ceil(box(1)), ceil(box(2)), ceil(box(3)), ceil(box(4))], 'EdgeColor', 'r', 'LineWidth', 2)
            
                csvTable = [csvTable; imds.Files{i}, "parasite", ceil(box(1)), ceil(box(2)), ceil(box(3)), ceil(box(4))];
            end

        end

        malariaDataset.parasite{i} = parasites;
        
    end
    
    %filename,cell_type,xmin,xmax,ymin,ymax
    
    header = {'filename','parasite_type','xmin','xmax','ymin','ymax'};
    csvTableHeader = [header; csvTable];
        writematrix(csvTableHeader, 'data/mp-idb-falciparum.csv');
end