function data = augmentData(data)
    % Randomly flip images and bounding boxes horizontally.
    tform = randomAffine2d('XReflection',true);
    
    %tform = randomAffine2d('Rotation', [-180, 180]);
    rout = affineOutputView(size(data{1}),tform);
    data{1} = imwarp(data{1},tform,'OutputView',rout);
    data{2} = bboxwarp(data{2},tform,rout);
end