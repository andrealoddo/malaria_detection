function Iout = preprocess_mpidb_label_images(filename, desired_size)

    % This function resizes the labels of MP-IDB to an image of desired size

    % Read the Image
    I = imread(filename);

    % Resize the image
    Iout = imresize(I, [desired_size(1) desired_size(2)]);
    
end