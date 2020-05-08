function Iout = preprocess_mpidb_label_images(filename, desired_size)

    % This function preprocesses malaria images using color constancy
    % technique and later reshapes them to an image of desired size
    % Author: Barath Narayanan

    % Read the Image
    I = imread(filename);


    % Resize the image
    Iout = imresize(I, [desired_size(1) desired_size(2)]);
    
end