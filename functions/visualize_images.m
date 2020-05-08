function [perm] = visualize_images(imds)
    
    % Number of Images
    num_images = length(imds.Labels);

    % Visualize random 20 images
    perm = randperm(num_images, 20);
    figure;
    for idx = 1:20

        subplot(4, 5, idx);
        imshow(imread(imds.Files{perm(idx)}));
        title(sprintf('%s', imds.Labels(perm(idx))))

    end

end