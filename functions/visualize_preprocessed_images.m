function [] = visualize_preprocessed_images(imds, perm)
    
    figure;

    % Visualize the Preprocessed Images
    for idx = 1:20
     subplot(4, 5, idx);
     imshow(preprocess_malaria_images(imds.Files{perm(idx)}, [250 250]))
     title(sprintf('%s', imds.Labels(perm(idx))))
    end

end