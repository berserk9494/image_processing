function output_image = bilat_filter_joint(input_image, guidance_image,radius, sigma_s, sigma_r, sigma_t)
    % Compute the spatial weights
    [X,Y]=meshgrid(-radius:radius,-radius:radius);
    spatial_weights = exp(-(X.^2+Y.^2)/(2*sigma_s^2)); 

    % Create waitbar.
    h = waitbar(0,'Wait ,Applying joint bilateral filter...');
    set(h,'Name','Joint Bilateral Fiter Processing');
    
    % Compute the size of the input and guidance images
    [rows, cols] = size(input_image);
    
    % Initialize the output image
    output_image = zeros(size(input_image));
    
    % Apply the improved joint bilateral filter
    for i = 1:rows
        for j = 1:cols
            % Compute the local region boundaries
            row_min = max(i - radius, 1);
            row_max = min(i + radius, rows);
            col_min = max(j - radius, 1);
            col_max = min(j + radius, cols);
            
            % Extract the local region of the input and guidance images
            local_input = input_image(row_min:row_max, col_min:col_max);
            local_guidance = guidance_image(row_min:row_max, col_min:col_max);
            
            % Compute the range weights
            range_weights = exp(-(local_input-input_image(i, j)).^2 / (2 * sigma_r^2));
              
            % Compute the temporal weights
            temporal_weights = exp(-(local_guidance-guidance_image(i, j)).^2 / (2 * sigma_t^2));
            
            % Compute the combined weights
            weights =  spatial_weights((row_min:row_max)-i+radius+1,(col_min:col_max)-j+radius+1).*range_weights.*temporal_weights;
            
            % Apply the weighted average to filter the input image
            output_image(i, j) = sum(weights(:).*local_input(:)) / sum(weights(:));
        end
        waitbar(i/rows);
    end
   % Close waitbar.
   close(h);
end
