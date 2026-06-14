function plotTFSurface(Z)
    [M,N] = size(Z);
    figure,imshow(Z,[]),title('H - TF');
    % Zoom in on the center of the plot
    ylim([floor(M/2)-100, floor(M/2)+100]);
    xlim([floor(N/2)-100, floor(N/2)+100]);

    % Create the surface plot
    figure;
    surf(Z,'FaceAlpha',0.5);
    xlabel('X');
    ylabel('Y');
    zlabel('Transfer Function');
    title('Surface Plot of Transfer Function');

    % Zoom in on the center of the plot
    ylim([floor(M/2)-100, floor(M/2)+100]);
    xlim([floor(N/2)-100, floor(N/2)+100]);

end