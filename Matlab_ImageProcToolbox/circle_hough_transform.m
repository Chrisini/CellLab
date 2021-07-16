testset = 1;
pipeline_mode = 1;
single_cell_mode = 0;
if testset == 1
    path = "C:\Users\Prinzessin\Documents\LifeSci\named_images_split\test\"
    pipeline_path = "C:\Users\Prinzessin\Documents\LifeSci\named_comp_masks\testset_pipeline_matlab_cht\"
    result_path = "C:\Users\Prinzessin\Documents\LifeSci\named_comp_masks\testset_matlab_cht\"
else
    path = "C:\Users\Prinzessin\Documents\LifeSci\named_images_type\dead\"
    pipeline_path = "C:\Users\Prinzessin\Documents\LifeSci\named_comp_masks\pipeline_matlab_cht\"
    result_path = "C:\Users\Prinzessin\Documents\LifeSci\named_comp_masks\matlab_cht\"
end
file_list = dir(path + "*.jpg")

sensitivity_value = 0.90 % 92
edge_threshold = 0.1
radius_range = [5 10]

allFileNames = {file_list(:).name};
for k = 1 : length(allFileNames)

    image = imread(path + allFileNames{k});

    if pipeline_mode == 1
        figure
        imshow(image)
        short = imdistline(gca,[100 110],[100 100]);
        setLabelVisible(short, false);
        long = imdistline(gca,[100 120],[150 150]);
        setLabelVisible(long, false);
        set(gca,'position',[0 0 1 1],'units','normalized')
        exportgraphics(gca, pipeline_path + "matlab_distline_" + allFileNames{k});

        % dark
        imshow(image)
        % radius range: 5 to 10
        [centersDark,radiiDark] = imfindcircles(image, radius_range, 'ObjectPolarity','dark', ... 
        'Sensitivity', sensitivity_value);
        hDark = viscircles(centersDark, radiiDark, 'Color','w', 'LineWidth', 0.5);
        set(gca,'position',[0 0 1 1],'units','normalized')
        exportgraphics(gca, pipeline_path + "matlab_dark_" + allFileNames{k});
        
        % bright
        imshow(image)
        [centersBright,radiiBright,metricBright] = imfindcircles(image,radius_range, ...
        'ObjectPolarity','bright','Sensitivity',sensitivity_value,'EdgeThreshold',edge_threshold);
        hBright = viscircles(centersBright, radiiBright, 'Color','w', 'LineWidth', 0.5);
        set(gca,'position',[0 0 1 1],'units','normalized')
        exportgraphics(gca, pipeline_path + "matlab_bright_" + allFileNames{k});

        black_image=zeros(size(image),'uint8');
        imshow(black_image)
        [centersBright,radiiBright,metricBright] = imfindcircles(image,radius_range, ...
        'ObjectPolarity','bright','Sensitivity',sensitivity_value,'EdgeThreshold',edge_threshold);
        for i = 1 : length(radiiBright)
            center = centersBright(i,:)
            radius = radiiBright(i)

            pos = [center(1)-radius, center(2)-radius, radius*2, radius*2]
            one_circle = rectangle('Position',pos,'Curvature',[1 1], 'FaceColor',[1 1 1], 'EdgeColor',[1 1 1], 'LineWidth', 3)
        end
        set(gca,'position',[0 0 1 1],'units','normalized')
        exportgraphics(gca, pipeline_path + "dead_all_" + allFileNames{k});
        close
    end
    
    if single_cell_mode == 1
        [centersBright,radiiBright,metricBright] = imfindcircles(image,radius_range, ...
            'ObjectPolarity','bright','Sensitivity',sensitivity_value,'EdgeThreshold',edge_threshold);
        
        for i = 1 : length(radiiBright)
            black_image=zeros(size(image),'uint8');
            imshow(black_image)
            
            center = centersBright(i,:)
            radius = radiiBright(i)

            pos = [center(1)-radius, center(2)-radius, radius*2, radius*2]
            one_circle = rectangle('Position',pos,'Curvature',[1 1], 'FaceColor',[1 1 1], 'EdgeColor',[1 1 1], 'LineWidth', 3)
            set(gca,'position',[0 0 1 1],'units','normalized')
            a = i-1
            exportgraphics(gca, result_path + "dead_" + a + "_" + allFileNames{k});
            close
        end
    end
        
end