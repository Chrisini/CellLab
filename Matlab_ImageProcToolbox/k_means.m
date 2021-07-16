testset = 0;
pipeline_mode = 1;
single_cell_mode = 0;
if testset == 1
    path = "C:\Users\Prinzessin\Documents\LifeSci\named_images_split\test\"
    pipeline_path = "C:\Users\Prinzessin\Documents\LifeSci\named_comp_masks\testset_pipeline_matlab_kmeans\"
    result_path = "C:\Users\Prinzessin\Documents\LifeSci\named_comp_masks\testset_matlab_kmeans\"
else
    path = "C:\Users\Prinzessin\Documents\LifeSci\named_images_type\alive\"
    pipeline_path = "C:\Users\Prinzessin\Documents\LifeSci\named_comp_masks\pipeline_matlab_kmeans\"
    result_path = "C:\Users\Prinzessin\Documents\LifeSci\named_comp_masks\matlab_kmeans\"
end
file_list = dir(path + "*.jpg")

clusters = 5;

allFileNames = {file_list(:).name};
for k = 1 : length(allFileNames)

    image = imread(path + allFileNames{k});
       
    %figure
    %imshow(image)

    % convert to CIE L*a*b space
    lab_image = rgb2lab(image);
    ab = lab_image(:,:,2:3);
    ab = im2single(ab);
    
    [pixel_labels, centers] = imsegkmeans(ab, clusters); % ,'NumAttempts',3)
    
    red = -999
    center = 0
    for c = 1 : length(centers)
        % get red most value
        if red < centers(c,1)
            red = centers(c,1)
            center = c
        end
    end
    
    mask1 = pixel_labels==center;
    
    cluster1 = image .* uint8(mask1);
    
    close
    figure
    imshow(cluster1)
    
    bw = imbinarize(rgb2gray(cluster1));
    bw = bwareaopen(bw,50);
    
    figure
    imshow(bw)
    
    CC = bwconncomp(rgb2gray(cluster1))
    
    CC.NumObjects
    
    %[L,n] = bwlabel(rgb2gray(cluster1))
    
    stats = regionprops(CC,'Area')
    
    counter = 0
    for i = 1 : CC.NumObjects
        areaaa = stats(i)
        if areaaa.Area > 15
            counter = counter + 1
            figure
            grain = false(size(bw));
            grain(CC.PixelIdxList{i}) = true;
            imshow(grain)
        end
    end
    counter
    
    
    
    figure
    imshow(image)
    
    %stats
    %imshow(L)
    %n
    
    % imshow(pixel_labels,[])
    
    break
    
end