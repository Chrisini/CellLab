testset = 0;
pipeline_mode = 1;
single_cell_mode = 0;
if testset == 1
    path = "C:\Users\Prinzessin\Documents\LifeSci\named_images_split\test\"
    pipeline_path = "C:\Users\Prinzessin\Documents\LifeSci\named_comp_masks\testset_pipeline_matlab_kmeans\"
    result_path = "C:\Users\Prinzessin\Documents\LifeSci\named_comp_masks\testset_matlab_kmeans\"
else
    path = "C:\Users\Prinzessin\Documents\LifeSci\named_images_type\inhib\"
    pipeline_path = "C:\Users\Prinzessin\Documents\LifeSci\named_comp_masks\pipeline_matlab_morph\"
    result_path = "C:\Users\Prinzessin\Documents\LifeSci\named_comp_masks\matlab_morph\"
end
file_list = dir(path + "*.jpg")

clusters = 5;

allFileNames = {file_list(:).name};
for k = 1 : length(allFileNames)
   
    image = imread(path + allFileNames{k});
    % image = imgaussfilt(image,5);
    image_gray = rgb2gray(image);
    figure 
    imshow(image_gray)
    figure
    [~,threshold] = edge(image_gray,'sobel');
    fudgeFactor = 0.4;
    image_edges = edge(image_gray,'sobel',threshold * fudgeFactor);
    
    %figure
    imshow(image_edges)
    
    image_edges = bwareaopen(image_edges,5);
    
    %figure
    imshow(image_edges)
       
    % structuring elements
    se90 = strel('disk',2);
    se0 = strel('disk',2);
    
    image_dilate = imdilate(image_edges, se0);
    
    %figure
    imshow(image_dilate)
    
    %seD = strel('diamond',1);
    %image_dilate = imerode(image_dilate, seD);
    
    %figure
    %imshow(image_dilate)
    
    %image_open = bwareaopen(image_dilate,50);
    
    %figure
    %imshow(image_open)
    
    image_fill = imfill (image_dilate, 'holes');
    
    %figure
    imshow(image_fill)
    
    image_clear = imerode(image_fill, se90);
    
    %figure
    imshow(image_clear)
    
    %seD = strel('diamond',1);
    %image_smooth = imerode(image_clear, seD);
    %image_smooth = imerode(image_smooth, seD);
    
    %figure
    %imshow(image_smooth)
    %close
    
    
    kernel = ones(10)/10^2;
    image_fill = conv2(single(image_clear), kernel, 'same');
    image_fill = image_fill > 0.5; % Rethreshold
    
    imshow(image_fill)
    %{
    %figure
    D = bwdist(~image_fill, 'quasi-euclidean');
    imshow(D,[])
    title('Distance Transform of Binary Image')
    
    %figure
    D = -D;
    imshow(D,[])
    L = watershed(D, 8);
    L(~image_fill) = 0;
    
    
    rgb = label2rgb(L,'jet',[.5 .5 .5]);
    imshow(rgb)
    %}
    %break
    
    %exportgraphics(gca, pipeline_path + "inhib_all_" + allFileNames{k});
    
    
    
end