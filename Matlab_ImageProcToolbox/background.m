
    blur = imgaussfilt(image,10);
    image = imsubtract(image, blur);
    
    colour = mean2(blur)
    [a, b, c] = size(image)
    black_image=zeros(a, b, c,'uint8');
    background = black_image;
    %black_image(:,:,:) = colour;
       
    for ind = 1:a
        for jnd = 1:b
            background(ind,jnd,:) = colour; %  [0, 1, 1];
        end
    end
    
    
    
    % background=cat(3, G, R, B)
    
    size(image)
    size(background)
    
    %image = imfuse(image, background); 
    
    % image = immultiply(image, background);
    
    image = imadd(image, background);
    
    level = graythresh(rgb2gray(image));
    BW = imbinarize(rgb2gray(image),level);
    
    BW = im2bw(image, 0.5);
    
    figure
    imshow(BW)
    