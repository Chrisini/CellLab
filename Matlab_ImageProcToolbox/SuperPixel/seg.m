function label = seg(image, sigma, min_size, K)
% Segment given image into super-pixels.
% Input:
%   image:     [M * N * 3, REQ] images to be segmented
%   sigma:     [REQ] standard deviation of Gaussian smooth kernel
%   min_size:  [REQ] minimum number of pixels in super-pixels
%   K:         [REQ] number of super-pixels 
% Output:
%   label:     [M * N] super-pixel label for each pixel

label = segment(double(image), sigma, min_size, K);