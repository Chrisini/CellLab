% To build the program, run 'mex -I.\ segment.cpp'
image = imread('data/cell56.jpg');
label = seg(image, 0.5, 50, 500);
showregion(label);

