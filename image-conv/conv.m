clc;
clear;
close all;

img = imread('test.jpg');
size(img)
size(img(:,:,1))

imshow(img);
weight=ones(5)*(0);
weight(3,1) = -1;
weight(3,2) = -1;
weight(3,3) = 2;
weight

r= double(img(:, :, 1));
g = double(img(:, :, 2));
b = double(img(:, :, 3));
r_out = conv2(r, weight, 'valid');
g_out = conv2(g, weight, 'valid');
b_out = conv2(b, weight, 'valid');


r_min_value = min(r_out(:));
r_max_value = max(r_out(:));
r_out = r_out - r_min_value;
r_out = r_out/(r_max_value-r_min_value);
r_out = uint8(r_out*255);

g_min_value = min(g_out(:));
g_max_value = max(g_out(:));
g_out = g_out - g_min_value;
g_out = g_out/(g_max_value-g_min_value);
g_out = uint8(g_out*255);

b_min_value = min(b_out(:));
b_max_value = max(b_out(:));
b_out = b_out - b_min_value;
b_out = b_out/(b_max_value-b_min_value);
b_out = uint8(b_out*255);
% 
% figure;
% imshow(r_out);
% figure;
% imshow(g_out);
% figure;
% imshow(b_out);

%rgb_out = zeros(size(r_out,1), size(r_out,2), 3);
rgb_out = cat(3, r_out, g_out, b_out);
size(rgb_out)
size(r_out)

figure;
imshow(rgb_out);


