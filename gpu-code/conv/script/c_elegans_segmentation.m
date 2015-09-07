function y = c_elegans_segmentation(in_file, out_dir)

color_img = imread(in_file);

gray_img = rgb2gray(color_img);

%value参数指定了区域的大小
mser_regions = detectMSERFeatures(gray_img, 'RegionAreaRange', [20 500000]);
mser_regions_pixels = vertcat(cell2mat(mser_regions.PixelList));

% figure;
% imshow(gray_img);
% hold on;
% plot(mser_regions, 'showPixelList', true, 'showEllipses',false);
% title('mser regions');

%显示被扣出来的区域
%初始化为0值
mser_mask = false(size(gray_img));
%元素下标转换为该元素在数组中对应的索引值
ind = sub2ind(size(mser_mask), mser_regions_pixels(:,2), mser_regions_pixels(:,1));
mser_mask(ind) = true;
% figure;
% imshow(mser_mask);

%找到大区域的bounding box
low_area = 150;
high_area = 500000;
%找到二值图像中的连通区域
conn_comp = bwconncomp(mser_mask);

stats = regionprops(conn_comp, 'BoundingBox', 'Area');

boxes = round(vertcat(stats(vertcat(stats.Area) > low_area & vertcat(stats.Area) < high_area) .BoundingBox));
for i = 1:size(boxes,1)
    crop_img = imcrop(gray_img, boxes(i,:));
    imwrite(crop_img, strcat(out_dir, int2str(i), '.png'));
end
y = 1;






















