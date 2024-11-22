% 清除所有变量
clear
% 关闭所有图窗
close all
% 清屏
clc

% 设置文件名
filename = 'ring_img.nii.gz';
% 显示窗设置
win_width=1600;
win_lev=200;
% 读取nii.gz格式的图像文件
ring_img = niftiread(filename);
% ring_img = repmat(ring_img,[1 1 200]);
% 获取nii.gz格式的图像信息

[no_ring_img_t] = ring_remove(ring_img);

% info_i = niftiinfo(filename);
% niftiwrite(no_ring_img_t,'no_ring.nii', info_i, 'Compressed',true);
figure;
artifact = double(ring_img)-double(no_ring_img_t);
subplot(1,3,1);imshow(ring_img(:,:,round(size(ring_img,3)./2)),[win_lev-(win_width./2) win_lev+(win_width./2)]);
title(strcat('Uncorrected image:',num2str(round(size(ring_img,3)./2)),'th slice'));
subplot(1,3,2); imshow(no_ring_img_t(:,:,round(size(ring_img,3)./2)),[win_lev-(win_width./2) win_lev+(win_width./2)]);
title(strcat('Corrected image:',num2str(round(size(ring_img,3)./2)),'th slice'));
subplot(1,3,3); imshow(artifact(:,:,round(size(ring_img,3)./2)),[]);
title(strcat('ring artifacts image:',num2str(round(size(ring_img,3)./2)),'th slice'));
