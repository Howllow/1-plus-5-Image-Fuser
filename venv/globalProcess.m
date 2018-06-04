clc, close all;
%关闭所有打开文件和变量
[filename,pathname]=uigetfile('*.*','选择图像文件');
tic;
preImage=imread(strcat(pathname,filename));
figure;
imshow(preImage)
title('处理前图像')
%选择图像，并且记录时间
[h,s,v] = rgb2hsv(preImage);
%ImageSize = fix((size(preImage,1)+size(preImage,2))/2);
c1 = 15;
c2 = 80;
c3 = 250;
filter1 = fspecial('gaussian',fix(5*c1),c1);
filter2 = fspecial('gaussian',fix(5*c2),c2);
filter3 = fspecial('gaussian',fix(5*c3),c3);
gaussian1 = imfilter(v,filter1,'replicate');
gaussian2 = imfilter(v,filter2,'replicate');
gaussian3 = imfilter(v,filter3,'replicate');
aveGaus = (gaussian1+gaussian2+gaussian3)/3;
figure;
imshow(aveGaus);
title('光照强度分量图');
%创造滤波器并且过滤光照分量
meanG = mean(aveGaus(:));
gamma = power(1/2,((meanG - aveGaus)/meanG));
outV = power(v,gamma);
figure;
imshow(outV,[]);
title('处理后光照分量');
%二维伽马函数处理光照分量
afterImage = hsv2rgb(h,s,outV);
figure;
imshow(afterImage);
title('处理后的图像');
imwrite(afterImage,'处理后图像.jpg');
toc;

