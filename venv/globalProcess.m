clc, close all;
%�ر����д��ļ��ͱ���
[filename,pathname]=uigetfile('*.*','ѡ��ͼ���ļ�');
tic;
preImage=imread(strcat(pathname,filename));
figure;
imshow(preImage)
title('����ǰͼ��')
%ѡ��ͼ�񣬲��Ҽ�¼ʱ��
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
title('����ǿ�ȷ���ͼ');
%�����˲������ҹ��˹��շ���
meanG = mean(aveGaus(:));
gamma = power(1/2,((meanG - aveGaus)/meanG));
outV = power(v,gamma);
figure;
imshow(outV,[]);
title('�������շ���');
%��ά٤����������շ���
afterImage = hsv2rgb(h,s,outV);
figure;
imshow(afterImage);
title('������ͼ��');
imwrite(afterImage,'�����ͼ��.jpg');
toc;

