function [correct2_img] = ring_remove(ring_img)


% 获取图像数据类型
dataType = class(ring_img);
% 转换数据类型为函数句柄
dt = str2func(dataType);
% 将图像数据转换为双精度类型
ring_img_t = double(ring_img);
% % 初始化无环图像数据为0
% no_ring_img_t = zeros(size(ring_img_t));

% hx = waitbar(0,'Please wait...','Name','Ring Artifact Reduction For CT Image');
totalIter = size(ring_img_t,3); % 总的迭代次数

% start_time = tic; % 记录开始时间

[super_ring,Nn] = superpixels3(ring_img_t,100);
for ii=1:Nn
    super_ring(super_ring==ii)=mean(ring_img_t(super_ring==ii));
end

% 默认的图像大小
default_s = [384,384];

residual_img = ring_img_t - super_ring;
if sum([size(residual_img,1) size(residual_img,2)]-default_s)~=0
    residual_img_t = imresize3(residual_img, [default_s size(residual_img,3)]);
end

% 遍历图像的每一层
ring = zeros(size(residual_img_t));
for i=1:totalIter
    r_I = residual_img_t(:,:,i);

    % 如果图像大小不等于默认大小，进行图像大小调整

    % tic
    [polar_r_I, pol_s] = im_cart2pol(r_I,round(default_s),[],'cubic');
    win_width=1600;
win_lev=200;
% figure, imshow(polar_r_I,[win_lev-(win_width./2) win_lev+(win_width./2)]);

    polar_r_I = repmat(mean(polar_r_I,2),[1 size(polar_r_I,2)]);


    ring(:,:,i) = im_pol2cart(polar_r_I, pol_s, size(r_I));



    % toc
    % figure,imshow(ring_img_t(:,:,i),[]);
    % figure,imshow(ring_img_t(:,:,i)-imresize(ring,[size(ring_img_t,1),size(ring_img_t,2)]),[]);
end
if sum([size(residual_img,1) size(residual_img,2)]-default_s)~=0
    ring = imresize3(ring, size(ring_img_t));
end

correct1_img = ring_img_t-ring;

if sum([size(correct1_img,1) size(correct1_img,2)]-default_s)~=0
    correct1_img_t = imresize3(correct1_img, [default_s size(correct1_img,3)]);
end

% figure,imshow3Dfull(correct1_img,[]);
% figure,imshow3Dfull(ring_img_t,[0 1000]);
% 遍历图像的每一层
ring2 = zeros(size(residual_img_t));
for i=1:totalIter
    % 获取当前层的图像数据
    r_I = correct1_img_t(:,:,i);

    std_r = std(r_I(:));
    mean_r = mean(r_I(:));
    rtv_I = tsmooth((r_I-(mean_r-2*std_r))/(4*std_r),0.006);
    rtv_I = 4*rtv_I*std_r+mean_r-2*std_r;



    [residul_polar, pol_s] = im_cart2pol(r_I - rtv_I,size(r_I)*2,[],'cubic');

    % figure,imshow(residul_polar,[]);
    hsize=[1 40];
    h = fspecial('average',hsize);
    residual = [residul_polar,residul_polar,residul_polar];
    residual = imfilter(residual, h);
    residual = residual(:,size(residual,2)/3+1:2*size(residual,2)/3);


    ring2(:,:,i) = im_pol2cart(residual, pol_s, size(r_I));
    

    % 调整图像的亮度

    % elapsed_time = toc(start_time); % 计算已经过去的时间
    % remaining_time = (totalIter - i) * (elapsed_time / i); % 估计剩余时间
    %
    % waitbar(i/totalIter, hx, sprintf('Processing... Time remaining: %.2f seconds', remaining_time)); % 更新进度条和剩余时间
end
if sum([size(residual_img,1) size(residual_img,2)]-default_s)~=0
    ring2 = imresize3(ring2, size(ring_img_t));
end
correct2_img = correct1_img-ring2;
        win_width=1600;
win_lev=200;
% figure, imshow(correct2_img(:,:,1),[win_lev-(win_width./2) win_lev+(win_width./2)]);


% 将无环图像写入nii文件
correct2_img = dt(correct2_img);
% figure,imshow3Dfull(correct2_img,[0 1000]);
% figure,imshow3Dfull(single(ring_img_t)-single(correct2_img),[]);
% close(hx) % 关闭进度条
end


function S = tsmooth(I,lambda)
sigma=3.0;
sharpness = 0.02;
maxIter=3;
x = I;
sigma_iter = sigma;
lambda = lambda/2.0;
dec=2.0;
for iter = 1:maxIter
    [wx, wy] = computeTextureWeights(x, sigma_iter, sharpness);
    x = solveLinearEquation(I, wx, wy, lambda);
    sigma_iter = sigma_iter/dec;
    if sigma_iter < 0.5
        sigma_iter = 0.5;
    end
end
S = x;
end

function [retx, rety] = computeTextureWeights(fin, sigma,sharpness)

vareps_s = sharpness;
vareps = 0.001;

fx = diff(fin,1,2);
fx = padarray(fx, [0 1 0], 'post');
fy = diff(fin,1,1);
fy = padarray(fy, [1 0 0], 'post');

wto = max(sum(sqrt(fx.^2+fy.^2),3)/size(fin,3),vareps_s).^(-1);

fbin = lpfilter(fin, sigma);
gfx = diff(fbin,1,2);
gfx = padarray(gfx, [0 1], 'post');
gfy = diff(fbin,1,1);
gfy = padarray(gfy, [1 0], 'post');

wtbx = max(sum(abs(gfx),3)/size(fin,3),vareps).^(-1);
wtby = max(sum(abs(gfy),3)/size(fin,3),vareps).^(-1);

retx = wtbx.*wto;
rety = wtby.*wto;

retx(:,end) = 0;
rety(end,:) = 0;

end

function ret = conv2_sep(im, sigma)
ksize = bitor(round(5*sigma),1);
g = fspecial('gaussian', [1,ksize], sigma);
ret = conv2(im,g,'same');
ret = conv2(ret,g','same');
end

function FBImg = lpfilter(FImg, sigma)
FBImg = FImg;
for ic = 1:size(FBImg,3)
    FBImg(:,:,ic) = conv2_sep(FImg(:,:,ic), sigma);
end
end

function OUT = solveLinearEquation(IN, wx, wy, lambda)
[r,c,ch] = size(IN);
k = r*c;
dx = -lambda*wx(:);
dy = -lambda*wy(:);
B=zeros(size(dx,1),2);
B(:,1) = dx;
B(:,2) = dy;
d = [-r,-1];
A = spdiags(B,d,k,k);
e = dx;
w = padarray(dx, r, 'pre'); w = w(1:end-r);
s = dy;
n = padarray(dy, 1, 'pre'); n = n(1:end-1);
D = 1-(e+w+s+n);
A = A + A' + spdiags(D, 0, k, k);

% L = my_ichol(A);
L = ichol(A,struct('michol','on'));
% L = ichol(A);
OUT = IN;
for ii=1:ch
    tin = IN(:,:,ii);
    [tout, ~] = pcg(A, tin(:),0.1,100, L, L');
    OUT(:,:,ii) = reshape(tout, r, c);
end

end

% inpaintn函数是用于图像修复的，它使用了n维离散余弦变换(DCT)和反离散余弦变换(IDCT)来进行图像的恢复。
% x是输入的有缺失的图像，n是迭代次数，y0是初始猜测的图像，m是用于Lambda计算的参数。

function y = inpaintn(x,n,y0,m)
x = double(x);
if nargin==1 || isempty(n), n = 100; end  % 如果没有输入n或者n为空，则默认迭代次数为100

% 获取x的大小和维度
sizx = size(x);
d = ndims(x);
Lambda = zeros(sizx);

% 计算Lambda的值
for i = 1:d
    siz0 = ones(1,d);
    siz0(i) = sizx(i);
    Lambda = bsxfun(@plus,Lambda,...
        cos(pi*(reshape(1:sizx(i),siz0)-1)/sizx(i)));
end
Lambda = 2*(d-Lambda);

% 初始条件
W = isfinite(x);  % 创建一个逻辑数组，其中非NaN元素为真，NaN元素为假
if nargin==3 && ~isempty(y0)
    y = y0;
    s0 = 3; % 注意: s = 10^s0
else
    if any(~W(:))
        [y,s0] = InitialGuess(x,isfinite(x));
    else
        y = x;
        return
    end
end
x(~W) = 0;  % 将x中的NaN元素设置为0

if isempty(n) || n<=0, n = 100; end  % 如果n为空或者n<=0，则默认迭代次数为100

s = logspace(s0,-6,n);  % 创建一个对数间隔的向量s

RF = 2; % relaxation factor

if nargin<4 || isempty(m), m = 2; end
Lambda = Lambda.^m;

% h = waitbar(0,'Inpainting...');
for i = 1:n
    Gamma = 1./(1+s(i)*Lambda);
    y = RF*idctn(Gamma.*dctn(W.*(x-y)+y)) + (1-RF)*y;  % 这是主要修复图像的步骤，其中使用了DCT和IDCT
    % waitbar(i/n,h)
end
% close(h)

y(W) = x(W);

end

% Initial Guess函数是用于产生一个初始猜测图像的。
% y是输入的有缺失的图像，I是一个逻辑数组，其中非NaN元素为真，NaN元素为假。
function [z,s0] = InitialGuess(y,I)

if license('test','image_toolbox')
    %-- 最近邻插值
    [~,L] = bwdist(I);
    z = y;
    z(~I) = y(L(~I));
    s0 = 3; % 注意: s = 10^s0
else
    warning('MATLAB:inpaintn:InitialGuess',...
        ['BWDIST (Image Processing Toolbox) does not exist. ',...
        'The initial guess may not be optimal; additional',...
        ' iterations can thus be required to ensure complete',...
        ' convergence. Increase N value if necessary.'])
    z = y;
    z(~I) = mean(y(I));
    s0 = 6; % 注意: s = 10^s0
end

end

% dctn函数是用于计算n维离散余弦变换的。
% y是输入的需要进行变换的数据。
function y = dctn(y)

y = double(y);
sizy = size(y);
y = squeeze(y);
dimy = ndims(y);

if isvector(y)
    dimy = 1;
    if size(y,1)==1, y = y.'; end
end

w = cell(1,dimy);
for dim = 1:dimy
    n = (dimy==1)*numel(y) + (dimy>1)*sizy(dim);
    w{dim} = exp(1i*(0:n-1)'*pi/2/n);
end

if ~isreal(y)
    y = complex(dctn(real(y)),dctn(imag(y)));
else
    for dim = 1:dimy
        siz = size(y);
        n = siz(1);
        y = y([1:2:n 2*floor(n/2):-2:2],:);
        y = reshape(y,n,[]);
        y = y*sqrt(2*n);
        y = ifft(y,[],1);
        y = bsxfun(@times,y,w{dim});
        y = real(y);
        y(1,:) = y(1,:)/sqrt(2);
        y = reshape(y,siz);
        y = shiftdim(y,1);
    end
end

y = reshape(y,sizy);

end

%% IDCTN
function y = idctn(y)  % 定义idctn函数，输入是待处理的矩阵y

y = double(y);  % 将y转换为double型
sizy = size(y);  % 获取y的大小
y = squeeze(y);  % 去除y中的单维条目
dimy = ndims(y);  % 获取y的维度数

% 如果Y是向量，则需要进行一些修改
if isvector(y)  % 判断y是否为向量
    dimy = 1;
    if size(y,1)==1  % 如果y是行向量
        y = y.';  % 将y转换为列向量
    end
end

% 定义权重向量
w = cell(1,dimy);  % 初始化权重向量
for dim = 1:dimy  % 遍历每一个维度
    n = (dimy==1)*numel(y) + (dimy>1)*sizy(dim);  % 计算n的值
    w{dim} = exp(1i*(0:n-1)'*pi/2/n);  % 计算权重向量
end

% --- IDCT算法 ---
if ~isreal(y)  % 如果y包含复数
    y = complex(idctn(real(y)),idctn(imag(y)));  % 对y的实部和虚部分别进行IDCT
else
    for dim = 1:dimy  % 遍历每一个维度
        siz = size(y);  % 获取y的大小
        n = siz(1);  % 获取y的行数
        y = reshape(y,n,[]);  % 将y重新整形为n行的矩阵
        y = bsxfun(@times,y,w{dim});  % 将y与权重向量进行点乘
        y(1,:) = y(1,:)/sqrt(2);  % 对y的第一行进行操作
        y = ifft(y,[],1);  % 对y进行傅里叶反变换
        y = real(y*sqrt(2*n));  % 对y进行操作
        I = (1:n)*0.5+0.5;  % 定义索引向量I
        I(2:2:end) = n-I(1:2:end-1)+1;  % 对索引向量I进行操作
        y = y(I,:);  % 重新排序y
        y = reshape(y,siz);  % 将y重新整形为原来的大小
        y = shiftdim(y,1);  % 将y的维度顺序进行移位
    end
end

y = reshape(y,sizy);  % 将y重新整形为原来的大小

end

% 输入：极坐标图像im_pol，极坐标图像大小sz_pol，笛卡尔坐标图像大小sz_cart
function im_cart = im_pol2cart(im_pol, sz_pol, sz_cart)
% 确定笛卡尔坐标的原点
origin = sz_cart ./ 2;

origin_c = origin;

% 计算最大半径和最大角度
[r_max_c, t_max_c] = compute_max_c(origin_c, sz_cart);
x_basis_i = 1 : sz_cart(2);
x_basis_c = (x_basis_i - origin_c(2)) ./ r_max_c;
y_basis_i = 1 : sz_cart(1);
y_basis_c = (y_basis_i - origin_c(1)) ./ r_max_c;
[x_c, y_c] = meshgrid(x_basis_c, y_basis_c);

t_basis_c = 1 : sz_pol(2);
r_basis_c = 1 : sz_pol(1);

% 创建网格
[r_c, t_c] = meshgrid(r_basis_c, t_basis_c);
r_c = imresize(r_c,[size(im_pol,2) size(im_pol,1)]);
t_c = imresize(t_c,[size(im_pol,2) size(im_pol,1)]);

% 计算半径和角度的转换因子
[r_i_to_c, t_i_to_c] = compute_i_to_c(r_max_c, t_max_c);
r_query_i = sqrt(x_c.^2 + y_c.^2);
r_query_c = r_query_i ./ r_i_to_c;
t_query_i = atan2(y_c, x_c) + pi;
t_query_c = t_query_i ./ t_i_to_c;

% 使用插值方法将极坐标图像转换为笛卡尔坐标图像
im_cart = griddata(r_c, t_c, double(im_pol)', r_query_c, t_query_c);
im_cart = fix_output_type(im_pol, im_cart, 0.5);
im_cart = imrotate(im_cart,180);
im_cart(round(size(im_cart,1)/2)-5:round(size(im_cart,1)/2)+5,round(size(im_cart,1)/2)-5:end) =...
    inpaintn(im_cart(round(size(im_cart,1)/2)-5:round(size(im_cart,1)/2)+5,round(size(im_cart,1)/2)-5:end),10);
% 处理异常值
im_cart(isnan(im_cart))=0;
im_cart(isinf(im_cart))=0;
im_cart = imtranslate(im_cart,[-1 -1]);
end

% 将笛卡尔坐标系下的图像转换为极坐标系下的图像
function [im_pol,sz_pol] = im_cart2pol(im_cart,polar_size, origin, interp_method)
im_max = max(single(im_cart(:))); % 计算输入图像的最大值
im_min = min(single(im_cart(:))); % 计算输入图像的最小值
if nargin < 3 % 如果没有指定原点位置
    origin = []; % 将原点位置设为空
end
if nargin < 4 % 如果没有指定插值方法
    interp_method = "makima"; % 使用默认的插值方法
end
sz_cart = size(im_cart); % 获取输入图像的尺寸
if isempty(origin) % 如果没有指定原点位置
    origin = sz_cart ./ 2; % 将原点位置设为图像的中心
end
interp_method = string(interp_method); % 将插值方法转换为字符串形式
origin_c = origin; % 获取原点位置
x_basis_c = 1 : sz_cart(2); % 定义x坐标基
y_basis_c = 1 : sz_cart(1); % 定义y坐标基
[x_c, y_c] = meshgrid(x_basis_c, y_basis_c); % 创建笛卡尔坐标网格
[r_max_c, t_max_c] = compute_max_c(origin_c, sz_cart); % 计算最大半径和最大角度
sz_pol = ceil([r_max_c, t_max_c]); % 计算极坐标图像的尺寸
[r_i_to_c, t_i_to_c] = compute_i_to_c(r_max_c, t_max_c); % 计算极坐标系到笛卡尔坐标系的转换系数
t_basis_i = 1 : sz_pol(2); % 定义极角基
t_basis_c = t_basis_i .* t_i_to_c; % 计算极角基在笛卡尔坐标系下的值
r_basis_i = 1 : sz_pol(1); % 定义半径基
r_basis_c = r_basis_i .* r_i_to_c; % 计算半径基在笛卡尔坐标系下的值
[t_c, r_c] = meshgrid(t_basis_c, r_basis_c); % 创建极坐标网格
x_query_c = r_c .* cos(t_c) .* r_max_c + origin_c(2); % 计算查询点在笛卡尔坐标系下的x坐标
y_query_c = r_c .* sin(t_c) .* r_max_c + origin_c(1); % 计算查询点在笛卡尔坐标系下的y坐标
x_query_c = imresize(x_query_c,polar_size); % 将查询点的x坐标调整到目标尺寸
y_query_c = imresize(y_query_c,polar_size); % 将查询点的y坐标调整到目标尺寸
im_pol = interp2(x_c, y_c, double(im_cart), x_query_c, y_query_c, interp_method); % 对图像进行插值
im_pol = fix_output_type(im_cart, im_pol, 0.5); % 调整输出图像的类型
im_pol(im_pol>im_max)=im_max; % 将超过最大值的像素设置为最大值
im_pol(im_pol<im_min)=im_min; % 将低于最小值的像素设置为最小值
end

% 根据输入图像的类型调整输出图像的类型
function im_out = fix_output_type(im_in, im_out, logical_threshold)
t = string(class(im_in));
if t == "logical" % 如果输入是逻辑型
    im_out = im_out > logical_threshold; % 对输出进行二值化
else
    im_out = cast(im_out, t); % 否则，将输出转换为与输入相同的类型
end
end

% 计算最大半径和最大角度
function [r_max_c, t_max_c] = compute_max_c(origin_c, sz_cart)
corners_c = [... % 定义四个角点
    1 1; ...
    sz_cart(1) 1; ...
    1 sz_cart(2); ...
    sz_cart ...
    ];
sums = sum((origin_c - corners_c) .^ 2, 2); % 计算原点到四个角点的距离的平方
r_max_c = sqrt(max(sums)); % 计算最大半径
t_max_c = 2 * pi * r_max_c; % 计算最大角度
end

% 计算极坐标系到笛卡尔坐标系的转换系数
function [r_i_to_c, t_i_to_c] = compute_i_to_c(r_max_c, t_max_c)
r_i_to_c = 1 ./ r_max_c; % 计算半径转换系数
t_i_to_c = 2 .* pi ./ t_max_c; % 计算角度转换系数
end



