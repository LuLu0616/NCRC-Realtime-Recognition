function TD = D_pre_process(TD,isdenoise)


TD.ts = TD.ts - TD.ts(1);
if isdenoise
    TD = NewDenoise(TD);
end

if (min(TD.x) == 0 || min(TD.y) == 0)
    
    TD.x = TD.x + 1;
    TD.y = TD.y + 1;
end
if size(TD.ts,1)~=1
    TD.ts = TD.ts';
    TD.x = TD.x';
    TD.y = TD.y';
    TD.p = TD.p';
end
end
function matrix = NewDenoise(TD)

tau1 = 200e3;
radiu = 1;
th = 3;
dt = 1; 

t = 0:dt:10*tau1;
lut1 = exp(-t/tau1);  % Lookup Table
V0 = 1/max(lut1);


if (min(TD.x) == 0 || min(TD.y) == 0)
    TD.x = TD.x + 1;
    TD.y = TD.y + 1;
end
matrix = TD;


x = matrix.x + radiu;
y = matrix.y + radiu;
ts = matrix.ts / dt;

% 图像大小
size1 = max(x) + radiu;
size2 = max(y) + radiu;

num = length(ts);
addr = false(1, num);
t_last = zeros(size1, size2);
K1 = zeros(size1, size2);

filter = getfilter(2*radiu+1, 1.68);
for i = 1:num
    ti = ts(i);
    xi = x(i);
    yi = y(i);
    
    % 防止越界
    x1 = xi-radiu; x2 = xi+radiu;
    y1 = yi-radiu; y2 = yi+radiu;
    delta_t = ti-t_last(x1:x2, y1:y2);
    t_last(x1:x2, y1:y2) = ti;

    Sc1 = getSc(delta_t, lut1, dt);
    Sc1 = V0.*Sc1;
    K1(x1:x2, y1:y2) = Sc1.*K1(x1:x2, y1:y2);
    K1(x1:x2, y1:y2) = K1(x1:x2, y1:y2) + filter;
    
    if K1(xi, yi) > th
        addr(i) = true;
    end
end

matrix.x = matrix.x(addr);
matrix.y = matrix.y(addr);
matrix.ts = matrix.ts(addr);
matrix.p = matrix.p(addr); 

end

function Sc = getSc(delta_t,lut, dt)
lut_addr = round(delta_t/dt) + 1;
add = lut_addr > length(lut);
lut_addr(add) = length(lut);
Sc = lut(lut_addr);
end

function filter = getfilter(size, sigma)
cx = ceil(size/2);
cy = ceil(size/2);
[x, y] = meshgrid(-floor((size-1)/2):floor(size/2),-floor((size-1)/2):floor(size/2));
filter = 1/(2*pi*sigma^2)*exp(-(x.*x+y.*y)/(2*sigma^2));
filter = filter * 10;
filter(cx, cy) = 0;
end