function feat = Sim_timesurface(TD,layer,ispool,pooling_extent,refractory_period)
global params; 
if layer == 1
    param = params.layer1;
elseif layer == 2

    param = params.layer2;
end

radius = param.radius;
tau = param.tau;
num_feature = param.num_feature;
C = param.C; 

image_size = param.image_size;
size_x = image_size(1);
size_y = image_size(2);
dt = 10;
t = -0:dt:tau;
lut = exp(-t/tau);
t_last  = ones(size_y+2*radius,size_x+2*radius,max(TD.p)) * (-tau);
feat = TD;
for i = 1:length(TD.ts)
    xi  = TD.x(i) + radius;
    yi  = TD.y(i) + radius;
    ti  = TD.ts(i);
    pi  = TD.p(i);
    
    S = zeros(2*radius+1, 2*radius+1);
    for rx = -radius:radius
        for ry = -radius:radius
     
            delta_t = ti - t_last(yi+ry,xi+rx,pi);
            if delta_t < tau
                lut_addr = round(delta_t/dt) + 1;
                S(radius+ry+1,radius+rx+1) = lut(lut_addr);
            end
        end
    end
    
    t_last(yi,xi,pi) = ti;
        
    if sum(sum(S~=0))==0
        output_index = 0;
    else
        
        min_distance = inf;
        for index = 1:num_feature
            temp = C(:,:,index)-S;
            distance = (sqrt(sum(temp(:).^2)));
            if distance < min_distance
                min_distance = distance;
                output_index  = index;
            end
        end
        
    end
    
    feat.x(i)  = TD.x(i);
    feat.y(i)  = TD.y(i);
    feat.ts(i) = TD.ts(i);
    feat.p(i)  = output_index;
end
feat = RemoveNulls(feat, feat.p == 0);

if ispool == 1
    feat.x = ceil(feat.x/pooling_extent);
    feat.y = ceil(feat.y/pooling_extent);
    feat = ImplementRefraction(feat,refractory_period);
end
end

function TD = ImplementRefraction(TD, Refrac_time)
TD.ts = TD.ts + Refrac_time;
LastTime = zeros(max(TD.x), max(TD.y));
for i = 1:length(TD.ts)
    if ((TD.ts(i) - LastTime(TD.x(i), TD.y(i))) > Refrac_time)
        LastTime(TD.x(i), TD.y(i)) = TD.ts(i);
    else
        TD.ts(i) = 0;
    end
end
TD = RemoveNulls(TD, TD.ts == 0);
TD.ts = TD.ts - Refrac_time;
end
function result =  RemoveNulls(result, indices)
indices = logical(indices);
fieldnames = fields(result);
for i = 1:length(fieldnames)
    if ~strcmp(fieldnames{i}, 'meta')
        result.(fieldnames{i})(indices)  = [];
    end
end
end
