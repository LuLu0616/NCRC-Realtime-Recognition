function PtnCell = Feature2PtnCell(working_Feature,imsize2)

fprintf('\nFeature--> PtnCell Start ...\n');
PtnNums = 0;
ind = 0;
size_x = imsize2(1);
size_y = imsize2(2);

classes = dir(working_Feature);
classes(1:2) = [];
for class_i = 1:length(classes) 
    files = dir([working_Feature, '\', classes(class_i).name]);
    files(1:2) = [];
    
    fileNums = length(files);
    PtnCell(PtnNums+1: PtnNums + fileNums) = cell(1, fileNums);
    PtnNums = PtnNums + fileNums;
    
    for file_i = 1:fileNums
        load([working_Feature, '/', classes(class_i).name, '/', files(file_i).name]);
        Fout  = TD;
        
        if length(Fout.ts)<20 
            continue;
        end
        
        ind = ind + 1;
        AllVec = zeros(length(Fout.ts), 1);
        AllAddr = zeros(length(Fout.ts), 1);
        
        for i = 1:length(Fout.ts)
            x = Fout.x(i); y = Fout.y(i); t = Fout.ts(i); p = Fout.p(i);
            AllVec(i) = t;
            AllAddr(i) = (p-1)*size_x*size_y+(x-1)*size_y+y;
        end
        PtnCell{ind}.AllVec  = AllVec/1e3;
        PtnCell{ind}.AllAddr = AllAddr;
        PtnCell{ind}.Time_Chnl_Lbl = class_i-1;
    end
    PtnCell = PtnCell(1:ind);
end

