function [PtnCellTrn, PtnCellTst, indTrn, indTst,maxT] = RandSplit(PtnCell,ratio_Trn,nGroup)

fprintf('\nRandomly splitting the dataset-->PtnCell_raw... ...\n');
N = length(PtnCell);%所有样本总数
Labels = zeros(1,N);%行向量的标签
maxT = -inf;
for i = 1:N %遍历所有样本，将所有标签存到Labels中
    B = PtnCell{i}.Time_Chnl_Lbl;
    Labels(i) = B;
    AllVec = PtnCell{i}.AllVec;%依次取出每个样本的输出脉冲时间
    maxT = max(maxT,max(AllVec(:)));

end

indTrn = zeros(1, round(N*ratio_Trn)); %训练样本的索引——行向量
indTst = zeros(1, N-round(N*ratio_Trn)); %测试样本的索引——行向量
count1 = 0;
count2 = 0;
for i = 1:nGroup
    ind = find(Labels==(i-1));%分类索引 在所有样本中的索引
    n0 = length(ind); %每类样本数

    n1 = round(n0*ratio_Trn);%每类中的训练样本数
    n2 = n0-n1;%每类中的测试样本数
    
    ind2 = randperm(n0);
    indTrn(count1+1:count1+n1) = ind(ind2(1:n1));%乱序全部训练集索引（在整体中的索引）
    indTst(count2+1:count2+n2) = ind(ind2(n1+1:n0));%乱序全部测试集索引（在整体中的索引）
    count1 = count1+n1;
    count2 = count2+n2;
end

indTrn = indTrn(1:count1);
indTst = indTst(1:count2);

numTrn = length(indTrn);
numTst = length(indTst);

indTrn = indTrn(randperm(numTrn));
indTst = indTst(randperm(numTst));

PtnCellTrn = PtnCell(indTrn);%一行cell
PtnCellTst = PtnCell(indTst);%一行cell

end
    
    
    


