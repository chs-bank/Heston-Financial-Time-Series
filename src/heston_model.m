%% 导入电子表格中的数据

%% 设置导入选项并导入数据
opts = spreadsheetImportOptions("NumVariables", 4);

% 指定工作表和范围
opts.Sheet = "Sheet1";
opts.DataRange = "A33:D134";

% 指定列名称和类型
opts.VariableNames = ["Var1", "Var2", "SH", "SH_1"];
opts.SelectedVariableNames = ["SH", "SH_1"];
opts.VariableTypes = ["char", "char", "double", "double"];

% 指定变量属性
opts = setvaropts(opts, ["Var1", "Var2"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var1", "Var2"], "EmptyFieldRule", "auto");

% 导入数据（这里用的是相对路径导入，需要该matlab的脚本文件和excel要放置在同一个路径下即可）
stock_price = readtable("../data/stock_price.xlsx", opts, "UseExcel", false);

%% 转换为输出类型
stock_price = table2array(stock_price);

%% 清除临时变量
clear opts

%% 收集可转债标的股票的历史数据，计算开盘价和收盘价的均值𝑆
S = mean(stock_price,2);

%% 计算每日相对于前一日的对数收益率
% 计算对数收益率  
% 注意：我们跳过第一天，因为没有前一日的数据  
R = diff(log(S)); % diff 计算连续元素之间的差异  
  
% 如果需要，可以在结果前面添加一个NaN或0来表示第一天的收益率（这里使用0）  
R = [0; R];
R_ave = mean(R);%对数收益率的均值
%disp(R_ave);
%% 选择一个时间窗口W天，计算:窗口内对数收益率的均值
% 选择时间窗口W  
W = 10; % 例如，选择10天的窗口  
  
% 初始化一个向量来存储每个窗口的均值  
R_ave_w = zeros(1, length(R) - W + 1);  
  
% 遍历数据并计算每个窗口的均值  
for i = 1:(length(R) - W + 1)  
    R_ave_w(i) = mean(R(i:i+W-1));  
end 
%% 计算窗口内对数收益率的波动率
% 初始化一个向量来存储每个窗口的波动率（即标准差）  
d = zeros(1, length(R) - W + 1);  
  
% 遍历数据并计算每个窗口的波动率  
for i = 1:(length(R) - W + 1)  
    d(i) = std(R(i:i+W-1));  
end 
%% 4.计算各个参数:
O = mean(d);%O：波动率的长期均值；
d_v = std(d);%d_v ：波动率的波动率(标准差)，表示波动率 𝜈𝑡 的不确定性；
%% 计算K 𝜅：波动率回归速度，表示波动率回归到O的速率；
a = 0;
b = 0;
for i = 2:length(d)
    a = (d(i-1) - O)*(d(i)-d(i-1)) + a;
    b = b + (d(i-1)-O)^2;
end
cov = a/(length(d)-1);
var = b/(length(d)-1);
K = -cov/var;
%% 计算相关系数p
% p：𝑊𝑡(1)和𝑊𝑡(2)的相关系数，表示资产价格和其波动率之间的相关性。
A = R(1:93, :);
p = corr(A,d');
%% heston模型
v0 = O; %初始波动率
S0 = 4.87;  % 初始股票价格  
r = 0.05;  % 无风险利率  
T = 1;  % 到期时间  
M = 100;  % 时间步数  
dt = T / M;  % 时间步长  
N_paths = 20;  % 路径数量  
  
% 初始化路径数组  
vPaths = zeros(M + 1, N_paths);  
SPaths = zeros(M + 1, N_paths);  
vPaths(1,:) = v0;  
SPaths(1,:) = S0;  
  
% 设置随机种子以获得可复现的结果 这里就不设置了 
%rng(0, 'twister');  
  
% 生成随机布朗运动增量  
dW_t = sqrt(dt) * randn(M + 1, N_paths);  
dZ_t = sqrt(dt) * randn(M + 1, N_paths);  
  
% 模拟Heston模型的路径  
for t = 2:M + 1  
    for i = 1:N_paths  
        dv = K * (O - vPaths(t - 1, i)) * dt + d_v * sqrt(vPaths(t - 1, i)) * dW_t(t, i);  
        vPaths(t, i) = max(vPaths(t - 1, i) + dv, 0);  % 确保波动率不会变成负数  
        SPaths(t, i) = SPaths(t - 1, i)*(exp((r - vPaths(t, i)/2) * dt + sqrt(vPaths(t, i)) * (p * dW_t(t, i) + sqrt(1 - p^2) * dZ_t(t, i)))); 
    end  
end  
  
% 提取股票价格路径  
S_t_paths = SPaths(2:end, :);  
  
% 提取时间步长  
timeSteps = linspace(0, T, M + 1);  
timeSteps = timeSteps(2:end);  % 去掉初始的0时间点  
  
% 可视化多条股票价格时间序列  
figure;  
hold on;  
for i = 1:N_paths  
    plot(timeSteps, S_t_paths(:, i), 'LineWidth', 1.5);  
    legend(['Path ' num2str(i) ' '], 'Location', 'Best');  
end  
title('多条股票价格时间序列 - Heston 模型');  
xlabel('时间');  
ylabel('价格');  
grid on;  
hold off;







