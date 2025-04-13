% 从 Excel 文件读取数据
data = readtable('比较.xlsx');

% 获取唯一的架构类型和端口数
architectures = unique(data.approach);
portCounts = unique(data.duankou);

% 定义颜色映射，为每种架构分配一个颜色
colors = containers.Map();
colors('hybrid') = 'blue';
colors('rain-only') = 'green';
colors('spine-leaf') = 'orange';
colors('clos') = 'red';

% 创建图
figure('Position', [100, 100, 1200, 800]); % 设置图窗大小和位置

% 创建两个 Y 轴
[ax, h1, h2] = plotyy(1, 1, 1, 1);
ax(1).NextPlot = 'add'; % 允许在主轴上叠加柱状图
ax(2).NextPlot = 'add'; % 允许在次轴上叠加折线图

% 设置坐标轴标签
xlabel(ax(1), '端口数', 'FontSize', 14);
ylabel(ax(1), 'GPU 数量 (Ntotal)', 'FontSize', 14);
ylabel(ax(2), 'EPS 成本 (n_EPS)', 'FontSize', 14);
title('不同架构在不同端口数下的 GPU 数量和 EPS 成本', 'FontSize', 16);

% 关闭第二个坐标轴的背景
set(ax(2), 'color', 'none');

% 柱状图偏移量的宽度
barWidth = 4;

% 循环绘制每种架构的数据
for i = 1:length(architectures)
    arch = architectures{i};

    % 提取当前架构的数据
    archData = data(strcmp(data.approach, arch), :);

    % 计算对于每个端口数的柱状图的位置偏移
    xOffset = portCounts + barWidth * (i - (length(architectures) + 1) / 2);

    % 绘制柱状图，表示GPU数量 (左侧Y轴)
    bar(ax(1), xOffset, archData.Ntotal, barWidth, ...
        'FaceColor', get_color(colors(arch)), 'EdgeColor', 'none', 'DisplayName', [arch ' (GPU)'], 'FaceAlpha', 0.7);

    % 绘制折线图，表示EPS成本 (右侧Y轴)
    plot(ax(2), portCounts, archData.n_EPS, 'Color', get_color(colors(arch)), ...
        'Marker', 'o', 'LineStyle', '-', 'DisplayName', [arch ' (EPS)'], 'LineWidth', 1.5);
end

% 设置坐标轴范围
xlim(ax(1), [min(portCounts) - barWidth, max(portCounts) + barWidth]);

% 设置坐标轴刻度
set(ax(1), 'XTick', portCounts);

% 添加图例
legend('Location', 'NorthWest', 'FontSize', 10);

% 将第二个Y轴放在右边
ax(2).YAxisLocation = 'right';

% 避免坐标轴重叠
linkaxes(ax, 'x');

% 调整布局避免重叠
tightInset = get(gca, 'TightInset');
pos = get(gca, 'Position');
set(gca, 'Position', [pos(1) + tightInset(1) pos(2) + tightInset(2) ...
    (1-tightInset(1)-tightInset(3)) (1-tightInset(2)-tightInset(4))]);

% 辅助函数：将颜色名称转换为MATLAB可用的颜色格式
function color_value = get_color(color_name)
    switch color_name
        case 'red'
            color_value = 'red';
        case 'blue'
            color_value = 'blue';
        case 'green'
            color_value = 'green';
        case 'orange'
            color_value = 'orange';
        otherwise
            color_value = 'black'; % 默认颜色
    end
end

