% �� Excel �ļ���ȡ����
data = readtable('�Ƚ�.xlsx');

% ��ȡΨһ�ļܹ����ͺͶ˿���
architectures = unique(data.approach);
portCounts = unique(data.duankou);

% ������ɫӳ�䣬Ϊÿ�ּܹ�����һ����ɫ
colors = containers.Map();
colors('hybrid') = 'blue';
colors('rain-only') = 'green';
colors('spine-leaf') = 'orange';
colors('clos') = 'red';

% ����ͼ
figure('Position', [100, 100, 1200, 800]); % ����ͼ����С��λ��

% �������� Y ��
[ax, h1, h2] = plotyy(1, 1, 1, 1);
ax(1).NextPlot = 'add'; % �����������ϵ�����״ͼ
ax(2).NextPlot = 'add'; % �����ڴ����ϵ�������ͼ

% �����������ǩ
xlabel(ax(1), '�˿���', 'FontSize', 14);
ylabel(ax(1), 'GPU ���� (Ntotal)', 'FontSize', 14);
ylabel(ax(2), 'EPS �ɱ� (n_EPS)', 'FontSize', 14);
title('��ͬ�ܹ��ڲ�ͬ�˿����µ� GPU ������ EPS �ɱ�', 'FontSize', 16);

% �رյڶ���������ı���
set(ax(2), 'color', 'none');

% ��״ͼƫ�����Ŀ��
barWidth = 4;

% ѭ������ÿ�ּܹ�������
for i = 1:length(architectures)
    arch = architectures{i};

    % ��ȡ��ǰ�ܹ�������
    archData = data(strcmp(data.approach, arch), :);

    % �������ÿ���˿�������״ͼ��λ��ƫ��
    xOffset = portCounts + barWidth * (i - (length(architectures) + 1) / 2);

    % ������״ͼ����ʾGPU���� (���Y��)
    bar(ax(1), xOffset, archData.Ntotal, barWidth, ...
        'FaceColor', get_color(colors(arch)), 'EdgeColor', 'none', 'DisplayName', [arch ' (GPU)'], 'FaceAlpha', 0.7);

    % ��������ͼ����ʾEPS�ɱ� (�Ҳ�Y��)
    plot(ax(2), portCounts, archData.n_EPS, 'Color', get_color(colors(arch)), ...
        'Marker', 'o', 'LineStyle', '-', 'DisplayName', [arch ' (EPS)'], 'LineWidth', 1.5);
end

% ���������᷶Χ
xlim(ax(1), [min(portCounts) - barWidth, max(portCounts) + barWidth]);

% ����������̶�
set(ax(1), 'XTick', portCounts);

% ���ͼ��
legend('Location', 'NorthWest', 'FontSize', 10);

% ���ڶ���Y������ұ�
ax(2).YAxisLocation = 'right';

% �����������ص�
linkaxes(ax, 'x');

% �������ֱ����ص�
tightInset = get(gca, 'TightInset');
pos = get(gca, 'Position');
set(gca, 'Position', [pos(1) + tightInset(1) pos(2) + tightInset(2) ...
    (1-tightInset(1)-tightInset(3)) (1-tightInset(2)-tightInset(4))]);

% ��������������ɫ����ת��ΪMATLAB���õ���ɫ��ʽ
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
            color_value = 'black'; % Ĭ����ɫ
    end
end

