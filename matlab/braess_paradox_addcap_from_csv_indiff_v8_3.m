function braess_paradox_addcap_from_csv_indiff_v8_3(input_csv)
% Diagnostic version with detailed output to understand why edges aren't indifferent
%
% Usage:
%  braess_paradox_addcap_from_csv_indiff_v8_debug('nodes_input_withP.csv')

if nargin<1 || isempty(input_csv)
    input_csv = 'nodes_input_withP.csv';                %8 node base
    % input_csv = 'nodes_input_withP_check_14.csv';     %10 node 
    % input_csv = 'nodes_input_withP_check_15.csv';     %10 node different topologies
    % input_csv = 'nodes_input_withP_check_16.csv';
end
if ~isfile(input_csv)
    error('Input CSV file not found: %s', input_csv);
end

%% ---------- Read CSV ----------
fid = fopen(input_csv,'r'); raw = textscan(fid, '%s', 'Delimiter', '\n'); fclose(fid);
raw = raw{1};
lines = {};
for i = 1:numel(raw)
    s = strtrim(raw{i});
    if isempty(s) || startsWith(s,'#'), continue; end
    lines{end+1} = s;
end
if isempty(lines), error('No data lines found in CSV'); end
node_ids = []; coords = []; P_spec = [];
neighbors_list = containers.Map('KeyType','double','ValueType','any');
for i = 1:numel(lines)
    parts = strtrim(strsplit(lines{i}, ','));
    parts = parts(~cellfun(@isempty, parts));
    if numel(parts) < 4, error('Line %d must have NodeID,P,X,Y', i); end
    nid = round(str2double(parts{1}));
    Pv  = str2double(parts{2});
    x   = str2double(parts{3}); y = str2double(parts{4});
    if any(isnan([nid Pv x y])), error('Invalid numeric on line %d', i); end
    node_ids(end+1) = nid;
    P_spec(end+1) = Pv;
    coords(end+1,:) = [x y];
    neighs = [];
    for k = 5:numel(parts)
        nn = str2double(parts{k});
        if ~isnan(nn), neighs(end+1) = round(nn); end
    end
    neighbors_list(nid) = unique(neighs);
end

%% ---------- Build adjacency, coords, P_vec ----------
N = max(node_ids);
A = zeros(N);
for k = 1:numel(node_ids)
    id = node_ids(k);
    neighs = neighbors_list(id);
    for j = neighs
        if j>=1 && j<=N
            A(id,j) = 1; A(j,id) = 1;
        end
    end
end
x_coords = nan(1,N); y_coords = nan(1,N);
for k = 1:numel(node_ids)
    id = node_ids(k);
    x_coords(id) = coords(k,1); y_coords(id) = coords(k,2);
end
missing = isnan(x_coords);
if any(missing)
    idx = find(missing);
    theta = linspace(0,2*pi,numel(idx)+1); theta(end) = [];
    R = 5;
    for ii = 1:numel(idx)
        x_coords(idx(ii)) = R*cos(theta(ii)); y_coords(idx(ii)) = R*sin(theta(ii));
    end
end
P_vec = zeros(N,1);
for k = 1:numel(node_ids)
    P_vec(node_ids(k)) = P_spec(k);
end

%% ---------- Base parameters & edge list ----------
P_base = 1.0;
alpha = P_base;
[r,c] = find(triu(A));
edge_list = [r c];
m = size(edge_list,1);

fprintf('\n=== NETWORK INFO ===\n');
fprintf('Number of nodes: %d\n', N);
fprintf('Number of edges: %d\n', m);
fprintf('Edge list:\n');
for e = 1:m
    fprintf('  Edge %d: %d-%d\n', e, edge_list(e,1), edge_list(e,2));
end
fprintf('====================\n\n');

%% ---------- Settings for sync tests -----------
slope_threshold = 0.2;
epsilon_A = 0;
tspan_short = [0 80];
tspan_long = [0 1000];
y0 = zeros(2*N,1);
opts = odeset('RelTol',1e-6,'AbsTol',1e-9);

%% ---------- Compute critical Kc ----------
fprintf('Computing critical Kc...\n');
K_search_range = [0.1 5.0] * P_base;
K_search_len = 100;
r_thresh = 0.5;
dphi_tol = 1e-2;
K_list = linspace(K_search_range(1), K_search_range(2), K_search_len);
Kc = NaN;

for kk = 1:length(K_list)
    Ktest = K_list(kk);
    Kmat = Ktest * A;
    [t_tmp, y_tmp] = ode45(@(t,y) second_order_ode(t,y,P_vec,Kmat,alpha), tspan_short, y0, opts);
    phi_end = y_tmp(end,1:N);
    dphi_end = y_tmp(end,N+1:2*N);
    rval = abs(mean(exp(1i*phi_end)));
    if (rval >= r_thresh) && (max(abs(dphi_end)) <= dphi_tol)
        Kc = Ktest;
        break;
    end
end

if isnan(Kc)
    warning('No Kc found in search range. Using P_base.');
    Kc = P_base;
else
    fprintf('Found Kc = %.6f (Kc/P = %.6f)\n\n', Kc, Kc / P_base);
end

%% ---------- Define K_uniform and n_values ----------
K_uniform = Kc;
K_base = K_uniform * A;
kc_over_P = Kc / P_base;
n_values = unique([0.0, 1.0, 2, 10, 100]);
n_values = sort(n_values);
fprintf('Using n_values = [%s]\n\n', num2str(n_values));

%% ---------- Simulate base ----------
fprintf('Simulating base (K=Kc)...\n');
[t0, y0sol] = ode45(@(t,y) second_order_ode(t,y,P_vec,K_base,alpha), tspan_long, y0, opts);
phi_base = y0sol(:, 1:N);
phi_pi_base = phi_base / pi;
max_abs_phi_pi_per_node_base = max(abs(phi_pi_base), [], 1);
condA_base_per_node = (max_abs_phi_pi_per_node_base <= (1 + epsilon_A));
condA_base_all = all(condA_base_per_node);

fprintf('Base steady-state phi/pi values:\n');
for node = 1:N
    fprintf('  Node %d: %.4f\n', node, phi_pi_base(end, node));
end
fprintf('\n');

if ~condA_base_all
    fprintf('BASE SYSTEM UNSTABLE! Aborting.\n');
    return;
else
    fprintf('Base system is STABLE.\n\n');
end




%% ---------- Edge sweep with detailed diagnostics ----------
fprintf('Starting edge sweep...\n');
fprintf('==================================================\n\n');

edge_sync_matrix = false(m, numel(n_values));
metrics = [];

for e = 1:m
    i1 = edge_list(e,1); i2 = edge_list(e,2);
    fprintf('>>> EDGE %d-%d (edge %d/%d) <<<\n', i1, i2, e, m);
    
    edge_sync_for_all_n = true;  % Track if this edge syncs for ALL n
    
    for ni = 1:numel(n_values)
        n = n_values(ni);
        
        if abs(n - 1.0) < 1e-6  % n=1 is base case
            % Reuse base simulation
            condA_allnodes = condA_base_all;
            kidx = find(t0 >= 45, 1, 'first');
            if isempty(kidx), kidx = length(t0); end
            if kidx == 1
                slope_at_45 = zeros(1,N);
            else
                dt_local = t0(kidx) - t0(kidx - 1);
                slope_at_45 = (phi_pi_base(kidx, :) - phi_pi_base(kidx - 1, :)) / dt_local;
            end
            condB_per_node = (abs(slope_at_45) <= slope_threshold);
            condB_allnodes = all(condB_per_node);
            is_sync = condA_allnodes && condB_allnodes;
            
            metrics_row = struct();
            metrics_row.i = i1; metrics_row.j = i2; metrics_row.n = n;
            metrics_row.max_abs_phi_pi_node = max_abs_phi_pi_per_node_base;
            metrics_row.max_abs_phi_pi = max(max_abs_phi_pi_per_node_base);
            metrics_row.condA_allnodes = condA_allnodes;
            metrics_row.slope_at_45 = slope_at_45;
            metrics_row.max_abs_slope = max(abs(slope_at_45));
            metrics_row.condB_allnodes = condB_allnodes;
            metrics_row.is_sync = is_sync;
        else
            % Run simulation
            Km = K_base;
            Km(i1,i2) = n * K_uniform;
            Km(i2,i1) = n * K_uniform;
            
            [t,y] = ode45(@(t,y) second_order_ode(t,y,P_vec,Km,alpha), tspan_long, y0, opts);
            phi = y(:,1:N);
            
            phi_pi_all = phi / pi;
            max_abs_phi_pi_per_node = max(abs(phi_pi_all), [], 1);
            condA_per_node = (max_abs_phi_pi_per_node <= (1 + epsilon_A));
            condA_allnodes = all(condA_per_node);
            
            kidx = find(t >= 45, 1, 'first');
            if isempty(kidx), kidx = length(t); end
            if kidx == 1
                slope_at_45 = zeros(1,N);
            else
                dt_local = t(kidx) - t(kidx - 1);
                slope_at_45 = (phi_pi_all(kidx, :) - phi_pi_all(kidx - 1, :)) / dt_local;
            end
            condB_per_node = (abs(slope_at_45) <= slope_threshold);
            condB_allnodes = all(condB_per_node);
            is_sync = condA_allnodes && condB_allnodes;
            
            metrics_row = struct();
            metrics_row.i = i1; metrics_row.j = i2; metrics_row.n = n;
            metrics_row.max_abs_phi_pi_node = max_abs_phi_pi_per_node;
            metrics_row.max_abs_phi_pi = max(max_abs_phi_pi_per_node);
            metrics_row.condA_allnodes = condA_allnodes;
            metrics_row.slope_at_45 = slope_at_45;
            metrics_row.max_abs_slope = max(abs(slope_at_45));
            metrics_row.condB_allnodes = condB_allnodes;
            metrics_row.is_sync = is_sync;
        end
        
        edge_sync_matrix(e, ni) = is_sync;
        metrics = [metrics; metrics_row];
        
        % Print with color coding
        if is_sync
            status_str = 'SYNC ✓';
        else
            status_str = 'NO SYNC ✗';
            edge_sync_for_all_n = false;  % Failed at this n
        end
        
        fprintf('  n=%.4g | max|phi/pi|=%.4f | max|slope|=%.4f | A=%d B=%d -> %s\n', ...
            n, metrics_row.max_abs_phi_pi, metrics_row.max_abs_slope, ...
            condA_allnodes, condB_allnodes, status_str);
        
        % If failed, show which condition and nodes
        if ~is_sync
            if ~condA_allnodes
                bad_nodes = find(~condA_per_node);
                fprintf('    -> Failed Cond A at nodes: %s\n', mat2str(bad_nodes));
            end
            if ~condB_allnodes
                bad_nodes = find(~condB_per_node);
                fprintf('    -> Failed Cond B at nodes: %s (slopes: %s)\n', ...
                    mat2str(bad_nodes), mat2str(slope_at_45(bad_nodes), 3));
            end
        end
    end
    
    % Summary for this edge
    if edge_sync_for_all_n
        fprintf('  *** Edge %d-%d is INDIFFERENT (syncs for all n) ***\n', i1, i2);
    else
        fprintf('  --- Edge %d-%d is NOT indifferent ---\n', i1, i2);
    end
    fprintf('\n');
end

fprintf('==================================================\n');

%% ---------- Identify and save indifferent edges ----------
is_indifferent = all(edge_sync_matrix, 2);
indifferent_edges = edge_list(is_indifferent, :);

fprintf('\n=== FINAL SUMMARY ===\n');
fprintf('Total edges: %d\n', m);
fprintf('Indifferent edges: %d\n', sum(is_indifferent));
if sum(is_indifferent) > 0
    fprintf('Indifferent edge pairs:\n');
    for k = 1:size(indifferent_edges, 1)
        fprintf('  Edge %d-%d\n', indifferent_edges(k,1), indifferent_edges(k,2));
    end
else
    fprintf('NO INDIFFERENT EDGES FOUND!\n');
    fprintf('\nPossible reasons:\n');
    fprintf('  1. Kc might be incorrect (try longer tspan_short or different thresholds)\n');
    fprintf('  2. Network topology is sensitive to all edge modifications\n');
    fprintf('  3. slope_threshold (%.2f) might be too strict\n', slope_threshold);
    fprintf('  4. Check if n=0 case is failing (edge removal breaks connectivity)\n');
end
fprintf('=====================\n\n');

save('indifferent_edges_v8_debug.mat', 'indifferent_edges', 'edge_sync_matrix', ...
     'n_values', 'metrics', 'Kc', 'is_indifferent');

fid = fopen('edge_metrics_v8_debug.csv','w');
fprintf(fid, 'i,j,n,max_abs_phi_pi,max_abs_slope,condA,condB,is_sync\n');
for r = 1:numel(metrics)
    row = metrics(r);
    fprintf(fid, '%d,%d,%.8g,%.8f,%.8f,%d,%d,%d\n', ...
        row.i, row.j, row.n, row.max_abs_phi_pi, row.max_abs_slope, ...
        row.condA_allnodes, row.condB_allnodes, row.is_sync);
end
fclose(fid);


%% ---------- Plot ----------
% ===== FIGURE 1: Base check =====
figure('Name','Base check (at K=Kc)','Color','white','Units','normalized',...
       'Position',[0.05 0.05 0.85 0.75]);

% --- LEFT: Base topology ---
subplot(1,2,1); hold on;
for e = 1:m
    i1=edge_list(e,1); i2=edge_list(e,2);
    plot([x_coords(i1) x_coords(i2)], [y_coords(i1) y_coords(i2)], '-', ...
         'Color', [0.6 0.6 0.6], 'LineWidth', 1.2);
end
prod_idx = find(P_vec>0); 
cons_idx = find(P_vec<0);
if ~isempty(cons_idx)
    scatter(x_coords(cons_idx), y_coords(cons_idx), 120, ...
            'MarkerFaceColor', [0.2 1 0.2], 'MarkerEdgeColor',[0 0 0]);
end
if ~isempty(prod_idx)
    scatter(x_coords(prod_idx), y_coords(prod_idx), 120, ...
            'MarkerFaceColor', [1 0.2 0.2], 'MarkerEdgeColor',[0 0 0]);
end
for nidx=1:N
    text(x_coords(nidx)+0.06, y_coords(nidx), sprintf('N%d', nidx), ...
         'Color',[0 0.4470 0.7410],'FontWeight','bold');
end
axis equal; grid on; 
set(gca,'Color','white','XColor','k','YColor','k');
title('Base topology','Color','black'); 
axis on; 
hold off;

% --- RIGHT: Base phase plot ---
subplot(1,2,2);
set(gca,'Color','white','XColor','k','YColor','k'); 
hold on;
colors = hsv(max(N,7)); 
colors = colors(1:N,:);
for node = 1:N
    plot(t0, phi_pi_base(:,node), 'LineWidth', 1.5, 'Color', colors(node,:));
end
xlabel('\alpha t','Color','black'); 
ylabel('\phi/\pi','Color','black');
grid on;

title_txt = sprintf('Base phase vs time (K=%.3f P)', Kc/P_base);
if ~condA_base_all
    % Base system unstable
    title({title_txt; 'BASE SYSTEM UNSTABLE'}, ...
         'Color',[1 0.3 0.3], 'FontWeight','bold');
    viol_nodes = find(~condA_base_per_node);
    for vn = viol_nodes
        idx_first = find(abs(phi_pi_base(:,vn)) > (1 + epsilon_A), 1, 'first');
        if ~isempty(idx_first)
            text(t0(idx_first), phi_pi_base(idx_first,vn), ...
                 sprintf('  N%d violates', vn), 'Color',[1 0.6 0.6], 'FontWeight','bold');
        end
    end
else
    % Base system stable
    title({title_txt; 'Base system STABLE'}, ...
         'Color',[0 0.6 0], 'FontWeight','normal');
end
hold off;

% ===== FIGURE 2: Edge analysis results =====
figure('Name','Edge Analysis','Color','white','Units','normalized',...
       'Position',[0.05 0.05 0.9 0.75]);

% --- LEFT: Topology with indifferent edges highlighted ---
subplot(1,2,1); hold on;
for e = 1:m
    i1=edge_list(e,1); i2=edge_list(e,2);
    if is_indifferent(e)
        % Highlight indifferent edges in yellow/gold
        plot([x_coords(i1) x_coords(i2)], [y_coords(i1) y_coords(i2)], '-', ...
             'Color', [1 0.8 0.2], 'LineWidth', 4.0);
    else
        % Regular edges in gray
        plot([x_coords(i1) x_coords(i2)], [y_coords(i1) y_coords(i2)], '-', ...
             'Color', [0.45 0.45 0.45], 'LineWidth', 1.0);
    end
end
if ~isempty(find(P_vec<0,1))
    scatter(x_coords(find(P_vec<0)), y_coords(find(P_vec<0)), 120, ...
            'MarkerFaceColor',[0.2 1 0.2], 'MarkerEdgeColor','k');
end
if ~isempty(find(P_vec>0,1))
    scatter(x_coords(find(P_vec>0)), y_coords(find(P_vec>0)), 120, ...
            'MarkerFaceColor',[1 0.2 0.2], 'MarkerEdgeColor','k');
end
for n=1:N
    text(x_coords(n)+0.06, y_coords(n), sprintf('N%d', n), ...
         'Color',[0 0.4470 0.7410],'FontWeight','bold', 'FontSize', 10);
end
axis equal; axis on; grid on;
set(gca,'Color','white','XColor','k','YColor','k');
title(sprintf('Indifferent edges: %d/%d (Kc/P=%.3f)', sum(is_indifferent), m, kc_over_P), ...
      'Color','black', 'FontSize', 11);
hold off;

% --- RIGHT: Heatmap of sync status ---
subplot(1,2,2);
imagesc(edge_sync_matrix);
colormap([1 0.3 0.3; 0.3 1 0.3]);  % Red = no sync, Green = sync
set(gca, 'XTick', 1:numel(n_values), ...
         'XTickLabel', arrayfun(@(x) sprintf('%.2g', x), n_values, 'UniformOutput', false));
set(gca, 'YTick', 1:m);
ylabel('Edge number');
xlabel('n value (multiplier of Kc)');
title('Sync status per edge', 'FontSize', 11);
colorbar('Ticks', [0.25, 0.75], 'TickLabels', {'No Sync', 'Sync'});

fprintf('Saved results to edge_metrics_v8_debug.csv and .mat\n');
fprintf('Done.\n');
end






%% ---------- ODE helper ----------
function dydt = second_order_ode(~, y, P_vec, K_matrix, alpha)
    N = length(P_vec);
    phi = y(1:N);
    dphi = y(N+1:2*N);
    phi_diff = phi' - phi;
    interactions = K_matrix .* sin(phi_diff);
    coupling = sum(interactions, 2);
    d2phi = P_vec - alpha * dphi + coupling;
    dydt = [dphi; d2phi];
end