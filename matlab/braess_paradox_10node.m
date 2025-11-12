function braess_paradox_10node()
  
    % Network Parameters
    N = 10;                      % Number of nodes
    P_base = 1.0;               % Base power
    
    % Parameters from Figure 1 caption:
    K_uniform = 1.0208 * P_base;  % K>Kc
    alpha = P_base;             % Damping coefficient
    
    % Node types (Producers: +P, Consumers: -P)
    producers = [2, 3, 4, 7, 9]; % Nodes with +P
    consumers = [1, 5, 6, 8, 10]; % Nodes with -P
    
    P_vec = zeros(N, 1);
    P_vec(producers) = P_base;
    P_vec(consumers) = -P_base;
    
    % Adjacency Matrices and Coupling Strengths
    
   % --- 2. Adjacency Matrices and Coupling Strengths (Original) ---
    A_original = zeros(N, N);
    % Edges from Figure 1a (12 edges total)
    edges_orig = [1,2; 1,7; 2,3; 3,7; 4,8; 4,5; 5,6; 6,8; 3,9; 9,4; 1,10; 10,6; 9,10];
    for i = 1:size(edges_orig, 1)
        n1 = edges_orig(i, 1);
        n2 = edges_orig(i, 2);
        A_original(n1, n2) = 1;
        A_original(n2, n1) = 1;
    end
    K_matrix_original = K_uniform * A_original;
                  
    % --- Network 2: "Add capacity" (Figure 1b) ---
    % Doubles capacity of edge (3,4)
    A_add_capacity = A_original; % Topology is the same
    K_matrix_add_capacity = K_uniform * A_add_capacity; % Start with uniform couplings
    K_matrix_add_capacity(3, 9) = 2 * K_uniform; % Double K for edge (3,4)
    K_matrix_add_capacity(9, 3) = 2 * K_uniform; % Symmetric
    
    % --- Network 3: "Add line" (Figure 1c) ---
    % Adds new edge (4,2)
    A_add_line = A_original;
    A_add_line(4, 2) = 1; % Add new edge
    A_add_line(2, 4) = 1; % Symmetric
    K_matrix_add_line = K_uniform * A_add_line; % New edge also has uniform coupling K_uniform
    
    % --- 3. Simulation Parameters ---
    % tspan corresponds to alpha*t up to ~25, so t goes to 25 given alpha=P_base=1
    tspan = [0 60];
    
    % Initial conditions from Figure 1 caption: phi_j = 0, d(phi_j)/dt = 0
    % y = [phi_1, ..., phi_N, dphi_1, ..., dphi_N]  
    y0 = zeros(2 * N, 1); 
    
    % ODE solver options (increased precision for stability analysis)
    options = odeset('RelTol', 1e-6, 'AbsTol', 1e-9); 
    
    % --- 3.5. Plot Network Topologies ---
    fprintf('Visualizing network topologies...\n');
    
    % Create graph objects and node labels for plotting
    node_labels = cell(N, 1);
    for i = 1:N
        if ismember(i, producers)
            node_labels{i} = sprintf('%d (+P)', i);
        else
            node_labels{i} = sprintf('%d (-P)', i);
        end
    end
    
    % Node IDs:   1    2    3    4    5    6    7    8  9  10
    x_coords = [6.0, 5.0, 6.0, 3.0, 2.0, 3.0, 7.0, 4.0, 4.5, 4.5]; 
    y_coords = [2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 3.0, 3.0, 4.0, 2.0]; 

    % Create Graph objects for visualization
    G_original = graph(A_original, node_labels, 'omitselfloops');
    G_add_capacity = graph(A_add_capacity, node_labels, 'omitselfloops');
    G_add_line = graph(A_add_line, node_labels, 'omitselfloops');
    
    figure('Name', 'Network Topologies (Figure 1a, 1b, 1c)', 'NumberTitle', 'off', 'Position', [50, 500, 1200, 400]);
    
    % Subplot 1: Original Network (Fig. 1a)
    subplot(1, 3, 1);
    p1 = plot(G_original, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 2);
    title('a Original configuration');
    customize_plot(p1, producers, consumers, [], []);

    % Subplot 2: Add Capacity (Fig. 1b)
    subplot(1, 3, 2);
    p2 = plot(G_add_capacity, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 2);
    title('b Add capacity');
    customize_plot(p2, producers, consumers, [3,9], [0.1 0.1 0.9]); % Highlight edge (3,4) in blue
    
    % Subplot 3: Add Line (Fig. 1c)
    subplot(1, 3, 3);
    p3 = plot(G_add_line, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 2);
    title('c Add line');
    customize_plot(p3, producers, consumers, [4,2], [0.1 0.1 0.9]);
    
    % --- 4. Run Simulations ---
    fprintf('Simulating original network (Fig. 1d)...\n');
    [t_orig, y_orig] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_original, alpha), tspan, y0, options);
    fprintf('Original network simulation complete.\n');

    fprintf('Simulating "add capacity" network (Fig. 1e)...\n');
    [t_cap, y_cap] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_add_capacity, alpha), tspan, y0, options);
    fprintf('"Add capacity" network simulation complete.\n');

    fprintf('Simulating "add line" network (Fig. 1f)...\n');
    [t_line, y_line] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_add_line, alpha), tspan, y0, options);
    fprintf('"Add line" network simulation complete.\n');

    % --- 5. Analyze and Plot Results (Phases) ---
    fprintf('Plotting phase dynamics...\n');
    
    % Extract phases and convert to phi/pi
    phi_pi_orig = y_orig(:, 1:N) / pi;
    phi_pi_cap = y_cap(:, 1:N) / pi;
    phi_pi_line = y_line(:, 1:N) / pi;
    
    figure('Name', 'Simulation Results (Figure 1d, 1e, 1f)', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 700]);
    
    % Common Y-axis limits for all plots to match paper
    y_lims = [-2.5, 2.5]; 
    
    % --- Subplot 1: Original Network (Sync) - Fig 1d ---
    subplot(1, 3, 1);
    hold on;
    h_orig_prod = plot(t_orig, phi_pi_orig(:, producers), 'r-', 'LineWidth', 1.5);
    h_orig_cons = plot(t_orig, phi_pi_orig(:, consumers), 'g-', 'LineWidth', 1.5);
    hold off;
    title('d Sync.');
    xlabel('\alpha t');
    ylabel('\phi_j(t) / \pi');
    grid on;
    ylim(y_lims);
    legend([h_orig_prod(1), h_orig_cons(1)], {'Producers', 'Consumers'}, 'Location', 'southeast', 'AutoUpdate', 'off');
    
    % --- Subplot 2: Add Capacity (No Sync) - Fig 1e ---
    subplot(1, 3, 2);
    hold on;
    h_cap_prod = plot(t_cap, phi_pi_cap(:, producers), 'r-', 'LineWidth', 1.5);
    h_cap_cons = plot(t_cap, phi_pi_cap(:, consumers), 'g-', 'LineWidth', 1.5);
    hold off;
    title('e No Sync.');
    xlabel('\alpha t');
    ylabel('\phi_j(t) / \pi');
    grid on;
    ylim(y_lims);
    
    % --- Subplot 3: Add Line (No Sync) - Fig 1f ---
    subplot(1, 3, 3);
    hold on;
    h_line_prod = plot(t_line, phi_pi_line(:, producers), 'r-', 'LineWidth', 1.5);
    h_line_cons = plot(t_line, phi_pi_line(:, consumers), 'g-', 'LineWidth', 1.5);
    hold off;
    title('f No Sync.');
    xlabel('\alpha t');
    ylabel('\phi_j(t) / \pi');
    grid on;
    ylim(y_lims);
    
    fprintf('Done.\n');

    fprintf('Generating Figure 2 (c–d)');
    
    % === 2c: Order parameter r vs K ===
    fprintf('Sweeping coupling strength K for Fig. 2c...\n');
    
    K_list = linspace(0.7, 1.4, 25) * P_base;
    r_orig = zeros(size(K_list));
    r_cap  = zeros(size(K_list));
    r_line = zeros(size(K_list));
    
    for k = 1:length(K_list)
        % Original
        Kmat = K_list(k) * A_original;
        [~, y] = ode45(@(t,y) second_order_ode(t,y,P_vec,Kmat,alpha), [0 40], y0);
        phi = y(end,1:N);
        r_orig(k) = abs(mean(exp(1i*phi)));
    
        % Add capacity
        Kmat = K_list(k) * A_add_capacity;
        Kmat(3,4) = 2 * K_list(k);
        Kmat(4,3) = 2 * K_list(k);
        [~, y] = ode45(@(t,y) second_order_ode(t,y,P_vec,Kmat,alpha), [0 40], y0);
        phi = y(end,1:N);
        r_cap(k) = abs(mean(exp(1i*phi)));
    
        % Add line
        Kmat = K_list(k) * A_add_line;
        [~, y] = ode45(@(t,y) second_order_ode(t,y,P_vec,Kmat,alpha), [0 40], y0);
        phi = y(end,1:N);
        r_line(k) = abs(mean(exp(1i*phi)));
    end
    % === 2c: Order parameter r vs K (unchanged) ===
    figure('Name', 'Figure 2c Order Parameter', 'NumberTitle', 'off');
    plot(K_list/P_base, r_orig, 'k-', 'LineWidth',2); hold on;
    plot(K_list/P_base, r_cap,  'b--', 'LineWidth',2);
    plot(K_list/P_base, r_line, 'r-.', 'LineWidth',2);
    xlabel('K/P'); ylabel('r (order parameter)');
    legend('Original','Add capacity','Add line','Location','southeast');
    title('2c  Order parameter vs coupling strength');
    grid on;
    fprintf('Order-parameter sweep complete.\n');
    
    % === 2d: r(ΔK₃₄, K) map with x = ΔK/P and y = K/P ===
    fprintf('Generating 2D r(ΔK₃₄, K) map for Fig. 2d (x=ΔK/P, y=K/P)...\n');
    
    K_base  = linspace(0.8,1.4,25)*P_base;   % vertical axis (K/P)
    DeltaKs = linspace(0,1.0,25)*P_base;     % horizontal axis (ΔK/P)
    
    r_map   = zeros(length(K_base), length(DeltaKs));  % (rows=K, cols=ΔK)
    
    for i = 1:length(K_base)       % loop over base coupling (vertical)
        for j = 1:length(DeltaKs)  % loop over ΔK (horizontal)
            Kmat = K_base(i) * A_original;
            Kmat(3,4) = K_base(i) + DeltaKs(j);
            Kmat(4,3) = Kmat(3,4);
            [~, y] = ode45(@(t,y) second_order_ode(t,y,P_vec,Kmat,alpha), [0 30], y0);
            phi = y(end,1:N);
            r_map(i,j) = abs(mean(exp(1i*phi)));
        end
    end
    
    figure('Name', 'Figure 2d Synchrony Region', 'NumberTitle', 'off');
    imagesc(DeltaKs/P_base, K_base/P_base, r_map);  % X = ΔK/P, Y = K/P
    set(gca,'YDir','normal');
    xlabel('\DeltaK_{34}/P');
    ylabel('K/P');
    colorbar;
    title('2d  Synchrony region (order parameter r)');
    fprintf('Figure 2d generation complete (x = ΔK/P, y = K/P).\n');


end

% ODE Function
function dydt = second_order_ode(t, y, P_vec, K_matrix, alpha)
    
    N = length(P_vec);
    
    % Extract phases and velocities from the state vector y
    phi = y(1:N);
    dphi_dt = y(N+1:2*N);
    
    % Calculate the coupling term: sum(K_ij * sin(phi_i - phi_j))
    % This can be done efficiently with matrix operations:
    % 1. Create a matrix of all phase differences (phi_i - phi_j)
    phi_diff_matrix = phi' - phi; % (N x 1) - (1 x N) -> N x N matrix
    
    % 2. Element-wise multiply by K_matrix and take sine
    interactions = K_matrix .* sin(phi_diff_matrix);
    
    % 3. Sum contributions for each node (sum along rows, result is N x 1)
    coupling_term = sum(interactions, 2);
    
    % Calculate the second derivative (d^2(phi)/dt^2 = P_j - alpha * d(phi)/dt + coupling_term)
    d2phi_dt2 = P_vec - alpha * dphi_dt + coupling_term;
    
    % Return the derivative of the state vector
    dydt = [dphi_dt; d2phi_dt2];
end

% Helper function
function customize_plot(p_handle, producers, consumers, highlight_edge, highlight_color)
    p_handle.NodeFontSize = 10;
    p_handle.NodeFontWeight = 'bold';
    highlight(p_handle, producers, 'NodeColor', [0.9 0.1 0.1], 'MarkerSize', 8, 'Marker', 'o');
    highlight(p_handle, consumers, 'NodeColor', [0.1 0.7 0.1], 'MarkerSize', 8, 'Marker', 'o');
    
    % Add '+' and '-' labels to nodes (requires slightly different approach)
    N_nodes = numel(p_handle.NodeLabel);
    for i = 1:N_nodes
        if ismember(i, producers)
            label_char = '+';
            label_color = [0.9 0.1 0.1];
        else
            label_char = '-';
            label_color = [0.1 0.7 0.1];
        end
        text(p_handle.XData(i) + 0.1, p_handle.YData(i) + 0.1, label_char, ...
             'Color', label_color, 'FontSize', 12, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        p_handle.NodeLabel{i} = ''; % Clear default node label to prevent overlap
    end
    
    % Highlight a specific edge if provided (e.g., new line or increased capacity)
    if ~isempty(highlight_edge) && numel(highlight_edge) == 2
        highlight(p_handle, highlight_edge(1), highlight_edge(2), ...
                  'EdgeColor', highlight_color, 'LineWidth', 3, 'LineStyle', '-');
    end
    
    ax = gca;
    ax.XTickLabel = {}; % Remove x-axis tick labels for cleaner look
    ax.YTickLabel = {}; % Remove y-axis tick labels
    axis equal; % Maintain aspect ratio
end

