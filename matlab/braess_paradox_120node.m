function braess_paradox_120node()

    % --- 1. Network Parameters ---
    N = 120;                    % Number of nodes (120)
    P_base = 1.0;               % Base power
    N_gen = 10;                 % Number of Generators (Producers)
    N_cons = N - N_gen;         % Number of Consumers (110)
    K_uniform = 1.03 * P_base;  % Uniform coupling K > Kc (from 8-node example)
    alpha = P_base;             % Damping coefficient

    % --- 2. Node Types and Power Vector ---
    % Randomly select 10 nodes as generators, rest are consumers
    rng(42); % Set random seed for reproducibility
    % 'producers' contains the indices (1 to 120) of the generator nodes
    producers = sort(randperm(N, N_gen)); 
    % 'consumers' contains the indices of the consumer nodes
    consumers = setdiff(1:N, producers); 
    
    % Power vector P_vec: P_gen = 11*P_0, P_cons = -P_0 (as suggested in text)
    % To maintain power balance (sum(P_vec) = 0):
    % N_gen * P_gen + N_cons * P_cons = 0
    % 10 * P_gen + 110 * (-P_cons) = 0
    % P_gen = 11 * P_cons. If P_base is P_cons, then P_gen = 11*P_base.
    P_gen = 11 * P_base; 
    P_cons = -P_base; 

    P_vec = zeros(N, 1);
    P_vec(producers) = P_gen;
    P_vec(consumers) = P_cons;
    
    % Verify Power Balance (should be close to zero)
    if abs(sum(P_vec)) > 1e-9
        fprintf('Warning: Power vector sum is not zero (Sum = %f). Check power settings.\n', sum(P_vec));
    end

    % --- 3. Topology Definition (Original Network) ---
    fprintf('Generating a random scale-free network for the 120-node topology...\n');
    % This is a placeholder for the actual UK power grid topology.
    % You should replace this with the true adjacency matrix A_UK_grid if available.
    
    % Create a sparse, scale-free network (N=120 nodes, m=2 new edges per step)
    % This ensures the network has heterogeneous connectivity, like a real power grid.
    A_original = generate_scale_free_network(N, 2); 

    % Define the target edge for Braess's paradox (e.g., between two specific nodes)
    % We arbitrarily choose nodes (20, 45) for the paradox edge, similar to the 8-node model's specific edge.
    paradox_node_1 = 20;
    paradox_node_2 = 45;

    % K-matrices setup
    K_matrix_original = K_uniform * A_original; 
                  
    % --- Network 2: "Add capacity" ---
    % Doubles capacity of edge (20, 45)
    A_add_capacity = A_original;
    K_matrix_add_capacity = K_uniform * A_add_capacity; 
    K_matrix_add_capacity(paradox_node_1, paradox_node_2) = 2 * K_uniform; 
    K_matrix_add_capacity(paradox_node_2, paradox_node_1) = 2 * K_uniform; 
    
    % --- Network 3: "Add line" ---
    % Adds new edge (e.g., between two nodes that were not previously connected)
    % Check for an unconnected pair. For large random networks, this is harder, 
    % so we ensure a random unconnected pair is chosen.
    while A_original(paradox_node_1, paradox_node_2) == 1
        % If the chosen 'paradox edge' was already connected, pick a new unconnected edge
        paradox_node_1 = randi(N);
        paradox_node_2 = randi(N);
        if paradox_node_1 == paradox_node_2
            paradox_node_2 = mod(paradox_node_2, N) + 1;
        end
    end
    
    A_add_line = A_original;
    A_add_line(paradox_node_1, paradox_node_2) = 1; 
    A_add_line(paradox_node_2, paradox_node_1) = 1; 
    K_matrix_add_line = K_uniform * A_add_line; 

   % --- 3.5. Plot Network Topologies (Modified for 120 Nodes) ---
    fprintf('Visualizing a single 120-node network topology (Original config)...\n');
    fprintf('Note: Visualizing all 120 nodes with random coordinates will be dense.\n');

    % Create graph objects and node labels for plotting
    node_labels = cell(N, 1);
    for i = 1:N
        if ismember(i, producers)
            node_labels{i} = sprintf('%d (+P)', i);
        else
            node_labels{i} = sprintf('%d (-P)', i);
        end
    end
    
    % Generate random x,y coordinates for 120 nodes for visualization
    % Scale them to a reasonable range, e.g., 0 to 100
    rng(123); % For reproducible random coordinates
    x_coords = rand(N, 1) * 100;
    y_coords = rand(N, 1) * 100;
    
    % Create Graph objects for visualization
    
    % Create Graph objects for visualization
    G_original = graph(A_original, node_labels, 'omitselfloops');
    G_add_capacity = graph(K_matrix_add_capacity > 0, node_labels, 'omitselfloops'); % Use K_matrix to get topology
    G_add_line = graph(A_add_line, node_labels, 'omitselfloops');
        
    % === 1. Plot Original Configuration (No highlighted edge) ===
    figure('Name', '120-Node Network Topology (Original Configuration)', 'NumberTitle', 'off', 'Position', [50, 500, 800, 700]);
    p1 = plot(G_original, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 1); 
    title('Original 120-Node Configuration (Random Layout)');
    customize_plot(p1, producers, consumers, [], []); % <-- Pass [] so no edge is highlighted
    
    % === 2. Plot Add Capacity (Edge exists, capacity is doubled) ===
    % The paradox edge must exist in A_original for the 'Add Capacity' logic to be sound.
    figure('Name', '120-Node Network Topology (Add Capacity)', 'NumberTitle', 'off', 'Position', [850, 500, 800, 700]);
    p2 = plot(G_add_capacity, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 1);
    title(sprintf('120-Node Network with Added Capacity on Edge (%d, %d)', paradox_node_1, paradox_node_2));
    % Highlight the edge. This works because the edge is in the topology A_add_capacity.
    customize_plot(p2, producers, consumers, [paradox_node_1, paradox_node_2], [0.1 0.1 0.9]); 
    
    
    % === 3. Plot Add Line (New edge is added) ===
    figure('Name', '120-Node Network Topology (Add Line)', 'NumberTitle', 'off', 'Position', [1650, 500, 800, 700]);
    p3 = plot(G_add_line, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 1);
    title(sprintf('120-Node Network with Added Line (%d, %d)', paradox_node_1, paradox_node_2));
    % Highlight the new edge. This works because the new edge is in the topology A_add_line.
    customize_plot(p3, producers, consumers, [paradox_node_1, paradox_node_2], [0.9 0.1 0.1]);
    % The quantitative results (order parameter plots) are more informative for this scale.

    fprintf('Paradox edge selected for simulation: (%d, %d)\n', paradox_node_1, paradox_node_2);

    % --- 4. Simulation Parameters ---
    tspan = [0 40]; % Longer run time needed for larger systems to reach steady state
    y0 = zeros(2 * N, 1); 
    options = odeset('RelTol', 1e-6, 'AbsTol', 1e-9); 
    
    % --- 5. Run Simulations ---
    fprintf('Simulating original 120-node network...\n');
    [t_orig, y_orig] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_original, alpha), tspan, y0, options);
    fprintf('Original network simulation complete.\n');
    
    fprintf('Simulating "add capacity" network...\n');
    [t_cap, y_cap] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_add_capacity, alpha), tspan, y0, options);
    fprintf('"Add capacity" network simulation complete.\n');
    
    fprintf('Simulating "add line" network...\n');
    [t_line, y_line] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_add_line, alpha), tspan, y0, options);
    fprintf('"Add line" network simulation complete.\n');

    % --- 6. Analyze and Plot Results (Order Parameter Time Series) ---
    fprintf('Plotting order parameter dynamics...\n');
    
    % Order parameter r = |(1/N) * sum(e^(i*phi_j))|
    r_orig = abs(mean(exp(1i * y_orig(:, 1:N)), 2));
    r_cap = abs(mean(exp(1i * y_cap(:, 1:N)), 2));
    r_line = abs(mean(exp(1i * y_line(:, 1:N)), 2));

    figure('Name', '120-Node Braess Paradox Simulation', 'NumberTitle', 'off', 'Position', [100, 100, 800, 600]);
    
    subplot(2, 1, 1);
    plot(t_orig, r_orig, 'k-', 'LineWidth', 2); hold on;
    plot(t_cap, r_cap, 'b--', 'LineWidth', 2);
    plot(t_line, r_line, 'r-.', 'LineWidth', 2);
    title('Order Parameter r(t) for 120-Node Network');
    xlabel('\alpha t');
    ylabel('Order Parameter r');
    legend({'Original', 'Add Capacity', 'Add Line'}, 'Location', 'southwest');
    grid on;
    
    % Check final synchrony status (r at the end of simulation)
    final_r_orig = r_orig(end);
    final_r_cap = r_cap(end);
    final_r_line = r_line(end);

    subplot(2, 1, 2);
    bar([final_r_orig, final_r_cap, final_r_line]);
    set(gca, 'XTickLabel', {'Original', 'Add Capacity', 'Add Line'});
    title('Steady-State Order Parameter');
    ylabel('r_{steady-state}');
    
    fprintf('Final r values: Original = %.4f, Add Capacity = %.4f, Add Line = %.4f\n', ...
            final_r_orig, final_r_cap, final_r_line);
    fprintf('If Add Capacity/Add Line r is lower than Original r, the paradox is observed.\n');
    fprintf('Done.\n');
end

% --- Helper Functions (Same as Original Code) ---
% Helper function
function customize_plot(p_handle, producers, consumers, highlight_edge, highlight_color)
    p_handle.NodeFontSize = 8; % Reduced font size for 120 nodes
    p_handle.NodeFontWeight = 'bold';
    
    % Use smaller markers for the dense 120-node network
    highlight(p_handle, producers, 'NodeColor', [0.9 0.1 0.1], 'MarkerSize', 4, 'Marker', 'o');
    highlight(p_handle, consumers, 'NodeColor', [0.1 0.7 0.1], 'MarkerSize', 4, 'Marker', 'o');
    
    % Add '+' and '-' labels to nodes (optional for 120 nodes, can be cluttered)
    % I recommend commenting this section out for the 120-node plot, 
    % as 120 labels will make the graph unreadable.
    N_nodes = numel(p_handle.NodeLabel);
    for i = 1:N_nodes
        if ismember(i, producers)
            label_char = '+';
            label_color = [0.9 0.1 0.1];
        else
            label_char = '-';
            label_color = [0.1 0.7 0.1];
        end
        % text(p_handle.XData(i) + 0.5, p_handle.YData(i) + 0.5, label_char, ...
        %      'Color', label_color, 'FontSize', 6, 'FontWeight', 'bold', ...
        %      'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
        p_handle.NodeLabel{i} = ''; % Clear default node label
    end
    
    % Highlight a specific edge if provided
    if ~isempty(highlight_edge) && numel(highlight_edge) == 2
        highlight(p_handle, highlight_edge(1), highlight_edge(2), ...
                  'EdgeColor', highlight_color, 'LineWidth', 2, 'LineStyle', '-');
    end
    
    ax = gca;
    ax.XTickLabel = {}; % Remove x-axis tick labels
    ax.YTickLabel = {}; % Remove y-axis tick labels
    axis equal; % Maintain aspect ratio
end

% ODE Function (Same as before)
function dydt = second_order_ode(t, y, P_vec, K_matrix, alpha)
    N = length(P_vec);
    phi = y(1:N);
    dphi_dt = y(N+1:2*N);
    
    % Efficient calculation using matrix operations
    phi_diff_matrix = phi' - phi; 
    interactions = K_matrix .* sin(phi_diff_matrix);
    coupling_term = sum(interactions, 2);
    
    d2phi_dt2 = P_vec - alpha * dphi_dt + coupling_term;
    dydt = [dphi_dt; d2phi_dt2];
end

% Function to generate a Scale-Free (Barabási-Albert) Network
% Requires: N (nodes), m (edges to attach from new node to existing nodes)
function A = generate_scale_free_network(N, m)
    % A simple implementation of the Barabási-Albert model
    if N <= m
        A = ones(N) - eye(N);
        return;
    end
    
    % Start with a complete graph of m+1 nodes
    A = zeros(N);
    for i = 1:m+1
        for j = i+1:m+1
            A(i, j) = 1;
            A(j, i) = 1;
        end
    end
    
    % Iteratively add the remaining nodes
    for i = m+2:N
        % Calculate current degrees
        degrees = sum(A(1:i-1, 1:i-1), 2);
        total_degree = sum(degrees);
        
        % Calculate selection probabilities (proportional to degree)
        probabilities = degrees / total_degree;
        
        selected_nodes = randi(i-1, m, 1); % Select m nodes uniformly at random from 1 to i-1
        
        % Add connections
        for j = 1:m
            target = selected_nodes(j);
            A(i, target) = 1;
            A(target, i) = 1;
        end
    end
end