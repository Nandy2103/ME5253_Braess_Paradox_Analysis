function braess_paradox_addcap_optimized()
  
    % --- 1. Network and System Parameters ---
    N = 8;                      % Number of nodes
    P_base = 1.0;               % Base power
    K_uniform = 1.03 * P_base;  % K > Kc
    alpha = P_base;             % Damping coefficient
    update_factor = 2;        % The factor by which K_uniform is multiplied
    
    % Node types (Producers: +P, Consumers: -P)
    producers = [2, 3, 4, 7]; % Nodes with +P
    consumers = [1, 5, 6, 8]; % Nodes with -P
    
    P_vec = zeros(N, 1);
    P_vec(producers) = P_base;
    P_vec(consumers) = -P_base;
    
    % --- 2. Adjacency Matrices and Coupling Strengths (Original) ---
    A_original = zeros(N, N);
    % Edges from Figure 1a (12 edges total)
    edges_orig = [1,2; 1,6; 1,7; 2,3; 3,4; 3,7; 4,8; 4,5; 5,6; 6,8];
    for i = 1:size(edges_orig, 1)
        n1 = edges_orig(i, 1);
        n2 = edges_orig(i, 2);
        A_original(n1, n2) = 1;
        A_original(n2, n1) = 1;
    end
    K_matrix_original = K_uniform * A_original; 
    
    % --- Define the Edges to "Add Capacity" To ---
    % This list contains all 12 edges from the original network
    edges_to_update = edges_orig; 
    
    % --- 3. Simulation Parameters ---
    tspan = [0 50];
    y0 = zeros(2 * N, 1); 
    options = odeset('RelTol', 1e-6, 'AbsTol', 1e-9); 
    
    % --- 4. Store Simulation Results and K Matrices ---
    % Use a cell array to store the simulation results for each case
    num_updates = size(edges_to_update, 1);
    t_results = cell(1, num_updates);
    y_results = cell(1, num_updates);
    K_matrices = cell(1, num_updates);
    
    % --- 5. LOOP: Create K matrices, Run Simulations, and Store Results ---
    fprintf('Starting simulations for %d capacity updates...\n', num_updates);
    for k = 1:num_updates
        n1 = edges_to_update(k, 1);
        n2 = edges_to_update(k, 2);
        
        % Create K matrix for the current update
        K_matrix_current = K_uniform * A_original;
        K_matrix_current(n1, n2) = update_factor * K_uniform; % Apply update factor
        K_matrix_current(n2, n1) = update_factor * K_uniform; 
        K_matrices{k} = K_matrix_current;
        
        % Run Simulation
        fprintf('  Simulating update on edge (%d, %d)...\n', n1, n2);
        [t, y] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_current, alpha), tspan, y0, options);
        
        % Store Results
        t_results{k} = t;
        y_results{k} = y;
    end
    fprintf('All simulations complete.\n');
    
    % --- 6. Plot Network Topologies (Visualization Section) ---
    
    % Create graph objects and node labels for plotting
    node_labels = cell(N, 1);
    for i = 1:N
        if ismember(i, producers)
            node_labels{i} = sprintf('%d (+P)', i);
        else
            node_labels{i} = sprintf('%d (-P)', i);
        end
    end
    
    % Node IDs:   1    2    3    4    5    6    7    8  
    x_coords = [6.0, 5.0, 6.0, 3.0, 2.0, 3.0, 7.0, 4.0]; 
    y_coords = [2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 3.0, 3.0]; 
    A_add_capacity = A_original; % Topology is the same
    G_add_capacity = graph(A_add_capacity, node_labels, 'omitselfloops');
    
    figure('Name', 'Network Topologies: Edge Capacity Updates', 'NumberTitle', 'off', 'Position', [50, 500, 1400, 400]);
    
    % LOOP: Plot Topologies
    for k = 1:num_updates
        n1 = edges_to_update(k, 1);
        n2 = edges_to_update(k, 2);
        
        subplot(2, 5, k);
        p_plot = plot(G_add_capacity, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 2);
        title(sprintf('b K(%d,%d) = %.1f K_{orig}', n1, n2, update_factor));
        % Assuming customize_plot is defined elsewhere to color nodes/edges
        customize_plot(p_plot, producers, consumers, [n1, n2], [0.1 0.1 0.9]); 
    end
    fprintf('Visualizing network topologies...\n');
    
    % --- 7. Analyze and Plot Results (Phases) ---
    fprintf('Plotting phase dynamics...\n');
    
    figure('Name', 'Simulation Results: Phase Dynamics with Capacity Updates', 'NumberTitle', 'off', 'Position', [50, 50, 1400, 700]);
    y_lims = [-3.5, 3.5]; 
    
    % LOOP: Plot Phase Dynamics
    for k = 1:num_updates
        % Extract phases and convert to phi/pi
        phi_pi_current = y_results{k}(:, 1:N) / pi;
        t_current = t_results{k};
        
        subplot(2, 5, k);
        hold on;
        h_prod = plot(t_current, phi_pi_current(:, producers), 'r-', 'LineWidth', 1.5);
        h_cons = plot(t_current, phi_pi_current(:, consumers), 'g-', 'LineWidth', 1.5);
        hold off;
        xlabel('\alpha t');
        ylabel('\phi_j(t) / \pi');
        grid on;
        ylim(y_lims);
        
        % Add legend only to the first plot
        if k == 1
            legend([h_prod(1), h_cons(1)], {'Producers', 'Consumers'}, 'Location', 'southeast', 'AutoUpdate', 'off');
        end
        
        n1 = edges_to_update(k, 1);
        n2 = edges_to_update(k, 2);
        title(sprintf('K(%d,%d) Update: Phases', n1, n2));
    end
    
    fprintf('Done.\n');

% === 2c: Order parameter r vs K ===
    fprintf('Sweeping coupling strength K for Fig. 2c...\n');
    
    % --- Setup ---
    K_list = linspace(0.0, 3, 25) * P_base; % K values to sweep
    num_K = length(K_list);
    num_updates = size(edges_orig, 1); % Should be 12
    
    % Initialize results storage
    r_orig = zeros(1, num_K);
    
    % Initialize a cell array to store the r values for all 12 updated networks
    % Each cell will contain a vector of size num_K
    r_updates = cell(1, num_updates); 
    for i = 1:num_updates
        r_updates{i} = zeros(1, num_K);
    end
    
    % --- Main Sweep Loop ---
    for k_idx = 1:num_K
        K_val = K_list(k_idx);
        
        % --- 1. Original Network Calculation ---
        Kmat_orig = K_val * A_original;
        
        % Run simulation (using a short tspan of [0 40] to reach steady state)
        [~, y] = ode45(@(t,y) second_order_ode(t,y,P_vec,Kmat_orig,alpha), [0 40], y0);
        phi = y(end, 1:N);
        % Calculate Order Parameter r = |<e^(i*phi)>|
        r_orig(k_idx) = abs(mean(exp(1i * phi)));
        
        % --- 2. LOOP for 12 "Add Capacity" Networks ---
        for u_idx = 1:num_updates
            n1 = edges_orig(u_idx, 1);
            n2 = edges_orig(u_idx, 2);
            
            % Initialize K matrix with uniform coupling K_val
            Kmat_update = K_val * A_original; 
            
            % Apply the capacity update (update = 0.2 factor)
            Kmat_update(n1, n2) = update_factor * K_val;
            Kmat_update(n2, n1) = update_factor * K_val;
            
            % Run simulation
            [~, y] = ode45(@(t,y) second_order_ode(t,y,P_vec,Kmat_update,alpha), [0 40], y0);
            phi = y(end, 1:N);
            
            % Store Order Parameter
            r_updates{u_idx}(k_idx) = abs(mean(exp(1i * phi)));
        end
    end
    fprintf('Order-parameter sweep complete.\n');
    
    % --- Plotting Results (Figure 2c) ---
    figure('Name', 'Figure 2c Order Parameter', 'NumberTitle', 'off');
    hold on;
    
    % Plot Original Network (Black solid line)
    plot(K_list/P_base, r_orig, 'k-', 'LineWidth', 3, 'DisplayName', 'Original'); 
    
    % Plot the 12 "Add Capacity" Scenarios (Various colors/dashes)
    
    % Choose a set of distinguishable colors/styles for the 12 lines
    % Using blue for the first few to highlight (e.g., matching the paper)
    colors = [0 0 1; 0.5 0.5 1; 0 0 0.5; 1 0 0; 1 0.5 0.5; 0.5 0 0; 0 1 0; 0.5 1 0.5; 0 0.5 0; 0 1 1; 0.5 0.5 0; 0.5 0 0.5]; 
    
    h_updates = zeros(1, num_updates);
    for u_idx = 1:num_updates
        n1 = edges_orig(u_idx, 1);
        n2 = edges_orig(u_idx, 2);
        
        h_updates(u_idx) = plot(K_list/P_base, r_updates{u_idx}, '--', ...
            'Color', colors(u_idx,:), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('Update Edge (%d, %d)', n1, n2));
    end
    
    % If you have r_full, r_star, r_ring from previous analysis, include them here:
    % plot(K_list/P_base, r_full,  'b--', 'LineWidth', 2, 'DisplayName', 'Full System'); 
    % plot(K_list/P_base, r_star,  'g--', 'LineWidth', 2, 'DisplayName', 'Star System'); 
    % plot(K_list/P_base, r_ring,  'r--', 'LineWidth', 2, 'DisplayName', 'Ring System');
    
    xlabel('K/P_{base}'); 
    ylabel('r (Order Parameter)');
    
    % Legend placement might need adjustment based on figure appearance
    legend([h_updates(1), h_updates(2), h_updates(3), h_updates(4), h_updates(5), h_updates(6), h_updates(7), h_updates(8), h_updates(9), h_updates(10)],'Location','southwest');
    
    title('Order Parameter (r) vs Coupling Strength (K)');
    grid on;
    hold off;

end
    
% The plotting code for r_full, r_star, r_ring was left in the original snippet 
% but is commented out here since those variables are not calculated in the provided context.

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