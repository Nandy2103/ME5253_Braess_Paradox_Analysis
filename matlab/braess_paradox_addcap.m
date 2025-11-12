function braess_paradox_addcap()
  
    % Network Parameters
    N = 10;                      % Number of nodes
    P_base = 1.0;               % Base power
    
    % Parameters from Figure 1 caption:
    K_uniform = 1.03 * P_base;  % K>Kc
    alpha = P_base;             % Damping coefficient
    
    % Node types (Producers: +P, Consumers: -P)
    producers = [2, 3, 4, 7, 9]; % Nodes with +P
    consumers = [1, 5, 6, 8, 10]; % Nodes with -P
    
    P_vec = zeros(N, 1);
    P_vec(producers) = P_base;
    P_vec(consumers) = -P_base;
    
    % Adjacency Matrices and Coupling Strengths
    
    % --- Network 1: Original Configuration (Figure 1a) ---
    A_original = zeros(N, N);
    % Edges from careful tracing of Figure 1a (11 edges total)
    % (1,2), (1,6), (1,7), (2,3), (3,4), (3,7), (4,8), (4,5), (5,6), (6,8),
    edges_orig = [1,2; 1,6; 1,7; 2,3; 3,4; 3,7; 4,8; 4,5; 5,6; 6,8; 5,9; 7,10];
    for i = 1:size(edges_orig, 1)
        n1 = edges_orig(i, 1);
        n2 = edges_orig(i, 2);
        A_original(n1, n2) = 1;
        A_original(n2, n1) = 1;
    end
    K_matrix_original = K_uniform * A_original; % Uniform coupling K for all edges
                  
    % --- Network 2: "Add capacity" (Figure 1b) ---
    update = 0.2 ;
    % Doubles capacity of edge (3,4)
    A_add_capacity = A_original; % Topology is the same
    K_matrix_add_capacity1 = K_uniform * A_add_capacity; % Start with uniform couplings
    K_matrix_add_capacity1(3, 4) = update * K_uniform; % Double K for edge (3,4)
    K_matrix_add_capacity1(4, 3) = update * K_uniform; % Symmetric

    % --- Network 2: "Add capacity" (Figure 1b) ---
    A_add_capacity = A_original; % Topology is the same
    K_matrix_add_capacity2 = K_uniform * A_add_capacity;
    K_matrix_add_capacity2(1, 2) = update * K_uniform;
    K_matrix_add_capacity2(2, 1) = update * K_uniform;

    % --- Network 2: "Add capacity" (Figure 1b) ---
    A_add_capacity = A_original; % Topology is the same
    K_matrix_add_capacity3 = K_uniform * A_add_capacity;
    K_matrix_add_capacity3(1, 6) = update * K_uniform;
    K_matrix_add_capacity3(6, 1) = update * K_uniform;
    
    % --- Network 2: "Add capacity" (Figure 1b) ---
    A_add_capacity = A_original; % Topology is the same
    K_matrix_add_capacity4 = K_uniform * A_add_capacity;
    K_matrix_add_capacity4(1, 7) = update * K_uniform;
    K_matrix_add_capacity4(7, 1) = update * K_uniform;

    % --- Network 2: "Add capacity" (Figure 1b) ---
    A_add_capacity = A_original; % Topology is the same
    K_matrix_add_capacity5 = K_uniform * A_add_capacity;
    K_matrix_add_capacity5(3, 2) = update * K_uniform;
    K_matrix_add_capacity5(2, 3) = update * K_uniform;

    % --- Network 2: "Add capacity" (Figure 1b) ---
    A_add_capacity = A_original; % Topology is the same
    K_matrix_add_capacity6 = K_uniform * A_add_capacity;
    K_matrix_add_capacity6(3, 7) = update * K_uniform;
    K_matrix_add_capacity6(7, 3) = update * K_uniform;

    % --- Network 2: "Add capacity" (Figure 1b) ---
    A_add_capacity = A_original; % Topology is the same
    K_matrix_add_capacity7 = K_uniform * A_add_capacity;
    K_matrix_add_capacity7(4, 8) = update * K_uniform;
    K_matrix_add_capacity7(8, 4) = update * K_uniform;
    
    % --- Network 2: "Add capacity" (Figure 1b) ---
    A_add_capacity = A_original; % Topology is the same
    K_matrix_add_capacity8 = K_uniform * A_add_capacity;
    K_matrix_add_capacity8(4, 5) = update * K_uniform;
    K_matrix_add_capacity8(5, 4) = update * K_uniform;

    % --- Network 2: "Add capacity" (Figure 1b) ---
    A_add_capacity = A_original; % Topology is the same
    K_matrix_add_capacity9 = K_uniform * A_add_capacity;
    K_matrix_add_capacity9(5, 6) = update * K_uniform;
    K_matrix_add_capacity9(6, 5) = update * K_uniform;

    % --- Network 2: "Add capacity" (Figure 1b) ---
    A_add_capacity = A_original; % Topology is the same
    K_matrix_add_capacity10 = K_uniform * A_add_capacity;
    K_matrix_add_capacity10(6, 8) = update * K_uniform;
    K_matrix_add_capacity10(8, 6) = update * K_uniform;

    % --- Network 2: "Add capacity" (Figure 1b) ---
    A_add_capacity = A_original; % Topology is the same
    K_matrix_add_capacity11 = K_uniform * A_add_capacity;
    K_matrix_add_capacity11(5, 9) = update * K_uniform;
    K_matrix_add_capacity11(9, 5) = update * K_uniform;

    % --- Network 2: "Add capacity" (Figure 1b) ---
    A_add_capacity = A_original; % Topology is the same
    K_matrix_add_capacity12 = K_uniform * A_add_capacity;
    K_matrix_add_capacity12(7, 10) = update * K_uniform;
    K_matrix_add_capacity12(10, 7) = update * K_uniform;

    
    % --- 3. Simulation Parameters ---
    % tspan corresponds to alpha*t up to ~25, so t goes to 25 given alpha=P_base=1
    tspan = [0 50];
    
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
    
    % Node numbers are 1-8. These coordinates map node ID to its visual position.
    % Node IDs:   1    2    3    4    5    6    7    8  9  10
    x_coords = [6.0, 5.0, 6.0, 3.0, 2.0, 3.0, 7.0, 4.0, 1.0, 8.0]; % Corrected x-coordinates
    y_coords = [2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 3.0, 3.0, 3.0, 3.0]; % Corrected y-coordinates

    % Create Graph objects for visualization
    G_original = graph(A_original, node_labels, 'omitselfloops');
    G_add_capacity = graph(A_add_capacity, node_labels, 'omitselfloops');
    
    figure('Name', 'Network Topologies (Figure 1a, 1b, 1c)', 'NumberTitle', 'off', 'Position', [50, 500, 1200, 400]);
    

    % Subplot 2: Add Capacity (Fig. 1b)
    subplot(2, 6, 1);
    p2 = plot(G_add_capacity, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 2);
    title('b Add capacity');
    customize_plot(p2, producers, consumers, [3,4], [0.1 0.1 0.9]); % Highlight edge (3,4) in blue
    
    subplot(2, 6, 2);
    p2 = plot(G_add_capacity, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 2);
    title('b Add capacity');
    customize_plot(p2, producers, consumers, [1,2], [0.1 0.1 0.9]);

    subplot(2, 6, 3);
    p2 = plot(G_add_capacity, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 2);
    title('b Add capacity');
    customize_plot(p2, producers, consumers, [1,6], [0.1 0.1 0.9]);

    subplot(2, 6, 4);
    p2 = plot(G_add_capacity, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 2);
    title('b Add capacity');
    customize_plot(p2, producers, consumers, [1,7], [0.1 0.1 0.9]);

    subplot(2, 6, 5);
    p2 = plot(G_add_capacity, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 2);
    title('b Add capacity');
    customize_plot(p2, producers, consumers, [3,2], [0.1 0.1 0.9]);

    subplot(2, 6, 6);
    p2 = plot(G_add_capacity, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 2);
    title('b Add capacity');
    customize_plot(p2, producers, consumers, [3,7], [0.1 0.1 0.9]);

    subplot(2, 6, 7);
    p2 = plot(G_add_capacity, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 2);
    title('b Add capacity');
    customize_plot(p2, producers, consumers, [4,8], [0.1 0.1 0.9]);

    subplot(2, 6, 8);
    p2 = plot(G_add_capacity, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 2);
    title('b Add capacity');
    customize_plot(p2, producers, consumers, [4,5], [0.1 0.1 0.9]);

    subplot(2, 6, 9);
    p2 = plot(G_add_capacity, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 2);
    title('b Add capacity');
    customize_plot(p2, producers, consumers, [5,6], [0.1 0.1 0.9]);

    subplot(2, 6, 10);
    p2 = plot(G_add_capacity, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 2);
    title('b Add capacity');
    customize_plot(p2, producers, consumers, [6,8], [0.1 0.1 0.9]);

    subplot(2, 6, 11);
    p2 = plot(G_add_capacity, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 2);
    title('b Add capacity');
    customize_plot(p2, producers, consumers, [5,9], [0.1 0.1 0.9]);

    subplot(2, 6, 12);
    p2 = plot(G_add_capacity, 'XData', x_coords, 'YData', y_coords, 'LineWidth', 2);
    title('b Add capacity');
    customize_plot(p2, producers, consumers, [7,10], [0.1 0.1 0.9]);

    % --- 4. Run Simulations ---

    fprintf('Simulating "add capacity" network (Fig. 1e)...\n');
    [t_cap1, y_cap1] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_add_capacity1, alpha), tspan, y0, options);
    fprintf('"Add capacity" network simulation complete.\n');

    fprintf('Simulating "add capacity" network (Fig. 1e)...\n');
    [t_cap2, y_cap2] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_add_capacity2, alpha), tspan, y0, options);
    fprintf('"Add capacity" network simulation complete.\n');

    fprintf('Simulating "add capacity" network (Fig. 1e)...\n');
    [t_cap3, y_cap3] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_add_capacity3, alpha), tspan, y0, options);
    fprintf('"Add capacity" network simulation complete.\n');

    fprintf('Simulating "add capacity" network (Fig. 1e)...\n');
    [t_cap4, y_cap4] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_add_capacity4, alpha), tspan, y0, options);
    fprintf('"Add capacity" network simulation complete.\n');

    fprintf('Simulating "add capacity" network (Fig. 1e)...\n');
    [t_cap5, y_cap5] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_add_capacity5, alpha), tspan, y0, options);
    fprintf('"Add capacity" network simulation complete.\n');

    fprintf('Simulating "add capacity" network (Fig. 1e)...\n');
    [t_cap6, y_cap6] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_add_capacity6, alpha), tspan, y0, options);
    fprintf('"Add capacity" network simulation complete.\n');

    fprintf('Simulating "add capacity" network (Fig. 1e)...\n');
    [t_cap7, y_cap7] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_add_capacity7, alpha), tspan, y0, options);
    fprintf('"Add capacity" network simulation complete.\n');

    fprintf('Simulating "add capacity" network (Fig. 1e)...\n');
    [t_cap8, y_cap8] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_add_capacity8, alpha), tspan, y0, options);
    fprintf('"Add capacity" network simulation complete.\n');

    fprintf('Simulating "add capacity" network (Fig. 1e)...\n');
    [t_cap9, y_cap9] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_add_capacity9, alpha), tspan, y0, options);
    fprintf('"Add capacity" network simulation complete.\n');

    fprintf('Simulating "add capacity" network (Fig. 1e)...\n');
    [t_cap10, y_cap10] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_add_capacity10, alpha), tspan, y0, options);
    fprintf('"Add capacity" network simulation complete.\n');

    fprintf('Simulating "add capacity" network (Fig. 1e)...\n');
    [t_cap11, y_cap11] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_add_capacity11, alpha), tspan, y0, options);
    fprintf('"Add capacity" network simulation complete.\n');

    fprintf('Simulating "add capacity" network (Fig. 1e)...\n');
    [t_cap12, y_cap12] = ode45(@(t, y) second_order_ode(t, y, P_vec, K_matrix_add_capacity12, alpha), tspan, y0, options);
    fprintf('"Add capacity" network simulation complete.\n');


    % --- 5. Analyze and Plot Results (Phases) ---
    fprintf('Plotting phase dynamics...\n');
    
    % Extract phases and convert to phi/pi
    phi_pi_cap1 = y_cap1(:, 1:N) / pi;
    phi_pi_cap2 = y_cap2(:, 1:N) / pi;
    phi_pi_cap3 = y_cap3(:, 1:N) / pi;
    phi_pi_cap4 = y_cap4(:, 1:N) / pi;
    phi_pi_cap5 = y_cap5(:, 1:N) / pi;
    phi_pi_cap6 = y_cap6(:, 1:N) / pi;
    phi_pi_cap7 = y_cap7(:, 1:N) / pi;
    phi_pi_cap8 = y_cap8(:, 1:N) / pi;
    phi_pi_cap9 = y_cap9(:, 1:N) / pi;
    phi_pi_cap10 = y_cap10(:, 1:N) / pi;
    phi_pi_cap11 = y_cap11(:, 1:N) / pi;
    phi_pi_cap12 = y_cap12(:, 1:N) / pi;

    
    figure('Name', 'Simulation Results (Figure 1d, 1e, 1f)', 'NumberTitle', 'off', 'Position', [50, 50, 1200, 700]);
    
    % Common Y-axis limits for all plots to match paper
    y_lims = [-3.5, 3.5]; 
    
    % --- Subplot 1: Original Network (Sync) - Fig 1d ---
    subplot(2, 6, 1);
    hold on;
    h_orig_prod = plot(t_cap1, phi_pi_cap1(:, producers), 'r-', 'LineWidth', 1.5);
    h_orig_cons = plot(t_cap1, phi_pi_cap1(:, consumers), 'g-', 'LineWidth', 1.5);
    hold off;
    xlabel('\alpha t');
    ylabel('\phi_j(t) / \pi');
    grid on;
    ylim(y_lims);
    legend([h_orig_prod(1), h_orig_cons(1)], {'Producers', 'Consumers'}, 'Location', 'southeast', 'AutoUpdate', 'off');
    
    subplot(2, 6, 2);
    hold on;
    h_cap_prod = plot(t_cap2, phi_pi_cap2(:, producers), 'r-', 'LineWidth', 1.5);
    h_cap_cons = plot(t_cap2, phi_pi_cap2(:, consumers), 'g-', 'LineWidth', 1.5);
    hold off;
    xlabel('\alpha t');
    ylabel('\phi_j(t) / \pi');
    grid on;
    ylim(y_lims);
    
    subplot(2, 6, 3);
    hold on;
    h_orig_prod = plot(t_cap3, phi_pi_cap3(:, producers), 'r-', 'LineWidth', 1.5);
    h_orig_cons = plot(t_cap3, phi_pi_cap3(:, consumers), 'g-', 'LineWidth', 1.5);
    hold off;
    xlabel('\alpha t');
    ylabel('\phi_j(t) / \pi');
    grid on;
    ylim(y_lims);
    legend([h_orig_prod(1), h_orig_cons(1)], {'Producers', 'Consumers'}, 'Location', 'southeast', 'AutoUpdate', 'off');
    
    subplot(2, 6, 4);
    hold on;
    h_cap_prod = plot(t_cap4, phi_pi_cap4(:, producers), 'r-', 'LineWidth', 1.5);
    h_cap_cons = plot(t_cap4, phi_pi_cap4(:, consumers), 'g-', 'LineWidth', 1.5);
    hold off;
    xlabel('\alpha t');
    ylabel('\phi_j(t) / \pi');
    grid on;
    ylim(y_lims);

    subplot(2, 6, 5);
    hold on;
    h_orig_prod = plot(t_cap5, phi_pi_cap5(:, producers), 'r-', 'LineWidth', 1.5);
    h_orig_cons = plot(t_cap5, phi_pi_cap5(:, consumers), 'g-', 'LineWidth', 1.5);
    hold off;
    xlabel('\alpha t');
    ylabel('\phi_j(t) / \pi');
    grid on;
    ylim(y_lims);
    legend([h_orig_prod(1), h_orig_cons(1)], {'Producers', 'Consumers'}, 'Location', 'southeast', 'AutoUpdate', 'off');
    
    subplot(2, 6, 6);
    hold on;
    h_cap_prod = plot(t_cap6, phi_pi_cap6(:, producers), 'r-', 'LineWidth', 1.5);
    h_cap_cons = plot(t_cap6, phi_pi_cap6(:, consumers), 'g-', 'LineWidth', 1.5);
    hold off;
    xlabel('\alpha t');
    ylabel('\phi_j(t) / \pi');
    grid on;
    ylim(y_lims);

    subplot(2, 6, 7);
    hold on;
    h_orig_prod = plot(t_cap7, phi_pi_cap7(:, producers), 'r-', 'LineWidth', 1.5);
    h_orig_cons = plot(t_cap7, phi_pi_cap7(:, consumers), 'g-', 'LineWidth', 1.5);
    hold off;
    xlabel('\alpha t');
    ylabel('\phi_j(t) / \pi');
    grid on;
    ylim(y_lims);
    legend([h_orig_prod(1), h_orig_cons(1)], {'Producers', 'Consumers'}, 'Location', 'southeast', 'AutoUpdate', 'off');

    subplot(2, 6, 8);
    hold on;
    h_cap_prod = plot(t_cap8, phi_pi_cap8(:, producers), 'r-', 'LineWidth', 1.5);
    h_cap_cons = plot(t_cap8, phi_pi_cap8(:, consumers), 'g-', 'LineWidth', 1.5);
    hold off;
    xlabel('\alpha t');
    ylabel('\phi_j(t) / \pi');
    grid on;
    ylim(y_lims);

    subplot(2, 6, 9);
    hold on;
    h_orig_prod = plot(t_cap9, phi_pi_cap9(:, producers), 'r-', 'LineWidth', 1.5);
    h_orig_cons = plot(t_cap9, phi_pi_cap9(:, consumers), 'g-', 'LineWidth', 1.5);
    hold off;
    xlabel('\alpha t');
    ylabel('\phi_j(t) / \pi');
    grid on;
    ylim(y_lims);
    legend([h_orig_prod(1), h_orig_cons(1)], {'Producers', 'Consumers'}, 'Location', 'southeast', 'AutoUpdate', 'off');

    subplot(2, 6, 10);
    hold on;
    h_cap_prod = plot(t_cap10, phi_pi_cap10(:, producers), 'r-', 'LineWidth', 1.5);
    h_cap_cons = plot(t_cap10, phi_pi_cap10(:, consumers), 'g-', 'LineWidth', 1.5);
    hold off;
    xlabel('\alpha t');
    ylabel('\phi_j(t) / \pi');
    grid on;
    ylim(y_lims);

    subplot(2, 6, 11);
    hold on;
    h_orig_prod = plot(t_cap11, phi_pi_cap11(:, producers), 'r-', 'LineWidth', 1.5);
    h_orig_cons = plot(t_cap11, phi_pi_cap11(:, consumers), 'g-', 'LineWidth', 1.5);
    hold off;
    xlabel('\alpha t');
    ylabel('\phi_j(t) / \pi');
    grid on;
    ylim(y_lims);
    legend([h_orig_prod(1), h_orig_cons(1)], {'Producers', 'Consumers'}, 'Location', 'southeast', 'AutoUpdate', 'off');

    subplot(2, 6, 12);
    hold on;
    h_cap_prod = plot(t_cap12, phi_pi_cap12(:, producers), 'r-', 'LineWidth', 1.5);
    h_cap_cons = plot(t_cap12, phi_pi_cap12(:, consumers), 'g-', 'LineWidth', 1.5);
    hold off;
    xlabel('\alpha t');
    ylabel('\phi_j(t) / \pi');
    grid on;
    ylim(y_lims);
    
    fprintf('Done.\n');

  % === 2c: Order parameter r vs K ===
    fprintf('Sweeping coupling strength K for Fig. 2c...\n');
    
    % --- Setup ---
    K_list = linspace(0.0, 1.4, 25) * P_base; % K values to sweep
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
            Kmat_update(n1, n2) = update * K_val;
            Kmat_update(n2, n1) = update * K_val;
            
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
    legend([h_updates(1), h_updates(2), h_updates(3), h_updates(4), h_updates(5), h_updates(6), h_updates(7), h_updates(8), h_updates(9), h_updates(10), h_updates(11), h_updates(12)],'Location','southwest');
    
    title('Order Parameter (r) vs Coupling Strength (K)');
    grid on;
    hold off;
    
% The plotting code for r_full, r_star, r_ring was left in the original snippet 
% but is commented out here since those variables are not calculated in the provided context.
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

