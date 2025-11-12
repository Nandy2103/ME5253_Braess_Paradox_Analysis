# ME5253

This repository contains all the relevant matlab files.

The original paper results could be viewed by running the file braess_paradox_original_paper.m
A validation model using a 10 node system is implemented in braess_paradox_10node.m

To understand the effect of K_c on various topologies including fully connected system, run braess_paradox_topology.m

And for improvement paper simulations on understanding dysfunctional nodes in the original 8 node setup, run "braess_paradox_addcap_optimized.m"
And to check in 10 node setup, run "braess_paradox_addcap_optimized10.m"

And for any random node configuration, create a csv file similar to "nodes_input_withP.csv", and run "braess_paradox_addcap_from_csv_indiff_v8_3.m"
Sample configurations of csv setups are given already to test the code

To replicate our output, please make changes to the "braess_paradox_addcap_from_csv_indiff_v8_3.m" code
- 8 node system - nodes_input_withP.csv
<img width="608" height="729" alt="image" src="https://github.com/user-attachments/assets/d39a51b7-1f8a-41b2-bc61-56d5e4941db5" />

- 10 node systme - nodes_input_withP_check_16.csv

<img width="630" height="736" alt="image" src="https://github.com/user-attachments/assets/6153f53a-994e-4515-94de-e77ad2773888" />
