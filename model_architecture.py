import graphviz

# Specify the number of input nodes (cities)
num_input_nodes = 4  # You can change this based on your actual number of input nodes

# Define the DOT representation of the updated model with parallel encoder outputs
dot_content = f"""
digraph sequence_to_sequence_model {{
    rankdir=BT; // Left to right direction
    compound = True;
    
    
    subgraph cluster_input {{
        label="Input Nodes\n(e.g., Cities)";
        style=dotted;
        node [shape=circle, width=0.3, fixedsize=true]; // Set smaller circle size
      
        {" ".join([f"city_{i};" for i in range(num_input_nodes)])}
    }}

    subgraph cluster_embedding {{
        label="Embedding Layer";
        style=dotted;
        node [shape=box];
        
        embedded [label="Embedding"];
        node [shape=circle, width=0.3, fixedsize=true]; // Set smaller circle size
      
        {" ".join([f"z_{i};" for i in range(num_input_nodes)])}
        {" ".join([f"embedded -> z_{i};" for i in range(num_input_nodes)])}
    }}

    
    subgraph cluster_encoder {{
        label="Encoder Layer";
        style=dotted;
        node [shape=box];
        encoder [label="LSTM Encoder"];
        node [shape=circle, width=0.3, fixedsize=true]; // Set smaller circle size
        {" ".join([f"ref_{i};" for i in range(num_input_nodes)])}
       
        {" ".join([f"encoder -> ref_{i};" for i in range(num_input_nodes)])}
    }}

  
    subgraph cluster_decoder {{
        label="Decoder Layer";
        style=dotted;
        node [shape=box];
        
        decoder [label="LSTM Decoder"];
        query [label="Query"]
        pointer [label="Attention"];
        attention_weights [label="Logits/AttentionWeights"]
        mask [label="Mask"]
       
        #sampled_node [label="Sampled Node/Action"];
        decoder -> query -> pointer -> attention_weights -> mask;
    }}

    subgraph cluster_action_outputs {{
        label="Action Probability";
        style=dotted;
        #node [shape=box];
        node [shape=box, width=0.3, fixedsize=true]; // Set smaller circle size
        {" ".join([f"p{i};" for i in range(num_input_nodes)])}
    }}

    subgraph cluster_action_sampling {{
        #label="Smapled Action"
        style=dotted;
        action_sampling [ shape=box, label="action sampling"]; 
    }}

    subgraph cluster_action_chosen {{
        #label="Smapled Action"
        style=dotted;
        action_chosen [ shape=box, label="chosen city"]; 
    }}
    
    #sampled_node_embedding [shape=box, label="Sampled City Embedding"]
    #decoder_input[label="selected node embedding"]

    #connections
    city_0 -> embedded [ltail=cluster_input lhead=cluster_embedding]
    z_0 -> encoder [ltail=cluster_embedding lhead=cluster_encoder]
    
    z_0 -> decoder [ltail=cluster_embedding lhead=cluster_decoder]
    mask -> p0 [ltail=cluster_decoder lhead=cluster_action_outputs]
    p0 -> action_sampling [ltail=cluster_action_outputs lhead=cluster_action_sampling]
    
    action_sampling -> action_chosen [ltail=cluster_action_sampling lhead=cluster_action_chosen]
    action_chosen -> z_0 [lhead=cluster_embedding ltail=cluster_action_chosen]

    
    {" ".join([f"ref_{i} -> pointer;" for i in range(num_input_nodes)])}
   
   
    
}}
"""

# Create a Graphviz object
graph_sequence_to_sequence_parallel = graphviz.Source(dot_content)

# Save DOT file (optional)
graph_sequence_to_sequence_parallel.render(filename='model_architecture', format='png', cleanup=True)

# Display the graph in the notebook
graph_sequence_to_sequence_parallel
