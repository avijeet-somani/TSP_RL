import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset
import tsp_generator
import torch.nn as nn
import math
from modules import Attention, GraphEmbedding
from torch.distributions import Categorical
from solver import Solver, solver_LSTM
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.trial import TrialState
import optuna.visualization as vis
import matplotlib.pyplot as plt
import tsp_heuristic
import networkx as nx





def parse_arguments() : 
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, default="rnn")
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_tr_dataset", type=int, default=10000)
    parser.add_argument("--num_te_dataset", type=int, default=200)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_clip", type=float, default=1.5)
    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--train", type=bool, default=False) 
    parser.add_argument("--model_path", type=str, default="") 
    parser.add_argument("--run_dir", type=str, default="tsp_run") 
    parser.add_argument("--hp_trials", type=int, default=12) 
    parser.add_argument("--hp_parallelism", type=int, default=4) 
    parser.add_argument("--heuristic_compare", type=bool, default=False) 
    parser.add_argument("--blank_run", type=bool, default=False)  
    parser.add_argument("--print_graph", type=bool, default=False)  
    args = parser.parse_args()
    return args


def graph_visualize(nodes, order, graph_label='graph') :
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes and edges to the graph based on the order
        for i in range(len(order)-1):
            G.add_edge(order[i], order[i+1])

        # Draw the graph
        #pos = dict(zip(range(len(nodes)), nodes))
        nodes_np = nodes.numpy()
        pos = {i: nodes_np[i] for i in range(len(nodes_np))}
        #print('pos : ', pos)
        nx.draw(G, pos, with_labels=True, arrowsize=20)
        # Add labels with city coordinates
        labels = {i: f"{i}\n({nodes_np[i][0]:.4f}, {nodes_np[i][1]:.4f})" for i in range(len(nodes_np))}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        plt.title(graph_label)
        plt.savefig(graph_label + '.png')  
        plt.show()

def create_unique_directory(directory_name):
    # Check if the original directory already exists
    if not os.path.exists(directory_name):
        # If not, create the original directory
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created.")
        new_directory_name = directory_name
    else:
        # If the original directory exists, find a unique name
        count = 1
        while True:
            new_directory_name = f"{directory_name}_{count}"
            # Check if the new directory name exists
            if not os.path.exists(new_directory_name):
                # If not, create the new directory
                os.makedirs(new_directory_name)
                print(f"Directory '{new_directory_name}' created.")
                break
            else:
                # If the new directory name exists, try the next count
                count += 1
    return new_directory_name


def batch_train(args, train_dataset, model, optimizer) :
    
    beta = args.beta
    grad_clip = args.grad_clip
    num_epochs = args.num_epochs

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        )
    moving_avg = torch.zeros(args.num_tr_dataset)
    

    #generating first baseline
    for (indices, sample_batch) in tqdm(train_data_loader):
        
        rewards, _, _ = model(sample_batch)
        moving_avg[indices] = rewards

    # Train loop
    model.train()
    writer = SummaryWriter()
    for epoch in range(num_epochs):
        for batch_idx, (indices, sample_batch) in enumerate(train_data_loader):
            #print(batch_idx , indices ) 
            rewards, log_probs, action = model(sample_batch)
            #print('log_probs, action , rewards:' ,  log_probs.shape, action.shape , rewards.shape)
            log_probs = torch.sum(log_probs, dim=-1)
            log_probs[log_probs < -100] = -100

            moving_avg[indices] = moving_avg[indices] * beta + rewards * (1.0 - beta)
            #penalize if rewards > moving_avg . if rewards < moving_avg should i make the loss negative or zero ??
            #advantage = rewards - moving_avg[indices]
            #loss = (advantage * log_probs).mean()
            penalty = rewards - moving_avg[indices]
            loss = (penalty * log_probs).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        #print("loss , epoch ", loss , epoch)
        writer.add_scalar("Loss vs epoch", loss, epoch)
        writer.flush()

    writer.close()

def batch_test(test_dataset, model) : 
    model.eval()
    batch_size = len(test_dataset)
    distance = torch.zeros(batch_size)
    tour_list = []
    with torch.no_grad() :
        eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  
        for i, batch in eval_loader:
            R, _, action = model(batch)
            distance[i] = R
            tour_list.append(action)
            
        R_avg = R.mean().detach().numpy()
        
        return R_avg, distance, tour_list

def save_model(model, model_name) : 
    models_dir = os.getcwd() + '/models/' 
    if not os.path.exists(models_dir):
        # If not, create the original directory
        os.makedirs(models_dir)
    models_path = models_dir + model_name
    torch.save(model, models_path )

'''
def load_model( model_name) : 
    models_path = os.getcwd() + '/models/' + model_name
    model = torch.load(models_path )
    return model
'''

def create_datasets(args) : 
    train_dataset = tsp_generator.TSPDataset(args.seq_len, args.num_tr_dataset)
    test_dataset = tsp_generator.TSPDataset(args.seq_len, args.num_te_dataset)
    return train_dataset, test_dataset

    


def TSP_RL(args) : 
    train_dataset, test_dataset = create_datasets(args)
    
    model = solver_LSTM(
            args.embedding_size,
            args.hidden_size,
            args.seq_len,
            2, 10)
   
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0 * 1e-4)

    batch_train(args, train_dataset, model, optimizer)
    return model
    


class HPOptimizer :
    def __init__(self, args) : 
        self.args = args
        self.train_dataset = None
        self.test_dataset = None

    def kickstart(self):
        self.optimize_params()

    def optimize_params(self) : 

        sampler = optuna.samplers.TPESampler(seed=42)
        study_name = "distributed_TSP"
        storage_name = f"sqlite:///{study_name}.db"
        study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage_name, sampler=sampler, load_if_exists=True)
        
        self.train_dataset, self.test_dataset = create_datasets(self.args)

        
        # Start the optimization process with parallel execution.
        n_jobs = self.args.hp_parallelism # Set the number of parallel jobs as needed.
        study.optimize(self.objective, n_trials=self.args.hp_trials, n_jobs=n_jobs)
        #study.optimize(self.objective, n_trials=3, timeout=600)
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])    
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]) 
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        #self.visualize(study)


    def visualize(self, study):
        #visualization     
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig2 = optuna.visualization.plot_param_importances(study)
        #fig3 = optuna.visualization.plot_pareto_front(study)
        fig1.savefig('optimization_history_plot.png')
        fig2.savefig('optimization_param_importance.png')
        #fig3.savefig('optimization_pareto_front.png')
        plt.show()





    def define_model(self, trial) : 
        #model hyperparams : embedding_size, hidden_size
        embedding_size = trial.suggest_categorical("embedding_size", [32, 64,128]) 
        #embedding_size = trial.suggest_int("embedding_size", 32, 64, step=32)
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])

        model = solver_LSTM(embedding_size,hidden_size, self.args.seq_len, 2, 10)
        
        return model
    

    def define_optimizer(self, trial) :
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        learning_rate =  trial.suggest_categorical("learning_rate", [3.0*1e-4, 1e-5, 1e-1])
        return  [ optimizer_name, learning_rate ]
        #optimizer_name = torch.optim.Adam(model.parameters(), lr=3.0 * 1e-4)

    def objective(self, trial) : 
        model = self.define_model(trial)
        optimizer_name, learning_rate = self.define_optimizer(trial)
        optimizer = getattr(torch.optim, optimizer_name) (model.parameters(), lr=learning_rate)
        reward, _, _= batch_test(self.test_dataset, model)
        print("AVG Tour Distance before Training", reward)

        batch_train(self.args,  self.train_dataset, model , optimizer )
        reward, _, _ = batch_test( self.test_dataset, model) #reward is the tour distance
       
        print("AVG Tour Distance after Training", reward)

        model_signature = f'model_trial_{trial.number}.h5'
        save_model(model, model_signature)
        return reward


class ModelVsHeuristics :
    def __init__(self, args, model=None) : 
        self.model = model
       
        _,self.dataset = create_datasets(args)


    def compute_heuristic_distance(self, dataset=None) : 
        #dataset = tsp.TSPDataset(20, 1)
    
        dataset = self.dataset if dataset is None else dataset
        tour_list = []
        heuristic_distance = torch.zeros(len(dataset))
        for i, pointset in tqdm(dataset):
            heuristic_distance[i], tour = tsp_heuristic.get_ref_reward(pointset)
            tour_list.append(tour)
            #print(heuristic_distance[i])
        print(' compute_heuristic_distance : ' , heuristic_distance.mean().detach().numpy())
        return heuristic_distance, tour_list


    def compare(self, model, dataset=None, print_graph=False) : 
        print('compare')
        dataset = self.dataset if dataset is None else dataset
        heuristic_distance, heuristic_tour_list = self.compute_heuristic_distance(dataset)
        R_avg, model_distance, model_tour_list = batch_test(dataset, model)
        
        plt.figure(figsize=(8, 6))
        plt.boxplot([heuristic_distance.numpy(), model_distance.numpy() ], labels=['Heuristic', 'Model'])
        plt.title('Heuristic vs Model')
        plt.ylabel('Tour Distance')
        plt.savefig('TSP_heuristic_vs_model.png')
        plt.show()
        print('Heuristic vs model distances ',   heuristic_distance.mean().detach().numpy() , model_distance.mean().detach().numpy() )
        
        if print_graph : 
            for i, pointset in tqdm(dataset):
                #print(pointset)
                graph_visualize(pointset, heuristic_tour_list[i], graph_label='heuristic based')
                
                mtour = model_tour_list[i][0].tolist()
                mtour.append(mtour[0]) #edge addition was done but the last connection was implicit . adding to make the graph connect back to start-index
                graph_visualize(pointset, mtour,  graph_label='model based')


    



def main() : 
    args = parse_arguments()
    new_directory_name = create_unique_directory(args.run_dir)
    os.chdir(new_directory_name)
    if args.train : 
        hp_opt = HPOptimizer(args)    
        hp_opt.kickstart() 
    elif args.heuristic_compare : 
        print('model path : ', args.model_path)
        assert args.model_path != None
        assert os.path.exists(args.model_path) 
        model = torch.load(args.model_path)
        model_vs_heuristics = ModelVsHeuristics(args)
        model_vs_heuristics.compare(model)   
    elif args.print_graph  : 
        assert args.model_path != None
        assert os.path.exists(args.model_path) 
        dataset = tsp_generator.TSPDataset(args.seq_len, 1)
        model = torch.load(args.model_path)
        model_vs_heuristics = ModelVsHeuristics(args)
        model_vs_heuristics.compare(model, dataset, args.print_graph) 
        
    elif args.blank_run : 
        TSP_RL(args) 
        

    

if __name__ == "__main__":
    main()