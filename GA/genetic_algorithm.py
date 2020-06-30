import numpy as np
from .neural_net import NeuralNet
from utils.networks import tfSummary
from tqdm import tqdm
import gym
import random

from multiprocessing import Pool
from utils.continuous_environments import Environment

# Defining our class that handles populations of networks
class GeneticNetworks():
    
    # Defining our initialization method
    def __init__(self, 
                architecture=(4,16,2),
                population_size=50,
                generations=500,
                render_env=True,
                record=False,
                mutation_variance=0.02,
                verbose=False,
                print_every=1,
                survival_ratio=0.1,
                both_parent_percentage=0.8,
                one_parent_percentage=0.1,
                episodes=10, 
                stagnation_end=True,
                save_path='saved_models/'
        ):
        # Creating our list of networks
        self.networks = [NeuralNet(architecture) for _ in range(population_size)]
        self.population_size = population_size
        self.generations = generations
        self.mutation_variance = mutation_variance
        self.verbose = verbose
        self.print_every = print_every
        self.fitness = []
        self.episodes = episodes
        self.render_env = render_env
        self.record = record
        self.survival_ratio = survival_ratio
        self.save_path = save_path
        self.both_parent_percentage = both_parent_percentage
        self.one_parent_percentage = one_parent_percentage if one_parent_percentage else 1.0 - both_parent_percentage
        self.stagnation_end = stagnation_end

        self.global_info = []

        print("-" * 50)
        print("Logging params: ")
        print('population_size', self.population_size)
        print('generations', self.generations)
        print('mutation_variance', self.mutation_variance)
        print('survival_ratio', self.survival_ratio)
        print('both_parent_percentage', self.both_parent_percentage)
        print('one_parent_percentage', self.one_parent_percentage)
        print('episodes', self.episodes)
        print("-" * 50)
        
    # Defining our fiting method
    def fit(self, env, summary_writer, debug=False, num_cpus=4, env_args={}, test_env_args=None):
        stagnation = 1
        best_so_far = 0
        test_env = Environment(**env_args) if test_env_args else None

        # Create environements for all population
        envs = [Environment(**env_args) for i in range(self.population_size)]

        # Iterating over all generations
        tqdm_e = tqdm(total=self.generations, desc='Generation', leave=True, unit=" gen")
        for gen_i in range(self.generations):

            # Doing our evaluations
            args = [(self, self.networks[i], envs[i]) for i in range(self.population_size)]
            with Pool(num_cpus) as p:
                rewards = np.array(p.map(_run_par_evaluate, args))


            # Tracking best score per generation
            self.fitness.append(np.max(rewards))

            # Selecting the best network
            best_network = np.argmax(rewards)

            # Selecting top n networks
            n = int(self.survival_ratio * self.population_size)
            top_n_index = np.argsort(rewards)[-n:]

            # Creating our child networks
            new_networks = []
            for _ in range(self.population_size - n):
                # Origin will take -> 0 if both parent -> 1 if one parent and -> 2 if just get another network from previous run
                origin = np.random.choice([0,1,2], p=[self.both_parent_percentage, self.one_parent_percentage, 1 - self.both_parent_percentage - self.one_parent_percentage])

                # both parents
                if origin == 0:
                    new_net = NeuralNet(
                        parent1=self.networks[random.randint(0, len(top_n_index) - 1)], 
                        parent2=self.networks[random.randint(0, len(top_n_index) - 1)], 
                        var=self.mutation_variance
                    )
                # One parent
                elif origin == 1:
                    new_net = NeuralNet(
                        parent1=self.networks[random.randint(0, len(top_n_index) - 1)], 
                        parent2=None, 
                        var=self.mutation_variance
                    )
                else:
                    # Copy from other run (aside from the choosen best)
                    index = top_n_index[0]
                    while index not in top_n_index:
                        index = random.randint(0, len(self.networks) - 1)
                    new_net = self.networks[index]


                new_networks.append(new_net)

            # Setting our new networks
            maintain_best_n = [self.networks[i] for i in top_n_index]
            self.networks = maintain_best_n + new_networks
            

            # Export results for Tensorboard
            r_max = rewards.max()
            r_mean = rewards.mean()
            r_std = rewards.std()
            self.insert_info(r_max, r_mean, r_std)
            summary_writer.add_summary(tfSummary('Max rewards', r_max), global_step=gen_i)
            summary_writer.add_summary(tfSummary('Mean rewards', r_mean), global_step=gen_i)
            summary_writer.add_summary(tfSummary('STD rewards', r_std), global_step=gen_i)

            # Update stagnation
            if r_max > best_so_far:
                best_so_far = r_max
                stagnation = 1
            else: stagnation += 1

            #Update tqdm
            tqdm_e.set_description(
                'Generation:' + str(gen_i + 1) + '| Highest Reward:' + str(r_max) + '| Average Reward:' + str(r_mean) + '| std Reward: ' + str(r_std) + '| Stagnation: ' + str(stagnation) + '| Population size: ' + str(len(self.networks))
            )


            # Save current weights
            self.best_network = self.networks[best_network]
            self.save_weights(gen_i, maintain_best_n[0], self.save_path)

            # Update logs
            summary_writer.flush()
            tqdm_e.update(1)
            tqdm_e.refresh()

            # Se estiver estagnado por muito tempo, eu paro
            if stagnation > 10 and self.stagnation_end: break



        # Close the environments
        [e.close() for e in envs]
        
        # Returning the best network
        self.best_network = self.networks[best_network]

        return self.global_info


    def insert_info(self, max, mean, std):
        self.global_info.append({
            'max': max,
            'mean': mean,
            'std': std
        })

    def load_weights(self, path):
        self.networks[0].load(path)
        self.best_network = self.networks[0]

    def save_weights(self, gen, last_best_net, init_path='saved_models/'):
        path = init_path + 'DeepNeuro_Gen_best_{}'.format(gen + 1)
        self.best_network.save(path)
        path = init_path + 'DeepNeuro_Gen_last_{}'.format(gen + 1)
        last_best_net.save(path)


def _run_par_evaluate(args):
    '''
    Evaluate parallel envs
    '''
    ga, network, env = args
    return network.evaluate(env, ga.episodes, ga.render_env, ga.record, debug=False)