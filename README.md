# Genetic Algorithm

Simple implementation of a genetic algorithm applied into gym environments


## Getting Started
This implementation uses a jupyter machine inside a container, so to build and run the container just:
```
./run_notebook.sh

# Or manually
docker build -t ga-notebook .
docker run --rm -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v "$PWD":/home/jovyan/work ga-notebook
```

## Genetic Algorithm implementation

This genetic algorithm has some parameters that can be customized to explore different scenarios:

* population_size: number of genomes you'll have for each generation
* generations: Number of generations you'll run through
* mutation_variance: Percentage of mutation in each weight (so if 0.2 the weight has a probability of 20% of mutating)
* survival_ratio: Keeping the best <survival_ratio> genomes (if 0.1, only the best 0.1 genomes survive for the next generation)
* both_parent_percentage: Percentage of genomes that will come from crossover between two parents
* one_parent_percentage: Percentage of genomes that will come from one parent only (will not suffer from crossover, only from mutation)
* episodes: how many episodes it will run before extracting the fitness of the genomes
