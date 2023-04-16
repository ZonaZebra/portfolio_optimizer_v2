# Portfolio Optimizer

## Introduction

This is version 2 of my portfolio optimizer project, which uses a genetic algorithm to find optimal portfolios of stocks. The project was inspired by my interest in finance and investment, and my desire to learn more about optimization algorithms and how they can be applied in practical settings.

## How It Works

- The project consists of several Python scripts that use various libraries to fetch stock data, optimize portfolios using a genetic algorithm, and plot the results. Here's a brief overview of how it works:

- The **`get_stock_data`** function in data_fetcher.py uses the Alpha Vantage API to fetch historical stock prices for a set of symbols and a specified date range.

- The **`ga_solver`** function in ga_solver.py implements a genetic algorithm that generates a population of portfolios, evaluates their fitness using an objective function and a penalty function, and iteratively selects the best portfolios for breeding and mutation.

- The **`optimize_portfolio`** function in ga_solver.py uses the genetic algorithm to find the optimal portfolio for a given level of risk aversion.

- The **`plot_efficient_frontier`** function in plotter.py plots the efficient frontier of all possible portfolios, i.e., the set of portfolios that achieve the highest expected return for a given level of risk.

- The **`plot_pie_charts`** function in plotter.py plots pie charts showing the distribution of stocks in a given portfolio.

## Setup

To run the project, you'll need to set up a Python virtual environment and install the required libraries.

Here are the steps:

- Clone the repository to your local machine.

  ` git clone https://github.com/ZonaZebra/portfolio_optimizer_v2`

- Create a new virtual environment using:

  `python3 -m venv venv`

- Activate the virtual environment using:

  `source venv/bin/activate`

- Install the required libraries using:

  `pip install -r requirements.txt`

- Export your Alpha Vantage API key as an environment variable using

  `export ALPHA_VANTAGE_API_KEY=<your_api_key>`

- Run the main.py script using

  `python3 main.py`

## Conclusion

I hope you find this portfolio optimizer project useful and informative! If you have any questions or feedback, feel free to reach out to me.

## Further explanation of the Genetic Algorithm used in this project:

The GA is a type of optimization algorithm that is inspired by the process of natural selection. It involves creating a population of "candidate solutions" (in this case, portfolios), and then iteratively applying selection, crossover, and mutation operations to evolve the population towards better solutions.

In the case of this project, we start by generating an initial population of portfolios with random weights for each stock. We then evaluate the fitness of each portfolio, which is a measure of how well it satisfies our objective function (maximizing expected return while minimizing risk) and penalty function (penalizing portfolios that don't have a total weight of 1). Next, we select the fittest individuals from the population to "mate" and create new offspring. This selection process is based on fitness proportionate selection, where individuals with higher fitness have a higher probability of being selected. The selected individuals are then combined using crossover, which involves swapping portions of their weight vectors to create new offspring. After creating the new offspring, we apply mutation, which randomly perturbs their weights to add additional diversity to the population. We then evaluate the fitness of the new population and repeat the process until a stopping criteria is met (in this case, a fixed number of iterations). Throughout this process, we maintain a set of "elite" individuals, which are the fittest individuals from each generation. This ensures that we don't lose the best solutions found so far.

Overall, the GA is a powerful optimization algorithm that can be used to find optimal solutions to a wide range of problems, including portfolio optimization.
