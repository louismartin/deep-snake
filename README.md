# Deep Snake
This repo is a adaptation of Deep Policy Networks to play the game of Snake. It has been coded in Python using Tensorflow. The detailed theoretical explanationss, the setup parameters and experimental results can be found in the __report__ folder. 

1. [Setup](#setup)

2. [Playing around](#playing-around)

3. [Going further](#going-further)
 
## Setup 

You can find the requirements in the file __requirements.txt__. 

## Playing around

The algorithm is implemented in Python 3.5 using Tensorflow 0.11 in the file __compute_distance.py__. If you wish to run the __experiments.py__ file containing all the experiments made during this project, you will have to either 

* Use the precomputed cost matrix __C_most_common_1000_2__ and the associated keys __keys_most_common_1000_2__
* Compute it yourself by chosing the number of words to include and the order of the norm in the embedding space by using the cost_matrix function defined in __compute_cost_matrix.py__ 

The experiments may take time to run (especially on a CPU), I ran them on a NVIDIA K80 using AWS.

## Going further

If you are interested in this topic, you can read the full pdf report (__report__ folder) that details some theoretical aspects, the methodology and the experimental results on large datasets. A bibliography is included if you want to dig deeper on the theoretical side. 
