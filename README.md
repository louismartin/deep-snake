# Deep Snake
This repo is a adaptation of Deep Policy Networks to play the game of Snake. It has been coded in Python using Tensorflow. Have fun !

1. [Setup](#setup)

2. [Playing around](#playing-around)

3. [Going further](#going-further)
 
## Setup 

You can find the requirements in the file __requirements.txt__. All coding files can be found in the __snake__ folder.

## Playing around

We implemented a simple snake envioronment (__snake.py__). To play it yourself and try to beat the best AI score, simply run 
```
python demo_snake.py
```

We also implemented the policy network algorithm as described in our report in the file __policy_gradient.py__. You can try your own network structure (folder __models__). The pretrained weights are avilable in the __weights__ folder (there is a warm_restart argument to the training function) and the output graphs are available in the __graphs__ folder. To try our implementation, simply run
```
python demo_policy.py
```

## Going further

The detailed theoretical explanations, the setup parameters and experimental results can be found in the __report__ folder.  
