# DilloNN_1
Neural network builder/trainer. I wrote this simply to explore the algorithm more thouroughly, and gain a better apreciation for what's at play behind the popular modules. Unfinished, but currently functional. Currently cannot have multiple output nodes, and the output node it does have must be a linear activation (This goes against the guide I used, but I wanted the network to predict numbers, not categories). Used https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/ as reference, but all code is original. 

Version 2 can have multiple output nodes, and an error in calculating bias was fixed.  

Version 3 handles all elements in arrays rather than individually. Thhis boosts overall speed by a factor of how ever many elements are being handled. 
Output nodes are still exclusively linear.
