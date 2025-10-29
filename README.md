## Social Learning Strategies for Virtual Soft Robots using Bayesian Optimization

Optimizing the body and brain of a robot is a coupled challenge: the morphology determines what control strategies are effective, while the control parameters influence how well the morphology performs. 
    This joint optimization can be done through nested loops of evolutionary and learning processes, where the control parameters of each robot are learned independently. 
    However, the control parameters discovered, i.e., learned, by one robot may contain valuable information for others. 
    Thus, we introduce a social learning approach in which robots can exploit optimized parameters from their peers to accelerate their own brain optimization, via Bayesian optimization and sample inheritance.
    Within this framework, we systematically investigate how teachers selection - deciding which and how many robots to learn from - affects performance, experimenting with virtual soft robots across four tasks and environments. 
    In particular, we study the effect of inheriting experience from morphologically similar robots due the tightly coupled body and brain in robot optimization. Our results confirm the effectiveness of re-using optimized samples, as social learning clearly outperforms learning from scratch under equivalent computational budgets. 
    In addition, while the optimal teacher selection strategy remains open, our findings suggest that incorporating knowledge from multiple teachers can yield more consistent and robust improvements.