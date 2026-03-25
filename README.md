## Lamarckian Inheritance in Dynamic Environments: How Two Important Factors Influence Performance for Evolvable Robots

Optimizing the body and brain of a robot is a coupled challenge: the morphology determines what control strategies are effective, while the control parameters influence how well the morphology performs.
    To address this, we combine morphology optimization as evolution with controller optimization as lifetime learning, utilizing Lamarckian inheritance to transfer learned controller parameters from parent to offspring.
    In dynamic environments, existing literature presents conflicting evidence: while traditional evolutionary theory often suggests Lamarckian inheritance lacks benefit, recent studies in evolutionary robotics indicate it can improve performance.
    We hypothesize that this is because previous works have not included all relevant variables with dynamic environments.
    In this work, we show that the benefit of Lamarckian inheritance depends on two factors: how conflicting the environmental changes are to robot control, and the predictability of those changes for the robotic agent.
    Using virtual soft robots and two different learning approaches, Bayesian optimization and reinforcement learning, we show that Lamarckian inheritance only underperforms Darwinian inheritance when the changes are both conflicting and unpredictable.
    We find that adding a sensor to detect environmental changes restores the benefits for Lamarckian inheritance in conflicting environments, by allowing robotic agents to predict the need for a different behavior, thereby generalizing their control.

## Data
Videos of a selection of bidirectional agents can be found in the following zip file: `results/videos.zip`. 