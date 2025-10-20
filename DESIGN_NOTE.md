# Design Note

This setup abstracts **real QA automation** into a *family of directed graphs* where nodes are screens and edges are user actions (click, type, navigate).  
Noise and sparsity stem from **dead-ends**, **stochastic pop-ups**, and **hidden dependencies** that mimic latent real-world states (auth, cookies, A/B flags).  
Observations are deliberately compact (one-hot of current node, padded to `obs_dim = 32`), creating a **Partially Observable Markov Decision Process (POMDP)** and motivating **memory-based policies**.

We compare **value-based (DQN)** and **policy-gradient (PPO)** baselines, then introduce a **recurrent policy (PPO-LSTM)** to handle partial observability.  
The graph generator enables training over a *distribution* of flows and evaluation under **distribution shift** (deeper graphs, more pop-ups).  
Metrics include success rate and steps-to-success, aggregated across ≥ 5 seeds with 95 % CIs.  
All configs, seeds, and plotting scripts are deterministic for reproducibility.

**Results summary.**  
All agents succeed on ≈ 80–85 % of base graphs. PPO leads slightly overall, DQN remains competitive on shorter horizons.  
Under heavier pop-ups, **PPO-LSTM** degrades least, confirming **temporal memory** mitigates hidden-state uncertainty.  
Performance on deeper graphs declines gracefully, showing **generalisation** beyond training distribution.

**Extension to production.**  
This abstraction scales to QA or web-automation pipelines: scrape site navigations into graphs, randomise DOM-specific elements for robustness, and warm-start RL from logs.  
Memory and exploration remain essential for reaching **rare success paths** amid high noise and latent state.