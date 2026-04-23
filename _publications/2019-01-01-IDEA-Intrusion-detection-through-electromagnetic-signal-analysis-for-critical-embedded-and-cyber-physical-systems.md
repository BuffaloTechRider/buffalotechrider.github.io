---
title: "IDEA: Intrusion detection through electromagnetic-signal analysis for critical embedded and cyber-physical systems"
collection: publications
permalink: /publication/2019-01-01-IDEA-Intrusion-detection-through-electromagnetic-signal-analysis-for-critical-embedded-and-cyber-physical-systems
date: 2019-01-01
venue: 'IEEE Transactions on Dependable and Secure Computing'
category: manuscripts
citation: ' Haider Khan,  Nader Sehatbakhsh,  Luong Nguyen,  Robert Callan,  Arie Yeredor,  Milos Prvulovic,  Alenka Zaji{\&apos;c}, &quot;IDEA: Intrusion detection through electromagnetic-signal analysis for critical embedded and cyber-physical systems.&quot; IEEE Transactions on Dependable and Secure Computing, 2019.'
---
Use [Google Scholar](https://scholar.google.com/scholar?q=IDEA:+Intrusion+detection+through+electromagnetic+signal+analysis+for+critical+embedded+and+cyber+physical+systems){:target="_blank"} for full citation

## From EM-Based Intrusion Detection to Neural Network Interpretability

IDEA uses neural networks to analyze electromagnetic emanations from embedded processors, learning to distinguish normal execution patterns from compromised ones. Rather than relying on software-level signatures that malware can evade, we observe the physical side effects of computation — creating a monitoring layer that operates outside the attacker's control. The system learns a representation of "normal behavior" and flags deviations, even for previously unseen attacks.

This is essentially anomaly detection via learned representations of system behavior — the same conceptual framework that underpins much of modern AI safety work. Mechanistic interpretability asks: can we learn to read the internal representations of a neural network well enough to detect when it's doing something unexpected? Our work asked the analogous question for embedded systems: can we learn to read electromagnetic signatures well enough to detect when a processor is executing something it shouldn't be?

The technical skills transfer directly. Training classifiers on high-dimensional signal data, building robust feature extractors that generalize to unseen threat variants, and reasoning about what a learned model has actually captured versus what it might miss — these are the daily challenges of both EM-based security monitoring and neural network interpretability. The difference is the substrate being observed: electromagnetic fields from a processor versus activation patterns in a transformer.