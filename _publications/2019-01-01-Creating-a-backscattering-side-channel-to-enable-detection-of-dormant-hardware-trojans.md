---
title: "Creating a backscattering side channel to enable detection of dormant hardware trojans"
collection: publications
permalink: /publication/2019-01-01-Creating-a-backscattering-side-channel-to-enable-detection-of-dormant-hardware-trojans
date: 2019-01-01
venue: 'IEEE transactions on very large scale integration (VLSI) systems'
category: manuscripts
citation: ' Luong Nguyen,  Chia-Lin Cheng,  Milos Prvulovic,  Alenka Zaji{\&apos;c}, &quot;Creating a backscattering side channel to enable detection of dormant hardware trojans.&quot; IEEE transactions on very large scale integration (VLSI) systems, 2019.'
---
Use [Google Scholar](https://scholar.google.com/scholar?q=Creating+a+backscattering+side+channel+to+enable+detection+of+dormant+hardware+trojans){:target="_blank"} for full citation

## Connecting Hardware Trojan Detection to AI Safety

This paper addresses a fundamental challenge in hardware security: how do you detect a malicious modification that was specifically designed to remain dormant during testing? We developed an electromagnetic backscattering technique that creates a side channel capable of revealing these hidden trojans — even when they never activate during verification. The core insight is that by probing the physical substrate with an external signal, you can observe structural differences that the trojan designer never intended to be visible.

The parallel to AI alignment is striking. Modern neural networks are complex systems where unintended behaviors can lie dormant — emerging only under specific input distributions or deployment conditions that weren't covered during evaluation. Just as a hardware trojan evades traditional functional testing by remaining inactive, a misaligned objective might only manifest in out-of-distribution scenarios. The challenge in both domains is the same: how do you verify the absence of hidden, adversarial behavior in a system too complex to exhaustively test?

Our approach — creating a novel observation channel that reveals structural properties invisible to standard testing — maps directly onto the goals of mechanistic interpretability. Tools like sparse autoencoders and activation probing aim to create "side channels" into neural network internals, revealing learned features and circuits that aren't apparent from input-output behavior alone. Both lines of work share the conviction that understanding internal structure is essential when you can't rely on behavioral testing to guarantee safety.