---
permalink: /
title: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Hi, I'm Paul Nguyen (Luong Nguyen). I grew up in a small rural village in northern Vietnam, where I used to herd and ride water buffalo—an essential companion in our daily farming life. That experience shaped my resilience and grounded my journey into technology.

Currently, I build applied AI systems where reliability matters. At Amazon I lead the integration of generative AI into the Alexa Auto experience - fine-tuning and evaluating LLMs, building the infrastructure that gets them shipped, and coordinating across domain teams (music, car control, navigation, search) to turn generic model capability into automotive-specific experiences.

I also pursue personal projects where I develop models and explore machine learning solutions for various challenges I encounter. Currently building Autodidact, an open-source local-first AI agent that learns from its cloud queries - the local brain answers what it knows, escalates to the cloud when it doesn't, and distills what it learns back down.

In my free time, I enjoy traveling, hiking, camping, backpacking, skiing, and discovering great hot pot spots.

I'm skilled in C/C++, Java, Python, Scala, Tcl, and TypeScript, with strong experience in applied machine learning and computer vision, software design and development, computer architecture, and distributed systems. I hold a Master's degree in Computer Engineering from Seoul National University, where I focused on video compression and computer vision, and a Ph.D. in Computer Engineering, with a focus on software/hardware security and applying machine learning to security problems.

## Why AI Safety

My career has been a steady progression toward AI safety, though I didn't have the words for it at the time. My PhD was about detecting hidden malicious behaviors in complex systems - using electromagnetic side-channel analysis and machine learning to find hardware trojans and malware designed to evade detection. The core skill was probing opaque systems for structure and intent that weren't meant to be seen, which is strikingly close to what mechanistic interpretability researchers are doing with neural networks today.

Over the last few years at Amazon, I moved from analyzing hidden behaviors to building AI products where failures have real-world consequences. Shipping generative AI into a vehicle means taking graceful degradation, fail-safe defaults, and "when *not* to ship" seriously as design constraints — the same principles that shape responsible scaling for frontier AI. I spend more of my day on the applied-engineering side of this (prompt scaffolding, integration, evaluation, fine-tuning, monitoring) than on inference kernels or model architecture research, and I'm explicit about that. What I bring is the discipline of making AI systems actually work in the hands of real users under reliability constraints that don't forgive sloppy engineering.

The thread connecting these experiences: side-channel analysis maps to interpretability (extracting meaning from systems that don't readily reveal it), automotive fail-safes map to responsible scaling (knowing when to hold back), and adversarial detection maps to alignment (asking whether a system is doing what we intend, or something else). Ensuring AI systems behave as intended — and catching when they don't — is the natural extension of the work I've been doing, now applied to the most consequential systems being built.

## Latest Publications

{% for pub in site.publications reversed limit:3 %}
  * [{{ pub.title }}]({{ pub.url }}) - {{ pub.date | date: "%Y" }}
{% endfor %}
## Recent Updates

{% assign all_items = site.publications | concat: site.posts | concat: site.talks | sort: 'date' | reverse %}
{% for item in all_items limit:5 %}
  * [{{ item.title }}]({{ item.url }}) - {{ item.date | date: "%B %Y" }}
{% endfor %}
