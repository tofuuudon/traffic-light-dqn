# Reinforcement learning for multi-intersection traffic light controls with lane area detection range

This is my MSc Applied AI and Data Science's final project piece. It's a Simulation of Urban Mobility (SUMO) using a DQN to control the timing intervals of traffic lights.

- Left (NO-AI)
- Right (with AI, MLP-2)

https://user-images.githubusercontent.com/63963445/203042164-7fba0a8c-a45a-427d-8e2d-044cc4168384.mp4



## Abstract

Traffic congestion has always been a major issue in urban environments and is a huge contributor to carbon dioxide emissions and pollution. Solving this issue could reduce such pollutants as well as increase the productivity of the general population, reduce accidents, and bring down costs to resolve traffic-related problems.

With the rise in artificial intelligence (AI) and deep Q-learning (DQN), more methods are becoming increasingly available for researchers to determine a data-driven approach to managing intersections controlled by traffic lights. Using tools like the simulation of urban mobility (SUMO), much research has been conducted into the suitability of using a reinforcement learning (RL) approach to control traffic light timings and traffic flow. Though positive outcomes have been reported, there seems to be a lack of consideration for the use of DQNs in a real-world scenario with the additional constraint of traffic cameras.

This study aimed to implement a system that is adaptive to a real-world road infrastructure gathered from OpenStreetMap (OSM). Additionally, the impact of including lane area detectors with limited view distance to monitor queue length was also considered. The methodology includes a comparison of using a DQN with an increasing amount of hidden layers named MLP-0, MLP-1, and MLP-2 with the addition of a baseline fixed-timing (FT) approach. For each, results were gathered from 50 episodes consisting of 3,600 simulation time steps which is equivalent to 1 hour. Average time loss (ATL), average waiting time (AWT), vehicle count (VC), and episode rewards (ER) were the performance outcomes recorded and were logged using tensorboard.

The results showed that MLP-2 reduced ATL and VC by 16.56% (p<0.05) and 11.26% (p<0.05) — respectively, whilst MLP-0 reduced AWT by 19.18% (p<0.05). MLP-2 also had the largest episode rewards of 4084.75 amongst all the DQN models tested.

To conclude and in comparison to the existing literature, it seems that placing an additional constraint with lane area detectors and using real-world data with minimal modifications — leads to a less performant outcome. It is the hope that this research can help guide future studies in considering real-world applications and scenarios for traffic light management systems.

## Results

![image](https://user-images.githubusercontent.com/63963445/203041180-3e6e823b-6483-4995-a1db-4a312a0ddb8d.png)
![image](https://user-images.githubusercontent.com/63963445/203041232-072760aa-0a20-4fd0-a484-263431ffb3eb.png)
![image](https://user-images.githubusercontent.com/63963445/203041270-5de64e16-3856-4ab2-b2ea-455c2dd00f1c.png)
![image](https://user-images.githubusercontent.com/63963445/203041317-81e0821d-a31b-492b-93d7-e2faa7430ee8.png)


## SCAIDS Conference Presentation

[![Watch the video](https://img.youtube.com/vi/bqJHGqucmCg/maxresdefault.jpg)](https://youtu.be/bqJHGqucmCg)

## Acknowledgements

I would also like to thank Dr Shakeel Ahmad, Dr Drishty Sobnath, and Prins Butt for their teaching and guidance over the course of my degree. I would like to especially thank Dr Nick Whitelegg and Dr Olufemi Isiaq for their continuous support and dedication to my research.

## Publication

In-progress...

