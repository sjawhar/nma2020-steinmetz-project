# Neuromatch Academy 2020 - Ethereal Pony Group Project

[Video Presentation](https://youtu.be/F0XEv4pIRe8)

## Overview

**Project Topic:** Time Travel from V1 to M1

**Scientific Question:** How does the timing of information flow between V1 and M1 change as a function of task experience, outcome, and feedback?

**Brief scientific background:** Information processing theory typically assumes that information is processed sequentially, for example from sensing the visual stimulus to selecting and executing a response. This mental process manifests behaviorally in the reaction time. However, recent behavioral studies have challenged this view. It is important to provide some neural evidence to answer this question: is information processed sequentially or parallel between V1 and M1? Recent studies of representations in visual cortex have used population codes and decoder performance timing as proxies for resolved neural representations, with good results.

**Proposed Analyses:** Train decoders on V1 and M1 and find the time difference between decoder deicions in each region. In addition, using PCA, project V1 and M1 activity onto a lower-dimensional manifold and explore the neural trajectory during stimulus presentation and response.

**Predictions:** The time difference between the "decision points" in V1 and M1 is predictive of task performance, where a "decision point" is defined below. In addition, the time difference may depend on trial types.

## Results
* Neural trajectory, decoder analyses, and spiking data give **consistent timing information** within a session.
* Sessions with data from **primary or secondary visual and motor regions** show a clear time delay in visual and motor information availability.
* Sessions with a time delay have interesting **neural trajectories**.

## Definitions
**Decision point:** time point after stimulus presentation at which decoder confidence reaches a threshold  
**Deliberation time:** time difference between the "decision points" in V1 and M1

## Questions
* How does information flow temporally from V1 to M1?
  * Q1: How does the deliberation time differ between correct and incorrect trials?
  * Q2: How does the deliberation time differ between easy and difficult trials?
  * Q3: How does the deliberation time change over the course of several trials (learning)?
  * Q4: What characterizes the neural trajectory in V1/M1 at the decision point?
  * Q5: What is the time difference between the M1 decision point and when the mouse actually executes its choice? Positive? Negative?
* Can we see signs of recurrent information flow back to V1?
  * Q6: Does the representation of the visual stimulus in V1 change in trials after incorrect trials (attention)?
  * Q7: Incorrect ambiguous choice representation
    * Trial type A: the mouse is presented with a "no response" stimulus but (incorrectly) chooses left or right.
    * Trial type B: the mouse is presented with a "no response" stimulus and (correctly) chooses no response.
    * Trial type C: the mouse is presented with a "left" or "right" stimulus.
    * Question: is the representation in visual cortex in trial type A more similar to trial type B (same stimulus, different choice) or to trial type C (different stimulus, same choice)?

## Predictions
* Q1: Controlling for difficulty, incorrect trials will have a shorter deliberation time
* Q1: Deliberation time will lengthen after incorrect trials
* Q2: For correct trials, easy trials will have a shorter deliberation time
* Q3: Deliberation time will shorten over consecutive trials
  * Will it saturate?
* Q4: Decision time corresponds to point of minimum neural trajectory velocity
* Q5: There are times when action is executed before the M1 decision point
* Q6: Decoder performance increases for trials after incorrect trials
* Q7: Visual representation will show characteristics of the incorrect stimulus (top-down, recurrent effect)

## Methods
* GLM decoder: logistic classifier
* PCA -> manifold -> trajectory

## Next Steps
* More fine-grained quantifications of timing information from neural trajectory (Michaels et al., 2015; Wang et al. 2018) and non-linear decoder models might allow for detecting deliberation time even without data from V1 or M1.
* Could we use such methods to chart the regions involved in the flow of information from V1 to M1?
* What dynamics might be observed in those regions?

## References
* [Vyas, S., Golub, M. D., Sussillo, D., & Shenoy, K. V. (2020). Computation Through Neural Population Dynamics. _Annual Review of Neuroscience_, 43, 249-275.](https://sci-hub.tw/https://www.annualreviews.org/doi/abs/10.1146/annurev-neuro-092619-094115)
* [Kar, K., & DiCarlo, J. J. (2020). Fast recurrent processing via ventral prefrontal cortex is needed by the primate ventral stream for robust core visual object recognition. _NEURON-D-20-00886_.](https://www.biorxiv.org/content/10.1101/2020.05.10.086959v1)
* [Majaj, N. J., Hong, H., Solomon, E. A., & DiCarlo, J. J. (2015). Simple learned weighted sums of inferior temporal neuronal firing rates accurately predict human core object recognition performance. _Journal of Neuroscience, 35_(39), 13402-13418.](https://www.jneurosci.org/content/35/39/13402.short)
* [Gwilliams, L., King, J. R., Marantz, A., & Poeppel, D. (2020). Neural dynamics of phoneme sequencing in real speech jointly encode order and invariant content. _bioRxiv_.](https://www.biorxiv.org/content/10.1101/2020.04.04.025684v1.abstract)


## Development
To get start, just install Docker and Docker Compose. Then run

```bash
docker-compose up notebook
```

And Jupyter Lab will be running at http://localhost:8888.
