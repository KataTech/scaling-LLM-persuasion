# Analysis Overview

All of the analysis lies primarily in `analysis.ipynb`. In particular, I explored the following questions. 

## 1. Can the treatment messages be associated with academic theories of persuasion? 

The original paper focused on understanding how scaling LLMs affect persuasion. However, their analysis aggregates the persuasiveness at the model level, thereby foregoing a ton of information lying in each of the 730 messages. In this section, I tried to explore message-level explanations for what makes them more "persuasive". I first consulted ChatGPT to provide some popular theories of persuasion, and turned each of those theories into a binary question in the `prompts` folder. For instance, the "Elaboration Likelihood Model" proposed by (Petty & Cacioppo, 1986) postulates that persuasion can be grouped into "central" vs. "peripheral". In this case, "central" refers to a message using "reasons/evidence about the policy (statistics, causal claims)" (see `prompts/central_prompt.txt`). We then write a script `classify.py` to run a binary classification over each of 730 messages along these dimensions and generated one-hot encoding variables (see `llm_responses_labeled.csv`). 

To understand whether these variables meaningfully predict persuasion, we first fit a linear model. It turns out that many of these features are highly correlated, which isn't entirely surprising. We attempt to characterize such correlations better using correlation heatmaps and variation inflation factor. Next, we tried to do feature selection via LASSO. Our preliminary analysis was not able to meaningfully improve the model fit (R^2 ~= 0.023) though. We also did not consider interaction effects, which are likely to play a role. A temporary takeaway is that such coarse groupings may not be enough to explain the variations in persuasiveness. 

## 2. If we use a sentence embedding model on the messages, can we find meaningful clusters? 

Previous section suggests that the coarse grouping by existing persuasive theory may not be enough to explain, so we turn to an extreme form of extracting structured information from text. We embed each of the intervention messages using the SentenceTransformer package. From a 2-D t-SNE projection, we observe meaningful clusters surrounding the topic area (i.e. the issue that the intervention message aims to persuade for). Some outlier points can be found in `treatment_messages_tsne.html`. 

However, these clusters are telling us information we ALREADY know. So we adopted a modified approach of the Habermas Machine paper and tried to "remove" the topical component from each of the word embedding, in hopes that it will leave us with "residual" embeddings that form clusters which correspond to interpretable attributes. For example, it would be interesting to find clusters that correspond to the persuasive strategies used (e.g. central vs. peripheral). We observe no such meaningful patterns in `strategies_scatter_plot/residual_embeddings_central_vs_peripheral.html`. 

## 3. How hard is it to predict how much an individual would agree with a message after viewing a particular message? 

A particular decision that may be informed by this paper is: how can we prevent the use of AI-generated contents for political persuasion? Should we be concerned about it? 

We attempt to answer this question by checking how good existing word embedding models + classical models such as linear regression, LASSO, and SVM predicts user alignment with issue stance. We run K=5 cross validation for each of the model, and find that none achieves significantly better RMSE than the baseline (variance of Y). 

## 4. What would an "automatic hypothesis generating process" find in our messages? 

Since the hand-crafted feature approach in (1) and the full blown word embedding approach in (2) fails to explain how the message affects persuasiveness convincingly, we turn to a recently proposed tool for hypothesis generation, [HypotheSAEs](https://hypothesaes.org/), by (Moova et al. 2025). The approach uses a sparse autoencoder to represent the word embeddings X using a sparse activation matrix Z. Effectively, we now force each treatment message to only activate a few features, and annotate those features automatically with a LLM. Specifically, the LLM labels a feature with the few examples that activate it. The resulting features are fitted in a LASSO regression and the results are reported in `persuasion_hypotheses_lasso.csv`. 

# FAQs
- While we use the phrase `persuasiveness` for our response variable (as in the original paper), we actually measure "alignment with issue stance post intervention". Importantly, we make no attempt to estimate how the intervention message would've changed persuasiveness, since there are no obvious data to do this. The original analysis defines "intervention" at the model level and not message level, so they have "control" for each model, but not every message. Running a sophisticated causal inference may be of future interest. 
- For a thorough view of the original dataset, refer to the `data_cleaning.ipynb` notebook where I worked through various datasets in the `main_study` folder. 
- To run HypotheSAEs, we used OpenAI API. In particular, the user needs to create a `.env` file with "OPENAI_KEY_SAE=..."

# Future Directions 
- An interesting question would be whether we can meaningfully distinguish between AI and human generated persuasion messages. However, this dataset only contains 10 human-written messages so we cannot do that at this time. 

# References 
- Movva, R., Peng, K., Garg, N., Kleinberg, J., & Pierson, E. (2025). Sparse Autoencoders for Hypothesis Generation. In Proceedings of the 42nd International Conference on Machine Learning (ICML).
- Petty, R. E., & Cacioppo, J. T. (1986). The elaboration likelihood model of persuasion. In L. Berkowitz (Ed.), Advances in Experimental Social Psychology (Vol. 19, pp. 123-205). 

