 <b><h1><p  align="center"> NLP-Sentence-Similarity </p></h1></b>

<ol>
<li><a href="https://arxiv.org/pdf/1803.05449v1.pdf">SentEval: An Evaluation Toolkit for Universal Sentence Representations</a></li>
 <code>Abstract:</code>
 
  We introduce SentEval, a toolkit for evaluating the quality of universal sentence representations. SentEval encompasses a variety of tasks, including binary and multi-class classification, natural language inference and sentence similarity. The set of tasks was selected based on what appears to be the community consensus regarding the appropriate evaluations for universal sentence representations. The toolkit comes with scripts to download and preprocess datasets, and an easy interface to evaluate sentence encoders. The aim is to provide a fairer, less cumbersome and more centralized way for evaluating sentence representations.
 
<li><a href="https://arxiv.org/pdf/1802.05667v2.pdf">Calculating the similarity between words and sentences using a lexical database and corpus statistics</a></li>
 
 <code> Abstract: </code>
 
  Calculating the semantic similarity between sentences is a long dealt problem in the area of natural language processing. The semantic analysis field has a crucial role to play in the research related to the text analytics. The semantic similarity differs as the domain of operation differs. In this paper, we present a methodology which deals with this issue by incorporating semantic similarity and corpus statistics. To calculate the semantic similarity between words and sentences, the proposed method follows an edge-based approach using a lexical database. The methodology can be applied in a variety of domains. The methodology has been tested on both benchmark standards and mean human similarity dataset. When tested on these two datasets, it gives highest correlation value for both word and sentence similarity outperforming other similar models. For word similarity, we obtained Pearson correlation coefficient of 0.8753 and for sentence similarity, the correlation obtained is 0.8794.
 
<li><a href="https://arxiv.org/pdf/2004.03844v3.pdf">On the Effect of Dropping Layers of Pre-trained Transformer Models</a></li>
  <code> Abstract: </code>
 Transformer-based NLP models are trained using hundreds of millions or even billions of parameters, limiting their applicability in computationally constrained environments. While the number of parameters generally correlates with performance, it is not clear whether the entire network is required for a downstream task. Motivated by the recent work on pruning and distilling pre-trained models, we explore strategies to drop layers in pre-trained models, and observe the effect of pruning on downstream GLUE tasks. We were able to prune BERT, RoBERTa and XLNet models up to 40%, while maintaining up to 98% of their original performance. Additionally we show that our pruned models are on par with those built using knowledge distillation, both in terms of size and performance. Our experiments yield interesting observations such as, (i) the lower layers are most critical to maintain downstream task performance, (ii) some tasks such as paraphrase detection and sentence similarity are more robust to the dropping of layers, and (iii) models trained using a different objective function exhibit different learning patterns and w.r.t the layer dropping.
 
<li><a href="https://arxiv.org/pdf/1709.08878v2.pdf">Generating Sentences by Editing Prototypes</a></li>
   <code> Abstract: </code>
   We propose a new generative model of sentences that first samples a prototype sentence from the training corpus and then edits it into a new sentence. Compared to traditional models that generate from scratch either left-to-right or by first sampling a latent sentence vector, our prototype-then-edit model improves perplexity on language modeling and generates higher quality outputs according to human evaluation. Furthermore, the model gives rise to a latent edit vector that captures interpretable semantics such as sentence similarity and sentence-level analogies.
   
<li><a href="https://arxiv.org/pdf/1611.02654v2.pdf">Sentence Ordering and Coherence Modeling using Recurrent Neural Networks</a></li>
   <code> Abstract: </code>
 
   Modeling the structure of coherent texts is a key NLP problem. The task of coherently organizing a given set of sentences has been commonly used to build and evaluate models that understand such structure. We propose an end-to-end unsupervised deep learning approach based on the set-to-sequence framework to address this problem. Our model strongly outperforms prior methods in the order discrimination task and a novel task of ordering abstracts from scientific articles. Furthermore, our work shows that useful text representations can be obtained by learning to order sentences. Visualizing the learned sentence representations shows that the model captures high-level logical structure in paragraphs. Our representations perform comparably to state-of-the-art pre-training methods on sentence similarity and paraphrase detection tasks.
   
<li><a href="https://arxiv.org/pdf/1707.07806v2.pdf">Macro Grammars and Holistic Triggering for Efficient Semantic Parsing</a></li>
   <code> Abstract: </code>
 
   To learn a semantic parser from denotations, a learning algorithm must search over a combinatorially large space of logical forms for ones consistent with the annotated denotations. We propose a new online learning algorithm that searches faster as training progresses. The two key ideas are using macro grammars to cache the abstract patterns of useful logical forms found thus far, and holistic triggering to efficiently retrieve the most relevant patterns based on sentence similarity. On the WikiTableQuestions dataset, we first expand the search space of an existing model to improve the state-of-the-art accuracy from 38.7% to 42.7%, and then use macro grammars and holistic triggering to achieve an 11x speedup and an accuracy of 43.7%.
   
<li><a href="https://arxiv.org/pdf/1808.09663v6.pdf">Context Mover's Distance & Barycenters: Optimal Transport of Contexts for Building Representations</a></li>
   <code> Abstract: </code>
 
  We present a framework for building unsupervised representations of entities and their compositions, where each entity is viewed as a probability distribution rather than a vector embedding. In particular, this distribution is supported over the contexts which co-occur with the entity and are embedded in a suitable low-dimensional space. This enables us to consider representation learning from the perspective of Optimal Transport and take advantage of its tools such as Wasserstein distance and barycenters. We elaborate how the method can be applied for obtaining unsupervised representations of text and illustrate the performance (quantitatively as well as qualitatively) on tasks such as measuring sentence similarity, word entailment and similarity, where we empirically observe significant gains (e.g., 4.1% relative improvement over Sent2vec, GenSen). The key benefits of the proposed approach include: (a) capturing uncertainty and polysemy via modeling the entities as distributions, (b) utilizing the underlying geometry of the particular task (with the ground cost), (c) simultaneously providing interpretability with the notion of optimal transport between contexts and (d) easy applicability on top of existing point embedding methods. The code, as well as prebuilt histograms, are available under https://github.com/context-mover/.
   
<li><a href="https://arxiv.org/pdf/1812.08306v1.pdf">NeuralWarp: Time-Series Similarity with Warping Networks</a></li>
   <code> Abstract: </code>
 
   Research on time-series similarity measures has emphasized the need for elastic methods which align the indices of pairs of time series and a plethora of non-parametric have been proposed for the task. On the other hand, deep learning approaches are dominant in closely related domains, such as learning image and text sentence similarity. In this paper, we propose \textit{NeuralWarp}, a novel measure that models the alignment of time-series indices in a deep representation space, by modeling a warping function as an upper level neural network between deeply-encoded time series values. Experimental results demonstrate that \textit{NeuralWarp} outperforms both non-parametric and un-warped deep models on a range of diverse real-life datasets.
   
<li><a href="https://arxiv.org/pdf/1602.07019v2.pdf">Sentence Similarity Learning by Lexical Decomposition and Composition</a></li>
   <code> Abstract: </code>
   
<li><a href="https://arxiv.org/pdf/1603.06807v2.pdf">Generating Factoid Questions With Recurrent Neural Networks: The 30M Factoid Question-Answer Corpus</a></li>
   <code> Abstract: </code>
   
<li><a href="https://arxiv.org/pdf/1605.01194v1.pdf">IISCNLP at SemEval-2016 Task 2: Interpretable STS with ILP based Multiple Chunk Aligner</a></li>
   <code> Abstract: </code>
   
<li><a href="https://arxiv.org/pdf/1610.03098v3.pdf">Neural Paraphrase Generation with Stacked Residual LSTM Networks</a></li>
   <code> Abstract: </code>
   
<li><a href="https://arxiv.org/pdf/1709.01186v1.pdf">Learning Neural Word Salience Scores</a></li>
   <code> Abstract: </code>
   
<li><a href="https://arxiv.org/pdf/1807.00717v1.pdf">Transparent, Efficient, and Robust Word Embedding Access with WOMBAT</a></li>
   <code> Abstract: </code>
   
<li><a href="https://arxiv.org/pdf/1808.10025v1.pdf">Retrieval-Based Neural Code Generation</a></li>
   <code> Abstract: </code>
   
<li><a href="https://aclanthology.org/D18-1328.pdf">Fixing Translation Divergences in Parallel Corpora for Neural MT</a></li>
<li><a href="https://aclanthology.org/N19-1023.pdf">Evaluating Composition Models for Verb Phrase Elliptical Sentence Embeddings</a></li>
<li><a href="https://arxiv.org/pdf/2004.06190v3.pdf">A Divide-and-Conquer Approach to the Summarization of Long Documents</a></li>
<li><a href="https://aclanthology.org/2020.lrec-1.676.pdf">Contextualized Embeddings based Transformer Encoder for Sentence Similarity Modeling in Answer Selection Task</a></li>
<li><a href="https://arxiv.org/pdf/2005.08367v1.pdf">DEXA: Supporting Non-Expert Annotators with Dynamic Examples from Experts</a></li>
<li><a href="https://arxiv.org/pdf/2006.04666v2.pdf">Misinformation Has High Perplexity</a></li>
<li><a href="https://aclanthology.org/2020.acl-main.152.pdf">Parallel Sentence Mining by Constrained Decoding</a></li>
<li><a href="https://arxiv.org/pdf/2010.08269v1.pdf">Effective Distributed Representations for Academic Expert Search</a></li>
<li><a href="https://arxiv.org/pdf/2010.08684v2.pdf">Example-Driven Intent Prediction with Observers</a></li>
<li><a href="https://aclanthology.org/2020.conll-1.24.pdf">Representation Learning for Type-Driven Composition</a></li>
<li><a href="https://arxiv.org/pdf/2011.01421v1.pdf">WSL-DS: Weakly Supervised Learning with Distant Supervision for Query Focused Multi-Document Abstractive Summarization</a></li>
<li><a href="https://arxiv.org/pdf/2012.14700v1.pdf">Image-to-Image Retrieval by Learning Similarity between Scene Graphs</a></li>
<li><a href="https://arxiv.org/pdf/2101.06423v2.pdf">Match-Ignition: Plugging PageRank into Transformer for Long-form Text Matching</a></li>
<li><a href="https://arxiv.org/pdf/2104.08027v2.pdf">Fast, Effective, and Self-Supervised: Transforming Masked Language Models into Universal Lexical and Sentence Encoders</a></li>
<li><a href="https://arxiv.org/ftp/arxiv/papers/2105/2105.00648.pdf">A novel hybrid methodology of measuring sentence similarity</a></li>
<li><a href="https://aclanthology.org/2021.bionlp-1.16.pdf">BioELECTRA:Pretrained Biomedical text Encoder using Discriminators</a></li>
<li><a href="https://arxiv.org/pdf/2106.08648v1.pdf">Semantic sentence similarity: size does not always matter</a></li>
<li><a href="https://arxiv.org/pdf/2106.10955v1.pdf">Extractive approach for text summarisation using graphs</a></li>
<li><a href="https://arxiv.org/pdf/2107.04374v1.pdf">Benchmarking for Biomedical Natural Language Processing Tasks with a Domain Specific ALBERT</a></li>
<li><a href="https://arxiv.org/pdf/2107.05132v2.pdf">LexSubCon: Integrating Knowledge from Lexical Resources into Contextual Embeddings for Lexical Substitution</a></li>
<li><a href="https://arxiv.org/pdf/2109.08449v2.pdf">General Cross-Architecture Distillation of Pretrained Language Models into Matrix Embeddings</a></li>
<li><a href="https://arxiv.org/pdf/2109.10509v1.pdf">Unsupervised Contextualized Document Representation</a></li>
</ol>
