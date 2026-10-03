---
title: "Concept-Defined Semantic Subspaces for Interpretable Text Representation: A Literature Review"
date: 2026-10-03 10:00:00 +0000
categories: [Machine Learning, NLP]
tags: [embeddings, interpretability, semantic-axes, subtitles, localization, literature-review]
description: "A review of methods that represent text by a few human-defined, interpretable semantic dimensions: seed-defined axes, construct representations, LLM-defined embeddings and discovered axes, how to test whether such dimensions are independent, and how they apply to subtitle quality assessment."
math: true
---

*Compiled October 2026*

## Abstract

Dense text embeddings capture meaning well but suffer from two problems: their individual dimensions have no human meaning, and there are hundreds or thousands of them. This review surveys work on an alternative: a small set of semantic dimensions, each defined by humans through seed words, phrases, sentences or questions, onto which any text can be projected to yield a short, interpretable vector.

We organise the literature into four families. Seed-defined axes (semantic projection, SemAxis, POLAR) build directions from contrasting word sets. Sentence-level construct representations (DDR, CCR) come from computational psychology and define constructs by word lists or questionnaire items. LLM-defined embeddings (QA-Emb, CQG-MBQA, LDIR, QIME) make each dimension an answer to a natural-language question or a relatedness to an anchor text. Data-driven methods (ICA, sparse autoencoders) discover interpretable axes rather than imposing them.

We then review how the independence of dimensions can be measured geometrically, statistically and psychometrically, and survey resources for four dimensions relevant to subtitle text: register, formality, emotional intensity and domain. The central gap is that no existing work combines top-down dimension definition, rigorous independence testing and cross-lingual validation in one framework. Subtitle quality assessment is a natural testbed, because translation is already known to shift formality and attenuate emotional intensity.

## 1. Introduction

### 1.1 Motivation

Text embeddings are the default way to turn language into numbers. A sentence encoder maps any text to a vector of several hundred to several thousand coordinates, and distances between vectors approximate semantic similarity. This works well for retrieval and clustering, but it leaves two problems open.

The first is opacity. No single coordinate means anything a person can name, so an embedding cannot tell us *how* two texts differ, only *that* they do. The second is dimensionality. A 1,024-dimensional vector is hard to inspect, hard to store at catalogue scale, and far larger than the handful of properties an analyst usually cares about.

These problems matter in applied settings such as subtitle localization. A quality reviewer wants to know whether a translation kept the register, formality and emotional force of the source line, not merely whether the two lines are close in an unnamed space.

### 1.2 The core idea

The idea under review borrows directly from linear algebra. Define a few semantic dimensions, each described by a few keywords, phrases or sentences. Embed those descriptions to obtain a basis of k vectors spanning a subspace of the embedding space. Then represent any text by its coordinates in that subspace. The result is a k-dimensional vector, with k typically between 3 and 20, whose every coordinate has a name.

### 1.3 Research questions guiding the review

1. **RQ1 – Definition.** How can semantic dimensions be defined so that they capture the intended property and not a confound such as topic?
2. **RQ2 – Projection.** What is the right mathematical operation for mapping a text embedding onto a set of defined dimensions?
3. **RQ3 – Independence.** How can we test whether the defined dimensions are semantically distinct, and what should be done when they are not?
4. **RQ4 – Validity.** How do we show that projected scores measure what they claim to, including across languages?
5. **RQ5 – Application.** Which dimensions matter for subtitle text, and what resources exist to build and validate them?

### 1.4 Scope and structure

The review covers work from 2016 to 2026 in NLP, computational social science and psychology. Section 2 gives background on embedding geometry. Sections 3 to 6 cover the four families of methods. Section 7 reviews independence measurement. Section 8 surveys resources for subtitle-relevant dimensions. Section 9 synthesises gaps, and Section 10 proposes research directions.

## 2. Background: the geometry of embedding spaces

Projection onto concept directions only makes sense if concepts are, at least approximately, linear directions in embedding space. Two strands of research bear on whether that holds and how to measure it.

### 2.1 Anisotropy

Embedding spaces are not uniformly spread out. [Ethayarajh (EMNLP 2019)](https://arxiv.org/abs/1909.00512) found that contextualized representations from ELMo, BERT and GPT-2 are not isotropic in any layer. In practice, all texts share a large common component, so the raw cosine between almost any two embeddings is high.

This has a direct consequence for concept projection. A score computed as cosine to a single "formal" centroid mostly measures how text-like the input is, plus topical overlap. Methods that work well either centre the space (subtract the corpus mean) or define concepts as differences between two poles, which cancels the shared component. Both remedies recur throughout Sections 3 and 4.

### 2.2 The linear representation hypothesis

The linear representation hypothesis holds that high-level concepts are encoded as directions. [Park, Choe and Veitch (ICML 2024)](https://proceedings.mlr.press/v235/park24c.html) formalise it with counterfactual pairs and show it connects to both linear probing and model steering. Their key point for this review is that projection and cosine similarity require an inner product, and the Euclidean one is not necessarily right.

They identify a non-Euclidean causal inner product under which concepts that can vary freely of each other are orthogonal. The follow-up [Park et al. (ICLR 2025)](https://arxiv.org/abs/2406.01506) extends this to categorical and hierarchical concepts. Their results apply to LLM unembedding spaces rather than sentence encoders, but the lesson transfers: whether two concept axes look independent depends on the inner product used to measure them.

### 2.3 Implications

Three working assumptions follow for the methods reviewed below. Concepts can be approximated as directions or low-dimensional subspaces. Those directions should be estimated from contrasts rather than single poles. And geometric orthogonality is evidence of semantic independence only under a well-chosen inner product, which motivates the statistical independence tests of Section 7.

## 3. Seed-defined semantic axes

The oldest and most direct realisation of the subspace idea defines each dimension as the difference between two sets of seed words. This line began with word embeddings, and its methods carry over to sentence embeddings with modest changes.

### 3.1 Axes from antonym pairs

[SemAxis (An, Kwak and Ahn, ACL 2018)](https://aclanthology.org/P18-1228/) characterises word meaning along many semantic axes beyond sentiment, inducing 732 axes from ConceptNet antonym pairs. Each axis is the difference between the centroids of two pole word sets, and a word's score is its cosine with that axis. SemAxis showed that the same axes reveal different word meanings in different online communities, an early demonstration that axes are portable measurement instruments.

[Kozlowski, Taddy and Evans (*American Sociological Review*, 2019)](https://journals.sagepub.com/doi/full/10.1177/0003122419877135) brought the method to sociology. Dimensions induced by word differences such as rich minus poor corresponded to dimensions of cultural meaning, and projections matched survey responses. Their analysis of a century of books found that the basic cultural dimensions of class stayed stable while the markers of class shifted. The paper also tracked cosine similarity *between* dimensions over time, an early instance of measuring axis independence.

### 3.2 Semantic projection

[Grand, Blank, Pereira and Fedorenko (*Nature Human Behaviour*, 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10349641/) gave the method its most careful validation. They project word vectors onto feature lines such as size (small to big) or danger (safe to dangerous), and the projections recover human judgements across many object categories and features.

Their construction is the reference recipe. Each pole is defined by three near-synonyms, and the axis is the average of the 9 pairwise difference vectors. The averaging matters because individual seed lines for the same feature were only moderately aligned, with mean cosine 0.533, though far more aligned than lines across features at 0.095 ([arXiv version](https://arxiv.org/abs/1802.01241)). Single-pair axes are therefore noisy; multi-seed averaging is necessary.

### 3.3 From scores to a change of basis

SemAxis and semantic projection score each axis independently. [POLAR (Mathew et al., 2020)](https://arxiv.org/abs/2001.09876) instead treats the polar directions as a new basis and transforms embeddings into that subspace. It reports downstream performance comparable to the original embeddings. Writing A for the d × k matrix whose columns are the axis vectors, the two operations differ as follows.

$$
\text{per-axis scores: } s = A^{\top}x \qquad \text{subspace coordinates: } c = (A^{\top}A)^{-1}A^{\top}x
$$

The two coincide only when the axes are orthonormal. When axes correlate, per-axis scores double-count shared variance, while coordinates assign each component of x to exactly one axis. POLAR also identifies the failure mode: the change-of-basis matrix becomes ill-conditioned when the number of axes approaches the embedding dimension, and the coordinates become unreliable. The same instability arises with few but nearly collinear axes.

[SensePOLAR (Engler et al., Findings of EMNLP 2022)](https://arxiv.org/abs/2301.04704) extends POLAR to contextual embeddings by defining poles over word senses, which addresses polysemy. Its results on GLUE and SQuAD are comparable to the original contextual embeddings.

### 3.4 How many independent axes are there?

[Kozlowski, Dai and Boutyline (2025)](https://arxiv.org/abs/2508.10003) apply antonym-defined axes to LLM embedding matrices. Projections on axes such as kind–cruel correlate highly with human ratings, but they reduce to roughly a three-dimensional subspace resembling the Evaluation, Potency and Activity factors of the semantic differential tradition. Moving a token along one axis also shifts geometrically aligned features in proportion to their cosine similarity.

This is the most important cautionary result for the present project. Hand-defined axes are rarely independent, and the number of genuinely distinct dimensions can be far smaller than the number defined.

## 4. Sentence-level construct representations

Computational psychology arrived at the subspace idea independently, with constructs defined by word lists or by full sentences taken from validated questionnaires. This line is the closest existing match to defining a dimension through "a few keywords, phrases or sentences".

### 4.1 DDR and CCR

Distributed Dictionary Representation (DDR, Garten et al., 2018) represents a construct as the centroid of word embeddings from a dictionary, and a document by the centroid of its words; the score is their cosine. Contextualized Construct Representation (CCR) replaces word lists with sentences. As described by [Chen, Li, Li and Atari (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.151), CCR embeds each questionnaire item with SBERT, embeds the target text, and scores the text by its average cosine similarity to the items.

In that study, applied to classical Chinese, fine-tuned CCR outperformed DDR on every task and beat GPT-4 prompting on most ([arXiv](https://arxiv.org/abs/2403.00509)). The study also shows a practical pattern: seed items written in one language can be converted to another and used to measure the same construct across languages.

### 4.2 When to use words and when to use sentences

A methodological guide, ["Neural text embeddings in psychological research: A guide with examples in R"](https://cris.bgu.ac.il/en/publications/neural-text-embeddings-in-psychological-research-a-guide-with-exa-2/), compares DDR, CCR and a supervised alternative called correlational anchored vectors. It recommends word lists for abstract constructs that must generalise across genres, and sentence-based CCR when target texts resemble the seed sentences in form. The supervised method needs large, reliable labelled data.

For subtitles, this implies that seed sentences should be written as dialogue lines, not as definitions, if CCR-style dimensions are to work.

### 4.3 Projection into established norm spaces

A related practice projects embeddings into a space of dimensions validated by human rating studies rather than ad-hoc seeds. The [`semantic-features` tool (2025)](https://arxiv.org/abs/2506.06169) projects contextual embeddings into the Binder et al. (2016) experiential feature space and tracks how person-hood and place-hood features shift across syntactic constructions. The advantage is that each dimension comes with a published definition and human norms; the limitation is that the norms describe words, not sentences.

### 4.4 Assessment

This family contributes two things the NLP literature lacks: an explicit tradition of construct validation, and evidence that sentence-defined dimensions work. It inherits the anisotropy problem from Section 2, however. CCR scores are single-pole cosines, and they are rarely checked for topical confounds or for correlation between constructs.

## 5. LLM-defined interpretable embeddings

Since 2024 a third family has replaced geometric projection with direct questioning: each dimension is a natural-language question, and an LLM supplies the coordinate. This sidesteps embedding geometry entirely, at the cost of inference.

### 5.1 Questions as dimensions

[QA-Emb (Benara et al., NeurIPS 2024)](https://arxiv.org/abs/2405.16714) builds embeddings in which each feature is the answer to a yes/no question posed to an LLM. Learning the embedding reduces to choosing questions rather than training weights. Developed for predicting fMRI responses to language, it outperformed an established interpretable baseline with few questions. Code is at [csinva/interpretable-embeddings](https://github.com/csinva/interpretable-embeddings).

[CQG-MBQA (Sun et al., 2025)](https://arxiv.org/abs/2410.03435) automates question design. Contrastive Question Generation uses dense embeddings and a generative LLM to produce discriminative questions, and a multi-task binary QA model then answers them, cutting LLM calls. The resulting vectors are typically very high-dimensional, around 10,000 binary features.

### 5.2 Anchors as dimensions

[LDIR (Wang, Shen and Huang, Findings of ACL 2025)](https://arxiv.org/abs/2505.10354) answers the dimensionality problem. Each of fewer than 500 dense dimensions is the relatedness of the input to an anchor text chosen by farthest-point sampling. LDIR performs close to black-box embeddings on similarity, retrieval and clustering, and outperforms the question-based baselines with far fewer dimensions. Code is at [szu-tera/LDIR](https://github.com/szu-tera/LDIR). Its anchors are discovered rather than designed, so it occupies a middle ground between this section and Section 6.

### 5.3 Domain-grounded questions

[QIME (Tang et al., March 2026)](https://sotaverified.org/papers/260301690) grounds questions in a domain ontology for medical text, so that each dimension is a clinically meaningful yes/no question. It conditions on cluster-specific concept signatures to generate semantically atomic questions and offers a training-free construction that removes per-question classifiers. It outperforms prior interpretable embeddings on biomedical benchmarks and narrows the gap to black-box encoders. The pattern of grounding dimensions in an expert taxonomy transfers directly to localization, where MQM and style guides provide that taxonomy.

### 5.4 Assessment

Question-based embeddings make dimensions maximally readable and, unlike geometric axes, do not depend on anisotropy or inner-product choices. Their weaknesses are cost, binary granularity and reliance on the LLM answering faithfully. None of these papers treats the independence of questions as a primary design criterion; CQG pursues discriminativeness, which is related but not the same.

## 6. Data-driven discovery of interpretable axes

The fourth family inverts the direction of work: instead of imposing dimensions, it finds the axes already present in an embedding space and labels them afterwards. These methods matter here less as alternatives than as instruments for checking whether hand-defined dimensions match the space's own structure.

### 6.1 Independent component analysis

[Yamagiwa et al. (EMNLP 2023), "Discovering Universal Geometry in Embeddings with ICA"](https://arxiv.org/abs/2305.13175) focuses on the intrinsic independence within embeddings rather than on imposed axes. ICA extracts independent semantic components, and each embedding turns out to be a composition of a few interpretable axes that stay consistent across languages, algorithms and modalities.

Follow-up work, including "Understanding Higher-Order Correlations Among Semantic Components" (EMNLP 2024), examines dependence that ICA leaves behind, and a reproducibility study tests ICA axes within and across languages using clustering and statistical tests ([author page](https://catalyzex.com/author/Rongzhi%20Li)). For this project, ICA offers a direct check: if a defined formality axis aligns closely with an ICA component, it reflects real structure in the space.

### 6.2 Sparse autoencoders

[O'Neill et al. (2024), "Disentangling Dense Embeddings with Sparse Autoencoders"](https://arxiv.org/abs/2408.00657) trained SAEs on embeddings of over 420,000 scientific abstracts. The sparse features kept semantic fidelity, formed "feature families" at different levels of abstraction, and could steer semantic search. ["Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit" (December 2025)](https://arxiv.org/abs/2512.10092) builds SAE embeddings from LLM hidden states, where each dimension maps to a concept. These support clustering along chosen axes and beat dense embeddings on property-based retrieval.

SAEs produce thousands of features, so they solve the interpretability problem but not the dimensionality one. Their value here is as a dictionary in which to look for features matching each defined dimension.

### 6.3 Formal accounts of semantic independence

["Uncovering Meanings of Embeddings via Partial Orthogonality" (Jiang et al., NeurIPS 2023)](https://par.nsf.gov/biblio/10542247) gives the most formal treatment of what independence between meanings should look like in an embedding. It argues that partial orthogonality captures semantic independence and introduces independence-preserving embeddings. A complementary geometric view from EMNLP 2025 decomposes sentence embeddings into [semantic regions on the hypersphere](https://underline.io/lecture/132521-semantic-geometry-of-sentence-embeddings) that show hierarchical, inclusion-like organisation.

### 6.4 Assessment

Discovered axes are faithful to the space but have no guaranteed meaning; defined axes have meaning but no guaranteed faithfulness. The two are complementary, and the strongest designs will use discovery methods to audit defined dimensions.

## 7. Measuring independence among semantic dimensions

Independence can be assessed at three levels: the geometry of the axis vectors, the statistics of scores across a corpus, and the validity of the constructs themselves. Each answers a different question, and the literature rarely combines them.

### 7.1 Geometric independence

The Gram matrix of unit-normalised axes, G = AᵀA, gives the cosine between every pair of dimensions before any text is scored. Its condition number flags near-collinear pairs, which POLAR identifies as the source of unreliable coordinates (Section 3.3). Principal angles between multi-vector concept subspaces generalise this to dimensions defined by more than one direction.

Two caveats apply. Geometric orthogonality depends on the inner product, and Park et al. argue that the Euclidean one need not respect semantic independence (Section 2.2). And seed noise inflates apparent independence: Grand et al.'s within-feature alignment of 0.533 means a poorly seeded axis can look orthogonal to everything simply because it is noisy.

### 7.2 Statistical independence

Once texts are scored, the correlation structure of the score matrix shows whether dimensions co-vary in real data. The basic tools are Spearman correlation, partial correlation controlling for confounds such as length, and non-linear measures such as distance correlation and HSIC. For binary dimensions, the SAE toolkit of Section 6.2 reports normalised PMI and conditional co-occurrence between concepts.

Factor analysis or ICA on the score matrix estimates how many independent factors underlie the defined dimensions. Kozlowski et al. (2025) found that 28 antonym axes collapse to about three, and the study of factor structure in [human semantic ratings](https://arxiv.org/abs/2508.10003) suggests this collapse reflects language itself rather than an embedding artefact.

### 7.3 Psychometric validity

Psychometrics offers the most rigorous framework for deciding whether dimensions are distinct constructs: the multitrait-multimethod matrix (Campbell and Fiske, 1959). Each dimension is measured by at least two independent methods, for example embedding projection and an LLM rater. Discriminant validity holds when the same dimension agrees across methods more strongly than different dimensions agree within one method.

This guards against a failure the geometric and statistical views miss: method-induced correlation. If one LLM prompt rates all dimensions together, halo effects create correlations that are artefacts of measurement. Recent psychometrics work points the same way, using sentence embeddings to [identify semantically redundant scales](https://doi.org/10.1177/00131644261430762) and noting that cosine similarity is sensitive to surface lexical overlap.

### 7.4 Assessment

No reviewed work applies all three levels to defined text dimensions. Geometric checks are cheap but inner-product dependent; statistical checks reflect real co-variation but conflate method with meaning; psychometric checks separate the two but need multiple measurement methods. A framework combining them is an open contribution.

## 8. Candidate dimensions for subtitle text

Four dimensions are proposed for subtitle text: register, formality, emotional intensity and domain. Resources exist for all four, but they differ in maturity, language coverage and fit to dialogue.

| Dimension | Main resources | Languages | Fit to subtitles | Key caveat |
| --- | --- | --- | --- | --- |
| Register | [TurkuNLP multilingual CORE classifier](https://huggingface.co/TurkuNLP/web-register-classification-multilingual); Biber MDA via [pybiber](https://pypi.org/project/pybiber/) and [MFTE](https://pypi.org/project/MFTE) | Classifier: trained on 5, applied to ~100; MDA: English | Weak for web genres; good for Biber dimensions | Tagger choice changes dimension scores |
| Formality | [GYAFC and XFORMAL](https://arxiv.org/abs/2104.04108); [s-nlp XLM-R classifier](https://huggingface.co/s-nlp/xlmr_formality_classifier) | EN, PT-BR, FR, IT | Good; rewrite pairs suit minimal-pair axes | Overlaps heavily with register |
| Emotional intensity | [BRIGHTER / SemEval-2025 Task 11](https://arxiv.org/abs/2503.07269) | 30+ labelled, 11 with intensity | Good at cue level | Intensity is per emotion, 0–3 scale |
| Domain | No standard resource; ontology-grounded questions as in QIME | Any | Good at scene or episode level | Must be built from an in-house taxonomy |

### 8.1 Register

In NLP, register usually means document-level web genre. The TurkuNLP model is fine-tuned from XLM-RoBERTa-large on multilingual CORE corpora and predicts CORE labels for all languages XLM-R covers ([Henriksson et al.](https://arxiv.org/abs/2406.19892)). But almost all subtitle text belongs to one macro-register, scripted spoken dialogue, so this taxonomy discriminates little within it.

The variation that matters within subtitles is sociolinguistic, from intimate to ceremonial, and is best captured by Biber's multidimensional analysis (MDA). pybiber extracts Biber's 67 lexicogrammatical features and implements the MDA pipeline. A recent comparison of five open Biber-style taggers found that all reproduce known register differences, yet [dimension scores still vary substantially](https://www.researchsquare.com/article/rs-9163857) because a few features are implemented inconsistently.

### 8.2 Formality

Formality has the richest resources. GYAFC (English) and XFORMAL (Brazilian Portuguese, French, Italian) are the two main annotated collections, and [Dementieva et al. (RANLP 2023)](https://aclanthology.org/2023.ranlp-1.31) provide trained classifiers for public use. Because both datasets consist of informal–formal rewrite pairs, they are ideal for building topic-controlled formality axes from paired differences.

### 8.3 Emotional intensity

The [SemEval-2025 Task 11](https://arxiv.org/abs/2503.07269) data cover more than 30 languages, with emotion intensity annotated for 11 of them on a 0–3 scale. For continuous scores from LLM raters, [Bagdon et al. (NAACL 2024)](https://aclanthology.org/2024.naacl-long.439) found best–worst scaling more reliable than rating scales or paired comparisons, and a regressor trained on those labels performed nearly on par with one trained on manual labels.

### 8.4 Evidence that translation shifts these dimensions

Two findings make these dimensions directly relevant to translation quality. In the XFORMAL study, round-trip machine translation preserved formal sentences well but [shifted informal sentences toward formal values](https://arxiv.org/abs/2104.04108). A WASSA 2026 crowd study found that machine translation [systematically attenuates emotion intensity](https://preview.aclanthology.org/ingest-eacl/2026.wassa-1.11.pdf), and that LLM prompting reduces but does not remove the loss.

A counterweight applies. [Cui et al. (ICLR 2026)](https://mlanthology.org/iclr/2026/cui2026iclr-utterance/) show that subtitle corpora are translated far more liberally than other domains, and so are "not actually parallel" in a strict sense. Some shift along these dimensions is therefore legitimate, and any QA use must first calibrate a normal range against approved human translations.

## 9. Synthesis: gaps and open problems

The four families trade interpretability, compactness, faithfulness and cost against each other, and none satisfies all the requirements of the research questions in Section 1.3.

| Family | Representative work | How a dimension is defined | Typical dimensions | Unit | Independence addressed? | Main limitation |
| --- | --- | --- | --- | --- | --- | --- |
| Seed-defined axes | SemAxis, semantic projection, POLAR | Difference of pole word sets | 1–1,000 | Word (mostly) | Partly: Gram matrix, factor collapse | Seed noise; built for words |
| Construct representations | DDR, CCR | Word list or questionnaire sentences | 1–20 | Sentence, document | Rarely | Single-pole cosine; topical confounds |
| LLM-defined | QA-Emb, CQG-MBQA, LDIR, QIME | Yes/no question or anchor text | 100–10,000 | Sentence, document | Not as a design goal | Inference cost; binary granularity |
| Discovered axes | ICA, sparse autoencoders | Learned from the space, labelled after | Hundreds to thousands | Word, sentence | Yes, by construction (ICA) | Meaning not guaranteed |

### 9.1 Gaps

1. **No integrated framework.** No reviewed work combines top-down definition of a few sentence-level dimensions, independence testing at the geometric, statistical and psychometric levels, and external validation in one pipeline.
2. **Word-level origins.** The most rigorously validated axis methods (semantic projection, POLAR) were developed and tested on words. Their behaviour on sentence and paragraph embeddings, especially for stylistic rather than topical properties, is largely untested.
3. **Topical confounding is under-controlled.** Seed sets typically mix style with topic. Minimal-pair construction, natural for formality via GYAFC and XFORMAL, has not been adopted as standard practice for defining sentence-level axes.
4. **Projection operator choice is under-examined.** Per-axis scores and least-squares coordinates are used interchangeably, although they diverge whenever axes correlate.
5. **Cross-lingual validity is assumed, not tested.** Multilingual encoders carry language-identity directions, and it is unknown how well an axis built in one language ranks texts in another.
6. **Short texts.** Subtitle cues of 5–10 words sit below the length most methods were validated on, and different dimensions likely need different units of analysis.
7. **Few applications to translation quality.** Evidence of formality shift and emotion attenuation exists, but no work uses defined-dimension vectors of source and target as a QA signal.

## 10. Proposed research directions

The gaps point to one project: a framework for concept-defined semantic subspaces over sentence embeddings, validated for independence and cross-lingual consistency, and tested as a signal for subtitle quality assessment.

### 10.1 Research questions

1. **RQ-A (construction).** Do axes built from minimal pairs separate style from topic better than axes built from unpaired seed sets?
2. **RQ-B (projection).** For correlated axes, do least-squares subspace coordinates agree better with external labels than per-axis cosine scores?
3. **RQ-C (independence).** How many distinct dimensions underlie register, formality, emotional intensity and domain in subtitle text, by geometric, statistical and multitrait-multimethod criteria?
4. **RQ-D (cross-lingual).** Do axes built in one language rank texts correctly in another when projected through a multilingual encoder?
5. **RQ-E (application).** Do source–target deltas in the defined subspace distinguish approved from defective subtitle translations?

### 10.2 Methodology sketch

**Axis construction.** Each dimension is defined by paired seed texts that hold content constant and vary only the target property. Formality pairs come from GYAFC and XFORMAL; register and emotional-intensity pairs are produced by LLM rewriting of real subtitle lines; domain is defined by ontology-grounded questions or exemplar scenes. The axis is the mean of the paired difference vectors, in a centred embedding space.

**Projection.** Texts are mapped with both operators from Section 3.3 so that RQ-B can be tested directly. The residual fraction of each embedding outside the subspace is recorded to show how much meaning the defined dimensions capture.

**Independence analysis.** Geometric: Gram matrix, condition number and split-half seed stability. Statistical: Spearman and partial correlations controlling for cue length, distance correlation, and factor analysis or ICA on the score matrix. Psychometric: a multitrait-multimethod matrix pairing embedding projection with LLM best–worst-scaling ratings and the published classifiers of Section 8.

**Units of analysis.** Emotional intensity and formality are scored per cue with a context window of neighbouring cues; register and domain per scene. This tests Gap 6 explicitly.

### 10.3 Evaluation plan

| Stage | Data | Metric | Success criterion |
| --- | --- | --- | --- |
| Axis validity | GYAFC/XFORMAL held-out pairs; BRIGHTER intensity sets | Spearman ρ with gold labels | Projection within reach of the dedicated classifier |
| Seed robustness | Split-half seed sets per axis | Cosine between half-axes | Clearly above cross-axis cosines |
| Independence | ~2,000 subtitle cues, several genres | MTMM convergent vs. discriminant correlations | Convergent exceeds discriminant for retained dimensions |
| Cross-lingual | Parallel subtitle cues, at least two language pairs | Rank agreement of source and target coordinates | Agreement on approved translations |
| QA signal | Approved vs. known-defective translations | AUC of per-dimension and Mahalanobis deltas | Separation above a length-only baseline |

### 10.4 Expected contributions

The project would contribute a reusable recipe for building interpretable low-dimensional text vectors from human-written definitions; a three-level protocol for testing whether such dimensions are genuinely distinct; the first evidence on whether these vectors transfer across languages through multilingual encoders; and a calibrated, interpretable signal for register and tone preservation in subtitle translation.

## References

**Embedding geometry**

- Ethayarajh, K. (2019). How Contextual are Contextualized Word Representations? EMNLP-IJCNLP. [arXiv:1909.00512](https://arxiv.org/abs/1909.00512)
- Park, K., Choe, Y. J., & Veitch, V. (2024). The Linear Representation Hypothesis and the Geometry of Large Language Models. ICML. [PMLR](https://proceedings.mlr.press/v235/park24c.html)
- Park, K., Choe, Y. J., Jiang, Y., & Veitch, V. (2025). The Geometry of Categorical and Hierarchical Concepts in Large Language Models. ICLR. [arXiv:2406.01506](https://arxiv.org/abs/2406.01506)

**Seed-defined axes**

- An, J., Kwak, H., & Ahn, Y.-Y. (2018). SemAxis: A Lightweight Framework to Characterize Domain-Specific Word Semantics Beyond Sentiment. ACL. [ACL Anthology P18-1228](https://aclanthology.org/P18-1228/)
- Kozlowski, A. C., Taddy, M., & Evans, J. A. (2019). The Geometry of Culture: Analyzing the Meanings of Class through Word Embeddings. *American Sociological Review* 84(5). [SAGE](https://journals.sagepub.com/doi/full/10.1177/0003122419877135)
- Grand, G., Blank, I. A., Pereira, F., & Fedorenko, E. (2022). Semantic projection recovers rich human knowledge of multiple object features from word embeddings. *Nature Human Behaviour* 6(7). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10349641/) · [arXiv:1802.01241](https://arxiv.org/abs/1802.01241)
- Mathew, B., Sikdar, S., Lemmerich, F., & Strohmaier, M. (2020). The POLAR Framework: Polar Opposites Enable Interpretability of Pre-Trained Word Embeddings. [arXiv:2001.09876](https://arxiv.org/abs/2001.09876)
- Engler, J., et al. (2022). SensePOLAR: Word Sense Aware Interpretability for Pre-trained Contextual Word Embeddings. Findings of EMNLP. [arXiv:2301.04704](https://arxiv.org/abs/2301.04704)
- Kozlowski, A. C., Dai, C., & Boutyline, A. (2025). Semantic Structure in Large Language Model Embeddings. [arXiv:2508.10003](https://arxiv.org/abs/2508.10003)

**Construct representations**

- Chen, Y., Li, S., Li, Y., & Atari, M. (2024). Surveying the Dead Minds: Historical-Psychological Text Analysis with Contextualized Construct Representation (CCR) for Classical Chinese. EMNLP. [ACL Anthology](https://aclanthology.org/2024.emnlp-main.151) · [arXiv:2403.00509](https://arxiv.org/abs/2403.00509)
- Neural text embeddings in psychological research: A guide with examples in R. [Publication page](https://cris.bgu.ac.il/en/publications/neural-text-embeddings-in-psychological-research-a-guide-with-exa-2/)
- semantic-features: A User-Friendly Tool for Studying Contextual Word Embeddings in Interpretable Semantic Spaces (2025). [arXiv:2506.06169](https://arxiv.org/abs/2506.06169)
- Sentence embeddings and psychometric factor loadings (2026). *Educational and Psychological Measurement*. [DOI](https://doi.org/10.1177/00131644261430762)

**LLM-defined embeddings**

- Benara, V., Singh, C., Morris, J. X., Antonello, R., Stoica, I., & Huth, A. (2024). Crafting Interpretable Embeddings by Asking LLMs Questions. NeurIPS. [arXiv:2405.16714](https://arxiv.org/abs/2405.16714) · [code](https://github.com/csinva/interpretable-embeddings)
- Sun, Y., et al. (2025). A General Framework for Producing Interpretable Semantic Text Embeddings (CQG-MBQA). [arXiv:2410.03435](https://arxiv.org/abs/2410.03435)
- Wang, Y., Shen, Z., & Huang, H. (2025). LDIR: Low-Dimensional Dense and Interpretable Text Embeddings with Relative Representations. Findings of ACL. [arXiv:2505.10354](https://arxiv.org/abs/2505.10354) · [code](https://github.com/szu-tera/LDIR)
- Tang, Y., Lin, Z., Sun, Y., Hsu, W., Lee, M. L., & Tung, A. K. H. (2026). QIME: Constructing Interpretable Medical Text Embeddings via Ontology-Grounded Questions. [Paper page](https://sotaverified.org/papers/260301690)
- Interpretable Text Embeddings and Text Similarity Explanation: A Survey (2025). [arXiv:2502.14862](https://arxiv.org/abs/2502.14862)

**Discovered axes and independence**

- Yamagiwa, H., Oyama, M., & Shimodaira, H. (2023). Discovering Universal Geometry in Embeddings with ICA. EMNLP. [arXiv:2305.13175](https://arxiv.org/abs/2305.13175)
- O'Neill, C., Ye, C., Iyer, K., & Wu, J. F. (2024). Disentangling Dense Embeddings with Sparse Autoencoders. [arXiv:2408.00657](https://arxiv.org/abs/2408.00657)
- Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit (2025). [arXiv:2512.10092](https://arxiv.org/abs/2512.10092)
- Uncovering Meanings of Embeddings via Partial Orthogonality (2023). NeurIPS. [NSF PAR](https://par.nsf.gov/biblio/10542247)
- Semantic Geometry of Sentence Embeddings (2025). EMNLP. [Talk page](https://underline.io/lecture/132521-semantic-geometry-of-sentence-embeddings)

**Subtitle-relevant dimensions and translation**

- Henriksson, E., et al. Automatic Register Identification for the Open Web Using Multilingual Deep Learning. [arXiv:2406.19892](https://arxiv.org/abs/2406.19892) · [model](https://huggingface.co/TurkuNLP/web-register-classification-multilingual)
- pybiber. [PyPI](https://pypi.org/project/pybiber/) · MFTE Python. [PyPI](https://pypi.org/project/MFTE)
- Comparison of Biber-style tagging systems for multidimensional analysis. [Research Square](https://www.researchsquare.com/article/rs-9163857)
- Briakou, E., et al. (2021). Olá, Bonjour, Salve! XFORMAL: A Benchmark for Multilingual Formality Style Transfer. NAACL. [arXiv:2104.04108](https://arxiv.org/abs/2104.04108) · [data](https://github.com/Elbria/xformal-FoST)
- Dementieva, D., et al. (2023). Detecting Text Formality: A Study of Text Classification Approaches. RANLP. [ACL Anthology](https://aclanthology.org/2023.ranlp-1.31) · [model](https://huggingface.co/s-nlp/xlmr_formality_classifier)
- Muhammad, S. H., et al. (2025). SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection. [arXiv:2503.07269](https://arxiv.org/abs/2503.07269) · [dataset](https://brighter-dataset.github.io)
- Bagdon, C., Karmalkar, P., Gurulingappa, H., & Klinger, R. (2024). "You are an expert annotator": Automatic Best–Worst-Scaling Annotations for Emotion Intensity Modeling. NAACL. [ACL Anthology](https://aclanthology.org/2024.naacl-long.439)
- Crowd-Based Evaluation of Emotion Intensity Preservation (2026). WASSA. [PDF](https://preview.aclanthology.org/ingest-eacl/2026.wassa-1.11.pdf)
- Cui, C., et al. (2026). From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization. ICLR. [MLAnthology](https://mlanthology.org/iclr/2026/cui2026iclr-utterance/)

**Cited without retrieved link:** Binder et al. (2016), experiential semantic feature norms; Campbell & Fiske (1959), the multitrait-multimethod matrix; Garten et al. (2018), Distributed Dictionary Representation.
