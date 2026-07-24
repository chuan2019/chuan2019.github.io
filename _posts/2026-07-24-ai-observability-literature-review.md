---
title: "Observability of Modern AI Application Systems: A Review of the Literature"
date: 2026-07-24 10:00:00 +0000
categories: [AI, Observability]
tags: [llm, observability, opentelemetry, agents, llmops, evaluation, literature-review]
description: "A survey of the literature on observability for LLM and agentic AI systems — from distributed tracing and ML monitoring foundations to evaluation, agent tracing, standards, and open research gaps."
---

*Compiled July 2026*

## 1. Introduction

The rapid adoption of large language models (LLMs) and autonomous agents in production software has created a new engineering discipline at the intersection of classical distributed-systems observability, machine-learning monitoring, and AI evaluation research. Unlike conventional software, AI application systems are non-deterministic, prompt-sensitive, and prone to failure modes — hallucination, drift, prompt injection, silent tool failures — that produce no stack trace and no HTTP 500. Understanding *why* such a system misbehaved therefore requires telemetry that captures not only latency and errors but semantic content: prompts, completions, retrieved context, tool invocations, reasoning steps, and quality scores.

This review surveys the literature on observability of modern AI application systems across four bodies of work: (1) the classical observability and ML-monitoring foundations on which the field is built; (2) academic research on monitoring, evaluating, and tracing LLM-based and agentic systems; (3) the industry platforms and practitioner literature that have largely defined the category's vocabulary; and (4) emerging standards and regulatory frameworks that are converging on a common telemetry model. The review closes with a synthesis of convergent themes and open research gaps.

A note on method and source quality: because this field is young (the majority of dedicated work post-dates 2023), a substantial fraction of the influential literature is *grey* — vendor whitepapers, engineering blogs, open-source specifications, and arXiv preprints — rather than peer-reviewed. Following the convention of multivocal literature reviews in software engineering, such sources are included where they are demonstrably category-defining, but their status is flagged throughout.

## 2. Foundations: Observability in Distributed Systems

Modern AI observability inherits its conceptual machinery from two decades of distributed-systems work. The trace/span model originates with Google's Dapper (Sigelman et al., 2010), which established distributed tracing with low overhead, application-level transparency, and context propagation across RPC boundaries — the direct ancestor of Zipkin, Jaeger, and ultimately OpenTelemetry. The Google SRE book's monitoring chapter (Beyer et al., 2016) contributed the "four golden signals" — latency, traffic, errors, saturation — as the minimal metric set for any user-facing system, while Sridharan (2018) popularized the "three pillars" framing of logs, metrics, and traces as complementary telemetry types.

These threads consolidated into OpenTelemetry (OTel), formed in 2019 from the merger of OpenTracing and OpenCensus and graduated as a CNCF project in 2026, which provides the vendor-neutral APIs, SDKs, and wire protocol (OTLP) that essentially all AI-observability tooling now builds upon. The W3C Trace Context recommendation (2020) standardized the `traceparent` header for cross-service trace propagation — a mechanism that becomes newly relevant (and newly problematic) when agent-to-tool call chains traverse non-HTTP transports (see §7.3).

The significance of this lineage for AI systems is straightforward: an LLM application is a distributed system whose "requests" happen to carry multi-kilobyte natural-language payloads. The trace/span abstraction transfers directly — a RAG pipeline or agent run maps naturally onto a span tree — but, as Traceloop's founders observed, "nobody designed telemetry for multi-kilobyte prompts," motivating the AI-specific extensions surveyed in §8.

## 3. Pre-LLM Machine Learning Monitoring

A second foundation is the ML-systems literature of 2015–2022, which established that deployed models fail in ways invisible to conventional monitoring.

**Technical debt and production readiness.** Sculley et al. (2015), in the canonical "Hidden Technical Debt in Machine Learning Systems," argued that most of an ML system's failure surface lies outside the model code: entanglement, hidden feedback loops, undeclared consumers, data dependencies, and changes in the external world. Breck et al. (2017) operationalized this into the ML Test Score, a rubric of 28 tests and monitoring needs drawn from Google's production systems, whose monitoring section (training/serving skew, feature staleness, prediction-quality degradation) is arguably the first concrete checklist of what to observe in a deployed model. Breck et al. (2019) subsequently described TFX's data-validation system, establishing continuous data-quality monitoring — schema validation, skew detection, anomaly alerting — as a first-class pillar of ML observability.

**Drift detection.** The statistical layer of model monitoring rests on the concept-drift literature, comprehensively surveyed by Lu et al. (2019), who frame the problem as detection, understanding, and adaptation. Rabanser, Günnemann, and Lipton (2019) empirically benchmarked practical dataset-shift detection pipelines (dimensionality reduction plus two-sample tests) under the telling title "Failing Loudly" — ML systems fail *silently* by default, and the purpose of monitoring is to make them fail loudly. Klaise et al. (2020) bridged this research to deployable architecture, describing a production stack of performance monitoring, outlier detection, drift detection, and prediction explanation. Huyen (2022, Ch. 8) provides the standard textbook treatment of the drift taxonomy (covariate shift, label shift, concept drift) and detection statistics.

**From monitoring to observability.** Shankar and Parameswaran (2022) were among the first to use "observability" as an explicit framing for ML, proposing a bolt-on data-management system supporting detection, diagnosis, and *reaction* to ML bugs under delayed or missing labels — a useful conceptual demarcation between passively watching dashboards (monitoring) and being able to interrogate a system about arbitrary novel failures (observability). Complementary interview studies (Shankar et al., 2022) documented that practitioners experience production ML as a continual loop of experimentation, evaluation, and monitoring, with monitoring the dominant pain point — "we have no idea how models will behave in production until production."

**MLOps as organizing frame.** Kreuzberger, Kühl, and Hirschl (2023) produced the canonical definition and reference architecture of MLOps via a mixed-method study, situating monitoring and continuous feedback loops among its core principles. This lifecycle framing carries forward into LLMOps (§4).

## 4. From MLOps to LLMOps: The New Problem Space

The arrival of foundation-model applications changed the monitoring problem qualitatively. Classical ML monitoring assumes a fixed model, structured features, and (eventually) ground-truth labels; LLM applications feature prompt-driven behavior, unbounded text I/O, third-party API-served models with no access to internals, and no ground truth at all for open-ended generation.

Empirical studies document the resulting gaps. Xia et al. (2024) analyzed over 29,000 questions from the OpenAI developer forum to build a taxonomy of LLM application-development challenges, including prominent debugging and monitoring difficulties. A practitioner-view synthesis (arXiv:2411.08574, 2024) similarly concluded that LLMs' prompt sensitivity and few-shot behavior create monitoring and evaluation concerns distinct from traditional ML. Emerging LLMOps literature (e.g., the 2024 "LLMOps: Definition, Challenges, and Lifecycle Management" formalization) extends the Kreuzberger et al. lifecycle to include prompt management, cost/latency monitoring, and continuous evaluation as first-class stages, though this strand is not yet anchored in top venues.

The most influential problem statements of this period, however, came from practice. Carter's "All the Hard Stuff Nobody Talks About when Building Products with LLMs" (Honeycomb, 2023) — drawn from shipping one of the first production LLM features — enumerated six hard problems (context windows, prompt engineering, latency, correctness, security, compliance) and argued that non-deterministic LLM features must be debugged from *real user telemetry* rather than offline testing. Majors and Carter's "Observability in the Age of AI" (2023) extended the thesis: models cannot be understood in isolation from the surrounding software system, so good AI observability presupposes good observability of everything else. Carter's O'Reilly report *Observability for Large Language Models* (2024) is the first book-length treatment, showing how to close a production feedback loop — capture user behavior, performance, and feedback as OpenTelemetry data, and use it to refine prompts and evals systematically. Huyen's *AI Engineering* (2025) provides the corresponding textbook chapter, notable for elevating *user feedback* (explicit and conversational) to a first-class telemetry signal.

## 5. LLM Observability: Signals, Architectures, and Platforms

Between 2023 and 2026, "LLM observability" crystallized as a product category, and the vendor literature — while promotional — did much of the definitional work.

**Signal taxonomy.** Across vendors, a consistent signal set has emerged: (a) *traces and spans* of LLM calls, retrievals, and tool invocations; (b) *token usage, cost, and latency* per call and per user/feature; (c) *full prompt/response payload logging*; (d) *evaluation scores* computed online (LLM-as-judge and code-based scorers); (e) *human feedback and annotations*; (f) *drift* over prompts, topics, and embeddings; and (g) *safety signals* — toxicity, PII exposure, prompt injection, jailbreaks, guardrail violations. Arize's "LLM Observability 101" whitepaper (2023) organized an early version of this into "five pillars" (evaluation, traces/spans, prompt engineering, retrieval/RAG, fine-tuning); Braintrust's guides (2025–26) articulate the now-standard formula that observability = tracing + evals + monitoring over shared data.

**Architectural patterns.** Three instrumentation architectures recur. *SDK/decorator instrumentation* (LangSmith, Langfuse, W&B Weave — whose `@weave.op` decorator yields automatic hierarchical call trees) embeds tracing in application code. *Proxy/gateway interception* (Helicone) captures everything by routing LLM traffic through a logging gateway — architecturally simple, but coupling observability to the request path; Helicone's 2026 acquisition and move to maintenance mode is itself a data point on consolidation. *OTel-native auto-instrumentation* (OpenLLMetry, OpenLIT, HoneyHive) patches provider SDKs to emit standard OTLP, decoupling instrumentation from backend. The three approaches are converging on OTel semantics as the interoperability layer (§8).

**Platform landscape.** The LLM-native platforms each contributed distinct concepts: LangSmith (LangChain) established that *the same traces powering observability should power evaluation*, with annotation queues for human labeling; Langfuse provides the leading open-source, self-hostable implementation with OTLP ingestion; Arize/Phoenix carried pre-LLM drift rigor into embedding-drift detection for prompts and responses; WhyLabs LangKit represents the "metrics-from-text" approach, extracting quality/sentiment/toxicity/PII signals from payloads; Galileo closed the evaluation-to-guardrail loop, running the same research-backed metrics (via small purpose-trained judge models) as sub-200ms production guardrails; TruLens introduced the widely adopted RAG Triad (context relevance, groundedness, answer relevance) as a reference-free RAG failure-mode framework; DeepEval/Confident AI framed evaluation as unit testing ("pytest for LLMs") wired into CI.

Meanwhile the APM incumbents mainstreamed the category: Datadog's LLM Observability reached general availability in 2024 with span-level input/output capture, token/cost tracking, online hallucination/toxicity/injection evaluations, and topic-drift detection, later renaming toward "Agent Observability" and adopting the OTel GenAI conventions natively; New Relic and Dynatrace ship comparable AI-monitoring capabilities inside classical APM. The second edition of *Observability Engineering* (Majors, Fong-Jones, Miranda, with Parker; 2026) — substantially rewritten "for an AI world," including chapters on debugging LLM applications in production — marks the absorption of AI observability into the observability canon.

## 6. Evaluation as an Observability Signal

What most distinguishes AI observability from its predecessors is that *output quality* must be measured continuously, without ground truth, on live traffic. A substantial academic literature underpins this layer.

**LLM-as-judge.** Zheng et al. (2023) provided the methodological foundation, showing that GPT-4 judges reach over 80% agreement with human preferences — comparable to human–human agreement — while systematically characterizing judge biases (position, verbosity, self-enhancement). Virtually every online-evaluation feature in the platforms of §5 operationalizes this result. An important corrective comes from Shankar et al. (2024), whose EvalGen study identified *criteria drift*: evaluation criteria cannot be fully specified a priori but co-evolve with observation of outputs, implying that LLM-judge-based monitors require continuous human-in-the-loop recalibration rather than set-and-forget deployment.

**Hallucination detection.** The comprehensive taxonomy of Huang et al. (2023/2024) maps what quality monitoring must detect. Two detection approaches are particularly relevant to production settings: SelfCheckGPT (Manakul, Liusie, and Gales, 2023) detects hallucinations black-box — via consistency across sampled responses — matching the API-served-model setting of most deployments; and Farquhar et al. (2024, *Nature*) introduced semantic entropy, an uncertainty measure over clusters of semantically equivalent generations, with follow-up semantic-entropy probes (Kossen et al., 2024) approximating the signal cheaply enough from hidden states for runtime use.

**Guardrails.** Runtime enforcement both constrains behavior and *generates telemetry* (policy-violation events) that monitoring consumes. NeMo Guardrails (Rebedea et al., 2023) is the canonical open toolkit for programmable rails; Dong, Mu, et al. (2024) survey the guardrail landscape and argue for a systematic socio-technical approach to their construction and verification.

**Evaluation frameworks.** An open-source ecosystem connects offline evaluation to online observability using shared metric definitions: OpenAI Evals established the "evals as code plus shared registry" pattern; Ragas (2023) contributed the standard reference-free RAG metrics (faithfulness, answer relevancy, context precision/recall) now routinely scored over production traces; DeepEval and promptfoo cover CI-style regression testing and adversarial red-teaming respectively; and MLflow's GenAI tracing/evaluation ties evaluation runs to OTel-compatible traces so quality failures can be root-caused span by span.

## 7. Agent and Multi-Agent Observability

The 2024–2026 shift from single-completion LLM features to autonomous, tool-using agents restated the observability problem again: the unit of analysis becomes a multi-step, non-deterministic *workflow* with planning, memory, tool calls, and inter-agent handoffs.

### 7.1 Taxonomies and failure modes

Dong, Lu, and Zhu (2024) provide the foundational academic treatment, "AgentOps: Enabling Observability of LLM Agents" (CSIRO Data61), a systematic mapping of AgentOps tooling that identifies the artifacts to be traced across the agent lifecycle — plans, tool invocations, memory operations, guardrail events — motivated explicitly by AI-safety needs. Cemri et al. (2025) contribute MAST, the first empirically grounded multi-agent failure taxonomy: 14 failure modes in three categories (system-design flaws, inter-agent misalignment, task-verification failures) derived from more than 1,600 annotated traces across seven frameworks. Such failure taxonomies function, in effect, as requirements specifications for what agent observability must be able to surface.

### 7.2 Trace analytics and automated debugging

Moshkovich et al. (2025, IBM Research) argue that black-box benchmarking cannot characterize non-deterministic agentic systems and propose analytics built from runtime logs (flow discovery, issue analytics), with follow-up work framing a full observe–analyze–optimize loop. Deshpande et al. (2025) introduce TRAIL, a benchmark of 148 human-annotated agent traces containing 841 errors under a formal taxonomy; the best evaluated model localizes errors with only ~11% success, quantifying how far automated root-cause analysis over agent traces remains from practical reliability. At the systems layer, AgentSight (Zheng et al., 2025) demonstrates instrumentation-free observability via eBPF "boundary tracing" — intercepting TLS-encrypted LLM traffic for semantic intent and kernel events for system effects, causally correlating the two with under 3% overhead — addressing the semantic gap between what an agent *says* it is doing and what the operating system observes it *actually* doing.

### 7.3 Framework and protocol instrumentation

On the practice side, OpenAI's Agents SDK ships tracing on by default — structured spans for model calls, tool calls, handoffs, and notably *guardrail* executions — with pluggable exporters consumed by third-party platforms; LangGraph pairs deterministic graph execution with step-level state-transition traces and replay. The Model Context Protocol (MCP) has surfaced a distinctive gap: MCP servers are opaque to the calling agent (only request and final response are visible), hiding tool-internal failures, cascading downstream errors, and retry loops. Active work includes an official proposal to propagate OTel trace context through MCP's `_meta` mechanism — necessary because MCP transports (stdio, WebSockets) lack the HTTP headers W3C Trace Context assumes — and in-development OTel semantic conventions for MCP operations.

## 8. Standards and Interoperability

A defining development of 2024–2026 is convergence on OpenTelemetry as the substrate for AI telemetry.

The OTel **Generative AI semantic conventions** (`gen_ai.*`), developed by a dedicated SIG since April 2024, standardize the schema for LLM inference spans (`gen_ai.request.model`, `gen_ai.usage.input_tokens`/`output_tokens`, `gen_ai.response.finish_reasons`, structured input/output messages), client metrics (operation duration; token usage by type), and span kinds spanning `invoke_agent` → `chat` → `execute_tool` hierarchies. The conventions remain in *Development* (experimental) status — recently moved to a dedicated repository, reflecting rapid iteration — and should be cited as an emerging rather than finished standard; nonetheless, adoption is broad: Datadog, New Relic, and Honeycomb consume them, and frameworks emit them natively or via instrumentation packages. Parallel SIG work extends the conventions to agentic systems (tasks, actions, teams, memory) and MCP.

Two open specifications preceded and shaped this effort. **OpenLLMetry** (Traceloop, 2023) provided OTel-compatible instrumentation for LLM providers, vector databases, and frameworks, and co-led the semantic-conventions working group; **OpenInference** (Arize) defines a complementary attribute schema whose span-kind taxonomy (LLM, agent, tool, retriever, embedding, chain) underlies Phoenix and remains the principal alternative convention. MLflow Tracing documents explicit alignment with the OTel GenAI conventions, and Langfuse ingests OTLP natively — evidence that the ecosystem has effectively settled on OTLP as the interchange format even while the attribute schema stabilizes.

## 9. Regulatory and Governance Drivers

A final strand of literature reframes observability from engineering best practice to compliance obligation. The **EU AI Act** (Regulation 2024/1689) is the strongest driver: Article 12 mandates that high-risk AI systems technically enable automatic event logging over the system lifetime (minimum six-month retention, with per-use logging detail for biometric systems), and Article 72 requires providers to operate a documented post-market monitoring system collecting operational data. The **NIST AI Risk Management Framework** (2023) and its Generative AI Profile (NIST AI 600-1, 2024) define, through the MEASURE and MANAGE functions, what an AI observability program must be able to quantify — cataloguing twelve GenAI risks including confabulation, harmful content, and information-security failures. **ISO/IEC 42001:2023**, the first certifiable AI management-system standard, requires monitoring, measurement, analysis, and evaluation of AI systems (Clause 9), making observability data audit evidence. Collectively these instruments guarantee that the tracing and logging capabilities surveyed above are not optional for regulated deployments.

## 10. Synthesis and Research Gaps

**Convergent themes.** Four run through the literature. First, *continuity with lineage*: AI observability is best understood as the composition of distributed tracing (Dapper → OTel), ML monitoring (technical debt → drift detection), and LLM evaluation research — each surveyed strand supplies one layer. Second, *the trace as unifying abstraction*: across academic and industrial work, the span tree of an LLM/agent execution is the common data structure onto which cost, latency, payloads, eval scores, feedback, and safety signals attach; the "production traces → curated eval datasets → improved system" loop has become the standard workflow. Third, *evaluation is the differentiator*: what separates AI observability from APM is continuous, ground-truth-free quality measurement, resting academically on LLM-as-judge, self-consistency, and uncertainty methods — each with documented reliability caveats. Fourth, *standardization and consolidation*: the field is converging on OTel/OTLP with `gen_ai.*` conventions as schema, while the vendor landscape consolidates and regulation converts logging into legal obligation.

**Open gaps** relevant to a research project in this area:

1. **No mature, peer-reviewed survey of the whole field exists.** The closest candidates are the AgentOps taxonomy (Dong et al., 2024), MAST (Cemri et al., 2025), and a non-peer-reviewed multi-layer synthesis (Sisodia, 2026); a systematic (multivocal) literature review of AI-application observability remains an open contribution.
2. **Automated root-cause analysis over agent traces is unsolved.** TRAIL shows state-of-the-art models localize trace errors with ~11% accuracy — collecting rich traces has far outpaced the ability to interpret them automatically.
3. **Judge reliability and criteria drift.** Online evaluation inherits the biases of LLM judges, and criteria co-evolve with observed outputs (Shankar et al., 2024); principled methods for calibrating and maintaining production monitors are underdeveloped.
4. **Cost and overhead of semantic monitoring.** Uncertainty methods like semantic entropy impose multi-fold inference costs; cheap approximations (probes, small judge models) are early-stage.
5. **Cross-boundary trace propagation for agents.** Connecting agent-side and tool/MCP-server-side telemetry across non-HTTP transports is an open protocol problem, as is correlating semantic intent with system-level effects (the AgentSight "semantic gap").
6. **Standards immaturity.** The OTel GenAI, agent, and MCP conventions are experimental; schema churn and the OpenInference/`gen_ai.*` split leave interoperability incomplete.
7. **Privacy and payload tension.** Effective LLM observability wants full prompt/response capture; privacy regulation and PII exposure argue against it. Principled redaction/sampling strategies that preserve debuggability are largely unstudied academically.

## References

Peer-reviewed and archival sources are listed first; grey literature (specifications, vendor whitepapers, blogs, books) follows. URLs verified via web search, July 2026.

### Academic literature

1. Sigelman, B. H., Barroso, L. A., Burrows, M., et al. (2010). *Dapper, a Large-Scale Distributed Systems Tracing Infrastructure.* Google Technical Report. https://research.google/pubs/dapper-a-large-scale-distributed-systems-tracing-infrastructure/
2. Sculley, D., Holt, G., Golovin, D., et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS 2015. https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems
3. Breck, E., Cai, S., Nielsen, E., Salib, M., Sculley, D. (2017). *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction.* IEEE Big Data 2017. https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
4. Breck, E., Polyzotis, N., Roy, S., Whang, S. E., Zinkevich, M. (2019). *Data Validation for Machine Learning.* MLSys 2019. https://proceedings.mlsys.org/paper_files/paper/2019/hash/928f1160e52192e3e0017fb63ab65391-Abstract.html
5. Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., Zhang, G. (2019). *Learning under Concept Drift: A Review.* IEEE TKDE 31(12). https://arxiv.org/abs/2004.05785
6. Rabanser, S., Günnemann, S., Lipton, Z. C. (2019). *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.* NeurIPS 2019. https://arxiv.org/abs/1810.11953
7. Klaise, J., Van Looveren, A., Cox, C., Vacanti, G., Coca, A. (2020). *Monitoring and explainability of models in production.* ICML 2020 Workshop. https://arxiv.org/abs/2007.06299
8. Shankar, S., Parameswaran, A. (2022). *Towards Observability for Production Machine Learning Pipelines.* PVLDB 15(13). https://arxiv.org/abs/2108.13557
9. Shankar, S., Garcia, R., Hellerstein, J. M., Parameswaran, A. G. (2022). *Operationalizing Machine Learning: An Interview Study.* arXiv:2209.09125. https://arxiv.org/abs/2209.09125
10. Kreuzberger, D., Kühl, N., Hirschl, S. (2023). *Machine Learning Operations (MLOps): Overview, Definition, and Architecture.* IEEE Access 11. DOI 10.1109/ACCESS.2023.3262138
11. Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023 (D&B). https://arxiv.org/abs/2306.05685
12. Manakul, P., Liusie, A., Gales, M. J. F. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP 2023. https://aclanthology.org/2023.emnlp-main.557/
13. Rebedea, T., Dinu, R., Sreedhar, M., Parisien, C., Cohen, J. (2023). *NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails.* EMNLP 2023 Demos. https://arxiv.org/abs/2310.10501
14. Huang, L., et al. (2024). *A Survey on Hallucination in Large Language Models.* ACM TOIS. https://arxiv.org/abs/2311.05232
15. Farquhar, S., Kossen, J., Kuhn, L., Gal, Y. (2024). *Detecting hallucinations in large language models using semantic entropy.* Nature 630, 625–630. https://www.nature.com/articles/s41586-024-07421-0
16. Kossen, J., et al. (2024). *Semantic Entropy Probes.* arXiv:2406.15927. https://arxiv.org/abs/2406.15927
17. Dong, Y., Mu, R., Jin, G., et al. (2024). *Building Guardrails for Large Language Models.* ICML 2024 (position). https://arxiv.org/abs/2402.01822
18. Shankar, S., et al. (2024). *Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences.* UIST 2024. https://arxiv.org/abs/2404.12272
19. Xia, X., et al. (2024). *An Empirical Study on Challenges for LLM Application Developers.* arXiv:2408.05002. https://arxiv.org/abs/2408.05002
20. Anon. (2024). *Themes of Building LLM-based Applications for Production: A Practitioner's View.* arXiv:2411.08574. https://arxiv.org/abs/2411.08574
21. Dong, L., Lu, Q., Zhu, L. (2024). *AgentOps: Enabling Observability of LLM Agents.* arXiv:2411.05285 (CSIRO Data61). https://arxiv.org/abs/2411.05285
22. Moshkovich, D., Mulian, H., Zeltyn, S., et al. (2025). *Beyond Black-Box Benchmarking: Observability, Analytics, and Optimization of Agentic Systems.* arXiv:2503.06745. https://arxiv.org/abs/2503.06745
23. Moshkovich, D., Zeltyn, S. (2025). *Taming Uncertainty via Automation: Observing, Analyzing, and Optimizing Agentic AI Systems.* arXiv:2507.11277. https://arxiv.org/abs/2507.11277
24. Cemri, M., Pan, M. Z., Yang, S., et al. (2025). *Why Do Multi-Agent LLM Systems Fail?* arXiv:2503.13657. https://arxiv.org/abs/2503.13657
25. Deshpande, D., Gangal, V., Mehta, H., et al. (2025). *TRAIL: Trace Reasoning and Agentic Issue Localization.* arXiv:2505.08638. https://arxiv.org/abs/2505.08638
26. Zheng, Y., et al. (2025). *AgentSight: System-Level Observability for AI Agents Using eBPF.* arXiv:2508.02736. https://arxiv.org/abs/2508.02736
27. Zhang, et al. (2025). *A Survey of AIOps in the Era of Large Language Models.* arXiv:2507.12472. https://arxiv.org/abs/2507.12472
28. Sisodia, T. (2026). *AI Observability for Large Language Model Systems: A Multi-Layer Analysis.* arXiv:2604.26152 (preprint, not peer-reviewed). https://arxiv.org/abs/2604.26152

### Standards, specifications, and regulation

29. OpenTelemetry. *Semantic Conventions for Generative AI* (spans, metrics; status: Development). https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/ ; https://github.com/open-telemetry/semantic-conventions-genai
30. OpenTelemetry (2025). *AI Agent Observability — Evolving Standards and Best Practices.* https://opentelemetry.io/blog/2025/ai-agent-observability/
31. OpenTelemetry (2026). *Inside the LLM Call: GenAI Observability with OpenTelemetry.* https://opentelemetry.io/blog/2026/genai-observability/
32. OpenTelemetry. *Semantic Conventions for Model Context Protocol* (Development). https://opentelemetry.io/docs/specs/semconv/gen-ai/mcp/
33. W3C (2020). *Trace Context.* W3C Recommendation. https://www.w3.org/TR/trace-context/
34. Traceloop. *OpenLLMetry.* https://github.com/traceloop/openllmetry
35. Arize AI. *OpenInference Specification.* https://github.com/Arize-ai/openinference
36. Model Context Protocol. *OpenTelemetry trace support* (discussion #269). https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/269
37. European Union (2024). *AI Act, Regulation (EU) 2024/1689* — Art. 12 (Record-keeping), Art. 72 (Post-market monitoring). https://artificialintelligenceact.eu/article/12/
38. NIST (2023/2024). *AI Risk Management Framework 1.0* and *Generative AI Profile (NIST AI 600-1).* https://www.nist.gov/itl/ai-risk-management-framework
39. ISO/IEC (2023). *ISO/IEC 42001: AI Management Systems.* https://www.iso.org/standard/42001

### Books and long-form practitioner literature

40. Beyer, B., Jones, C., Petoff, J., Murphy, N. R., eds. (2016). *Site Reliability Engineering*, Ch. 6. O'Reilly. https://sre.google/sre-book/monitoring-distributed-systems/
41. Sridharan, C. (2018). *Distributed Systems Observability.* O'Reilly. https://www.oreilly.com/library/view/distributed-systems-observability/9781492033431/
42. Huyen, C. (2022). *Designing Machine Learning Systems.* O'Reilly. (Ch. 8: Data Distribution Shifts and Monitoring.) https://huyenchip.com/2022/02/07/data-distribution-shifts-and-monitoring.html
43. Carter, P. (2024). *Observability for Large Language Models.* O'Reilly. https://www.oreilly.com/library/view/observability-for-large/9781098159757/
44. Huyen, C. (2025). *AI Engineering: Building Applications with Foundation Models.* O'Reilly. https://www.oreilly.com/library/view/ai-engineering/9781098166298/
45. Majors, C., Fong-Jones, L., Miranda, G., with Parker, A. (2026). *Observability Engineering*, 2nd ed. O'Reilly. https://www.oreilly.com/library/view/observability-engineering-2nd/9781098179915/

### Category-defining industry sources (grey literature)

46. Carter, P. (2023). *All the Hard Stuff Nobody Talks About when Building Products with LLMs.* Honeycomb. https://www.honeycomb.io/blog/hard-stuff-nobody-talks-about-llm
47. Majors, C., Carter, P. (2023). *Observability in the Age of AI.* Honeycomb. https://www.honeycomb.io/blog/observability-age-of-ai
48. Arize AI (2023). *LLM Observability 101* (whitepaper). https://arize.com/wp-content/uploads/2023/11/LLM-Observability-101-1.pdf
49. Datadog (2024). *LLM Observability GA*; (2025) *OTel GenAI semantic-convention support.* https://docs.datadoghq.com/llm_observability/ ; https://www.datadoghq.com/blog/llm-otel-semantic-convention/
50. Braintrust (2025–26). *What is LLM observability?* ; *Agent observability: the complete guide for 2026.* https://www.braintrust.dev/articles/llm-observability-guide ; https://www.braintrust.dev/articles/agent-observability-complete-guide-2026
51. LangChain (2025). *Agent Observability Powers Agent Evaluation.* https://www.langchain.com/blog/agent-observability-powers-agent-evaluation
52. OpenAI. *Agents SDK — Tracing.* https://openai.github.io/openai-agents-python/tracing/
53. Es, S., et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* https://www.ragas.io/
54. TruLens. *The RAG Triad.* https://www.trulens.org/getting_started/core_concepts/rag_triad/
55. Langfuse. *Open-source LLM engineering platform.* https://github.com/langfuse/langfuse
56. Grafana Labs / OpenLIT (2025). *AI observability for MCP servers.* https://grafana.com/blog/ai-observability-MCP-servers/
