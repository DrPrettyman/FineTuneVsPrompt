# P1: Domain-Specific Document Classifier — Fine-Tune vs. Prompt Benchmark

## Full Project Plan

---

## 1. The Business Problem

An ML team is deploying text classification in production — routing customer complaints, triaging support tickets, categorising medical documents, classifying regulatory filings. They face a decision that recurs at every company adopting NLP:

- Should we fine-tune a small model (DistilBERT, DeBERTa, a 7B LLM with LoRA) on our labelled data?
- Or should we pay for LLM API calls (GPT-4, Claude) using zero-shot or few-shot prompting?
- At what volume does fine-tuning break even on cost?
- What accuracy do we sacrifice — or gain — with each approach?
- How does this change as our labelled dataset grows from 100 to 100,000 examples?

Today, answering these questions requires either:

- **Cloud ML consultancies** billing €200–400/hour to run benchmarks and advise on architecture
- **Internal ML platform teams** spending weeks running experiments before committing to an approach
- **Enterprise MLOps platforms** (AWS SageMaker, Google Vertex AI, Azure ML) that offer fine-tuning services but don't provide the decision framework for when to use them
- **Trial and error** — teams pick an approach, discover its limitations in production, then switch (expensive)

The cost of getting this wrong is real. A company classifying 100K documents/month that chooses GPT-4 API calls when a $8-trained LoRA adapter would outperform it is overspending by orders of magnitude. Conversely, a team that invests weeks fine-tuning a model for a low-volume task where zero-shot prompting would suffice has wasted engineering time.

**Resume line (draft):** "Built an interactive decision framework for fine-tuning vs. LLM API classification: trained DistilBERT, DeBERTa, and Mistral-7B (QLoRA) on CFPB consumer complaints across 6 training set sizes, benchmarked against GPT-4/Claude zero-shot. Produced learning curves, cost crossover charts, and latency analysis — fine-tuned DeBERTa matched GPT-4 accuracy at 1K examples and was 94% cheaper at scale. Deployed as Streamlit dashboard."

---

## 2. Why This Problem, For This Portfolio

### Strategic fit

**The most common production ML question.** "Build or buy?" is the question every ML engineering manager asks. A project that produces a reusable decision framework — not just "I fine-tuned a model" — demonstrates the kind of production thinking that engineering managers value. The portfolio principles reference (Principle 4) explicitly calls out cost/latency analysis as a key differentiator for LLM projects.

**PyTorch depth without being a pure benchmark.** The project requires custom training loops, LoRA adapter mechanics, GPU memory management, gradient accumulation, learning rate scheduling, and inference benchmarking — all core PyTorch skills. But the deliverable is a business tool (decision framework), not an academic benchmark.

**Domain alignment.** Using the CFPB Consumer Complaints Database ties to fintech (Revolut-relevant). Using PubMed abstracts would tie to health/biotech (Clarity AI-relevant). The dataset choice can be optimised for target employer alignment.

**Fills the "deep learning fine-tuning" gap.** The portfolio currently has classical ML (Steam, Private Label), LLM/RAG (L1), and time-series NLP (B1). None demonstrate hands-on PyTorch fine-tuning with transformers, which job descriptions increasingly list as a requirement.

### Portfolio overlap considerations

If B1 (Macro Surveillance) uses FinBERT as Method C, there is moderate overlap in "PyTorch + HuggingFace + transformer inference." However, P1 goes substantially deeper: custom training loops (vs. using a pretrained model), LoRA adapter configuration, training set size experiments, and GPU memory management. B1 uses FinBERT as a tool; P1 is *about* the fine-tuning process itself.

---

## 3. Novelty Assessment

### ⚠️ Critical Finding: This Space Is Heavily Saturated

Validation research revealed significant prior art across multiple dimensions. This section documents what exists honestly, then identifies the remaining novel contribution.

### What already exists

| Prior Art | What it does | Severity |
|---|---|---|
| **Predibase "LoRA Land"** (arXiv 2405.00732, May 2024) | 310 fine-tuned LoRA models across 10 base models (2B–7B) and 31 tasks, benchmarked against GPT-4. Interactive leaderboard. Cost analysis showing ~$8 per model to train. Fine-tuned 7B models outperform GPT-4 on 85% of specialised tasks. | **CRITICAL** — industrial-scale version of P1's core comparison. 700+ experiments. Published paper + interactive app. |
| **Stanford RCPedia** (Nov 2025) | Full benchmark comparing fine-tuned Qwen3-8B + LoRA vs GPT-4.1 mini vs Random Forest on Reddit classification. Includes cost analysis, accuracy comparison (base 41% → fine-tuned 78% → GPT-4.1 mini fine-tuned 89%). | **HIGH** — exact same framing as P1, including cost comparison. Blog post format, not a portfolio project. |
| **"Fine-Tuned 'Small' LLMs (Still) Significantly Outperform..."** (arXiv 2406.08660, Oct 2025) | Academic paper comparing RoBERTa, DeBERTa, ELECTRA, XLNet fine-tuned vs. GPT-3.5/GPT-4/Claude Opus zero-shot across 4 classification tasks. Includes learning curve plots (accuracy vs. training set size). Published toolkit. | **HIGH** — academic paper with learning curves, the exact analysis P1 proposes. |
| **"Think Before You Classify"** (MDPI Electronics, Mar 2025) | Evaluates 14 LLMs (GPT-4o, Claude 3.5, DeepSeek-V3, Gemini-2.0, o1, o3) zero-shot on CFPB consumer complaints classification. Same dataset, same task. | **HIGH** — exact dataset + zero-shot LLM evaluation already published. |
| **vilcek/fine-tuning-BERT-for-text-classification** (GitHub) | Complete tutorial using CFPB Consumer Complaints (472K rows) for product classification with DistilBERT + PyTorch. | **MEDIUM** — exact dataset + fine-tuning approach, but no LLM comparison or cost analysis. |
| **PMC clinical notes study** (JAMIA, Oct 2024) | GPT-4 vs PubMedBERT vs LSTM on breast cancer pathology classification. GPT-4 zero-shot matched or beat supervised models. | **MEDIUM** — biomedical domain, same comparison pattern. |
| **Codecademy, Google ML Crash Course, Nexla, Microsoft Foundry, Moveo.AI** (various 2024–2025) | Enterprise guides covering fine-tune vs. prompt decision framework with cost/accuracy trade-offs. | **MEDIUM** — conceptual framework widely covered, though not as portfolio projects with code. |
| **Pullflow blog** (2025) | Explicit break-even analysis: "~9 months assuming stable volume." | **LOW** — blog post, not a code project. But same conceptual contribution. |

### What does NOT exist (remaining novel space)

Despite the saturation, a specific gap remains:

1. **No portfolio project combines learning curves + cost crossover + interactive tool.** LoRA Land tested accuracy at full training set size only — it did NOT produce learning curves (accuracy vs. N at 10, 100, 1K, 10K, 100K). The arXiv paper (2406.08660) has learning curves but no cost crossover analysis or interactive tool. Stanford RCPedia has cost analysis but no learning curves. Nobody has combined all three into a single interactive deliverable.

2. **No one has produced a *reusable* decision framework.** All existing work presents findings for their specific datasets. Nobody has built a tool where you input your parameters (volume, labelled data size, accuracy requirements, budget) and get a recommendation.

3. **The CFPB + multi-architecture + cost crossover combination** is untrodden as a single project. Individual pieces exist everywhere, but the specific combination of CFPB data + BERT-class + 7B LoRA + GPT-4/Claude zero-shot + learning curves + cost analysis + interactive dashboard has not been assembled.

### Honest assessment

The novel contribution is **narrow but real**: it sits in the intersection of learning curve analysis, cost crossover modelling, and interactive tooling. The risk is that a reviewer sees "fine-tuned BERT on CFPB" and dismisses it before reaching the novel parts. **The framing must lead with the decision framework, not the fine-tuning.**

---

## 4. Data Sources

### Primary: CFPB Consumer Complaints Database

**What it is:** The Consumer Financial Protection Bureau maintains a public database of consumer complaints about financial products and services. Each complaint includes the consumer's narrative text and is labelled with product category, sub-product, issue type, and company.

**Access:** Freely downloadable from data.gov and the CFPB website. REST API available for programmatic access. Pre-processed versions on Kaggle.

**Scale:** ~4 million complaints total (as of Feb 2026). After filtering for complaints with narratives and standard product labels: ~500K–1M usable records.

**Labels for classification:**
- **Product** (primary): 18 categories (Credit reporting, Debt collection, Mortgage, Credit card, Bank account, Student loan, etc.)
- **Sub-product:** ~70 sub-categories (more granular, useful for hierarchical classification experiments)
- **Issue:** ~100 issue types (can serve as a harder multi-label variant)

**Why this dataset:**
- Fintech domain aligns with target employers (Revolut)
- Large enough to produce meaningful learning curves across 5 orders of magnitude (10 → 100K)
- Multi-class with natural class imbalance — realistic production scenario
- Real consumer language (messy, informal, variable length) — not sanitised academic text
- Free-text narratives require genuine NLP, not just keyword matching

**Political context (documented for completeness):** The CFPB faced significant political turbulence in 2025 — DOGE shutdown attempts, mass firing orders, union lawsuits, funding disputes. As of Feb 2026, the agency remains operational (DC Circuit restored funding injunction Dec 2025; Congress passed continued appropriations). The database is actively maintained. **Risk to project: LOW.** Historical data (millions of records) is already downloadable regardless of future agency status. The dataset is a static download, not a live dependency.

### Alternative: PubMed Abstracts with MeSH Headings

**What it is:** 35M+ biomedical abstracts from the NCBI PubMed database, each labelled with Medical Subject Headings (MeSH) — a hierarchical vocabulary of ~30,000 terms.

**Access:** Annual baseline XML dump from NCBI FTP (free, no authentication). Pre-processed version on HuggingFace (`ncbi/pubmed`). Kaggle dataset available for multi-label MeSH classification.

**Why it's the backup, not the primary:**
- MeSH classification is an extreme multi-label problem (~30K labels) with decades of academic literature. BioASQ has run annual challenges on this since 2013. State-of-the-art is ~0.6 F-measure.
- Domain-specific pretrained models already exist and are heavily benchmarked: PubMedBERT, BioBERT, Bioformer, SciBERT, BioGPT.
- Multiple academic papers already compare fine-tuned PubMedBERT vs. GPT-4 zero-shot on biomedical classification tasks (JAMIA 2024, Bioinformatics 2024).
- Competing against years of academic research with a portfolio project is a losing position.

**PubMed would make sense if:** The classification task is scoped to a manageable subset (e.g., top-level MeSH categories only, or a specific subdomain like cardiovascular diseases using the Ohsumed dataset), and the portfolio targets health/biotech employers specifically.

### Tertiary alternatives (not researched in depth)

- **arXiv paper classification:** Papers labelled by subject category. Less saturated than CFPB or PubMed for the fine-tune-vs-prompt comparison. Clean labels, large scale.
- **Patent classification (CPC codes):** Google Patents on BigQuery. Hierarchical codes, millions of patents. Interesting for IP/legal domain alignment but data access is more complex.

---

## 5. What the System Builds

### The Deliverable: An Interactive Decision Framework

The project produces three outputs:

**Output 1: Learning Curve Analysis**

For each model architecture, measure classification accuracy (macro-F1, weighted-F1, per-class precision/recall) at 6 training set sizes:

| Training set size | DistilBERT | DeBERTa-v3-base | Mistral-7B (QLoRA) | GPT-4 zero-shot | GPT-4 few-shot (k=5) | Claude zero-shot |
|---|---|---|---|---|---|---|
| 10 | ? | ? | ? | baseline | baseline+k | baseline |
| 100 | ? | ? | ? | (same) | baseline+k | (same) |
| 1,000 | ? | ? | ? | (same) | baseline+k | (same) |
| 10,000 | ? | ? | ? | (same) | baseline+k | (same) |
| 50,000 | ? | ? | ? | (same) | baseline+k | (same) |
| 100,000 | ? | ? | ? | (same) | baseline+k | (same) |

Each cell is the mean ± std of 3 runs with different random seeds. The zero-shot/few-shot LLM baselines are constant (they don't train on the data), so they appear as horizontal lines on the learning curve plot.

**Key question answered:** "At what training set size does a fine-tuned small model match or exceed GPT-4 zero-shot accuracy?"

**Output 2: Cost Crossover Analysis**

For each model, compute:
- **Training cost:** GPU-hours × spot price (measured on Google Colab T4 / Kaggle P100)
- **Inference cost per classification:** GPU-hours for local model OR API price per token for LLM
- **Monthly cost at volume V:** Training amortisation (over 3 months) + inference cost × V

Produce a crossover chart: at what monthly classification volume does fine-tuning become cheaper than API calls?

```
Monthly cost ($)
│
│  GPT-4 ╱
│       ╱
│      ╱        ← crossover point
│     ╱─────────── Fine-tuned DeBERTa (flat after training amortised)
│    ╱
│   ╱
│──╱
└──────────────────────── Monthly volume
```

**Key question answered:** "At N classifications/month, fine-tuning saves $X/month compared to API calls."

**Output 3: Latency Comparison**

Measure and report:
- Median latency per classification (ms) for each model
- P95 and P99 latency
- Throughput (classifications/second) on standardised hardware

This matters because some production systems require <100ms per classification (e.g., real-time routing), which eliminates API-based approaches regardless of cost.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
│                                                             │
│  CFPB Data ─→ Stratified Subsample (10, 100, 1K, ..., 100K)│
│       │                                                      │
│       ├─→ DistilBERT fine-tune (HF Trainer, full fine-tune) │
│       ├─→ DeBERTa-v3-base fine-tune (HF Trainer)            │
│       ├─→ Mistral-7B QLoRA (PEFT + bitsandbytes)            │
│       │                                                      │
│       │   For each: 3 seeds × 6 sizes = 18 training runs    │
│       │   Total: 54 fine-tuning runs + 1 zero-shot eval     │
│       │                                                      │
│       └─→ GPT-4 / Claude zero-shot + few-shot evaluation    │
│           (API calls on held-out test set)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Pipeline                       │
│                                                             │
│  Held-out test set (fixed across all experiments)           │
│                                                             │
│  For each model × training size:                            │
│    - Macro-F1, Weighted-F1                                  │
│    - Per-class precision, recall, F1                        │
│    - Confusion matrix                                       │
│    - Confidence calibration (predicted prob vs. actual)     │
│                                                             │
│  GPU time logging: training time, inference time per sample │
│  API cost logging: tokens in/out, price per call            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Cost-Benefit Analysis Engine                 │
│                                                             │
│  Inputs: accuracy results + timing data + API pricing       │
│                                                             │
│  Produces:                                                  │
│    - Learning curves (accuracy vs. training set size)       │
│    - Cost crossover charts (monthly cost vs. volume)        │
│    - Latency comparison table                               │
│    - Break-even calculator (parameterised)                  │
│    - Recommendation matrix:                                 │
│                                                             │
│      Volume  | Labelled data | Budget  | → Recommendation   │
│      Low     | <100          | Low     | → GPT-4 zero-shot  │
│      Low     | >1K           | Low     | → Fine-tuned BERT  │
│      High    | >1K           | Any     | → Fine-tuned BERT  │
│      High    | <100          | High    | → GPT-4 few-shot   │
│                                                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard                         │
│                                                             │
│  Interactive exploration:                                   │
│    - Toggle models on/off in learning curve plot            │
│    - Slider: monthly classification volume                  │
│    - Slider: available labelled training data               │
│    - Input: max acceptable latency                          │
│    - Output: recommended approach + estimated monthly cost  │
│                                                             │
│  Static results:                                            │
│    - Per-class performance heatmap                          │
│    - Confusion matrices (side by side)                      │
│    - Latency distribution plots                             │
│    - Cost breakdown table                                   │
└─────────────────────────────────────────────────────────────┘
```

### PyTorch Components (Used Naturally)

| Component | Purpose | Why it's the right tool |
|---|---|---|
| **HuggingFace Transformers** (PyTorch backend) | Model loading, tokenisation, fine-tuning | Industry-standard library for transformer fine-tuning |
| **Custom training loop** (for DistilBERT) | Full control over training dynamics, gradient accumulation, mixed precision | Demonstrates deeper PyTorch understanding than using Trainer alone |
| **HF Trainer** (for DeBERTa) | Efficient training with built-in evaluation callbacks | Shows familiarity with both approaches |
| **PEFT + bitsandbytes** | QLoRA fine-tuning of Mistral-7B in 4-bit | The practical way to fine-tune 7B models on a T4 GPU (16GB VRAM) |
| **Learning rate scheduling** | Warmup + cosine/linear decay | Standard practice, but must be tuned per architecture |
| **Gradient accumulation** | Effective batch size > GPU memory allows | Essential for 7B model on consumer GPU |
| **TorchMetrics** | Per-class precision, recall, F1; confusion matrices | Clean metric computation with proper averaging |
| **torch.cuda.Event** | Precise inference latency measurement | More accurate than wall-clock timing for GPU inference |
| **ONNX export** (optional) | Export fine-tuned model for optimised inference | Demonstrates deployment awareness |

### What this is NOT

- Not a model comparison leaderboard. The point is not "DeBERTa beats DistilBERT" — it's "here's when each approach makes business sense."
- Not a prompting tutorial. Zero-shot and few-shot prompting are baselines, not the subject of the project.
- Not trying to beat LoRA Land's 700 experiments. We test 3 architectures × 6 sizes × 3 seeds = 54 runs. The contribution is the decision framework, not the scale.
- Not claiming novelty in the fine-tuning itself. Fine-tuning BERT for classification is well-trodden. The novelty is in the learning curve + cost crossover + interactive tool combination.

---

## 6. Technical Challenges Worth Documenting

### 6a. Stratified Subsampling for Learning Curves

Creating training subsets of size 10, 100, 1K, 10K, 50K, 100K while maintaining class distribution is non-trivial:
- At N=10 with 18 classes, some classes have 0 or 1 examples — must handle gracefully
- At N=100, rare classes may have <3 examples — stratification with `sklearn.model_selection.train_test_split` may fail
- Solution: stratified sampling where possible, minimum 1 example per class, document the class coverage at each N
- The degradation of rare-class performance at small N is itself an interesting finding

**Blog-worthy insight:** "What happens to classification when you have 0 training examples for a class — and why the answer differs between fine-tuning and zero-shot approaches."

### 6b. QLoRA on a T4: Memory Management

Fine-tuning Mistral-7B (or similar) with QLoRA on a 16GB T4 GPU requires:
- 4-bit quantisation via bitsandbytes (model loads at ~4GB instead of ~14GB)
- LoRA rank selection (r=8 or r=16 — lower rank = fewer trainable parameters but potentially lower quality)
- Gradient checkpointing to trade compute for memory
- Sequence length management (CFPB complaints can be long; may need truncation at 512 or 1024 tokens)
- Batch size of 1–2 with gradient accumulation to simulate larger effective batches

**Blog-worthy insight:** "Fine-tuning a 7B model on a free GPU: the actual memory budget and what you sacrifice."

### 6c. Fair Comparison Between Fine-Tuned and Zero-Shot

Comparing a fine-tuned model against zero-shot GPT-4 requires care to avoid stacking the deck:
- **Same test set:** All models evaluated on the same held-out test set (never seen during training or few-shot selection)
- **Prompt engineering:** Zero-shot and few-shot prompts should be reasonably optimised (not strawman prompts that make the LLM look bad)
- **Few-shot example selection:** For k-shot prompting, examples should be representative, not adversarial — use stratified random selection from the training set
- **Label format:** LLM outputs must be parsed into the same label space — need a mapping from free-text LLM responses to categorical labels
- **Multiple evaluation runs:** Fine-tuned models vary by seed; LLM outputs can vary by temperature. Report mean ± std.

**Blog-worthy insight:** "The 5 ways people accidentally make fine-tuned models look better (or worse) than zero-shot LLMs."

### 6d. Cost Measurement Methodology

Accurate cost comparison requires:
- **Training cost:** Total GPU-hours measured (not estimated) via `torch.cuda.Event` and wall-clock time. Convert to dollars using Google Colab T4 pricing ($0.00/hr free tier, but report equivalent spot pricing ~$0.35/hr for A100 comparisons).
- **Inference cost for local models:** Measure time per classification on standardised hardware. Convert to cost assuming a cloud GPU instance.
- **API cost for LLMs:** Record exact token counts (input + output) per classification. Multiply by current API pricing (GPT-4: $30/$60 per 1M input/output tokens; Claude 3.5 Sonnet: $3/$15).
- **Amortisation period:** Training is a one-time cost. Assume 3-month amortisation (model retrained quarterly — realistic for production).
- **Document all assumptions:** Pricing changes. State the date prices were captured and provide the formula so readers can update.

### 6e. Handling LLM Output Parsing

GPT-4/Claude don't always return clean categorical labels. Responses may include:
- Extra explanation: "Based on the complaint, this appears to be about **Credit Reporting**..."
- Slight label variations: "Credit Report" vs. "Credit reporting" vs. "Credit Reporting/Repair"
- Refusals or hedging: "I'm not sure, but it could be..."
- Multiple labels: "This could be Debt Collection or Credit Reporting"

**Approach:** Define a strict parsing pipeline with fuzzy matching to canonical labels, a "parse failure" category, and document the parse success rate as part of the results.

---

## 7. Evaluation Framework

### Layer 1: Classification Accuracy (Per Model × Training Size)

**Metrics:**
- Macro-F1 (treats all classes equally — important because CFPB classes are imbalanced)
- Weighted-F1 (accounts for class frequency — shows real-world performance)
- Per-class precision, recall, F1 (reveals which classes are easy/hard for each approach)

**Presentation:** Learning curve plot — X axis: log-scale training set size, Y axis: macro-F1. One line per model. Horizontal lines for zero-shot/few-shot baselines.

### Layer 2: Statistical Significance

At each training set size, test whether the fine-tuned model's accuracy is significantly different from the zero-shot baseline:
- 3 seeds per configuration → paired comparison
- McNemar's test on per-example predictions (more powerful than comparing aggregate metrics)
- Report p-values and confidence intervals, not just point estimates

### Layer 3: Cost Analysis

For each model, produce:
- Training cost (measured GPU-hours × price)
- Inference cost per classification (measured time × price)
- Total monthly cost at volumes: 1K, 10K, 100K, 1M
- Crossover point: monthly volume at which fine-tuning becomes cheaper

**Presentation:** Cost crossover chart (monthly cost vs. volume, one line per model).

### Layer 4: Latency Analysis

For each model, measure:
- Median, P95, P99 inference latency on standardised hardware
- Throughput (classifications/second)
- Batch vs. single-sample latency

**Presentation:** Latency distribution plots (histogram + CDF) per model.

### Layer 5: Failure Mode Analysis

**Where does each approach fail?**
- **Fine-tuned models at small N:** Which classes drop to 0% recall? How does class coverage degrade?
- **Zero-shot LLMs:** Which classes do they systematically confuse? (Hypothesis: categories requiring domain-specific knowledge, e.g., distinguishing "Credit reporting" from "Credit card")
- **Parse failures:** What percentage of LLM responses can't be mapped to a valid label?
- **Confidence calibration:** Do fine-tuned models output well-calibrated probabilities? Do LLM responses with higher stated confidence actually have higher accuracy?
- **Text length effects:** Does performance differ for short complaints vs. long complaints?

**Presentation:** Error taxonomy table with examples. This is the section that demonstrates analytical maturity.

### Layer 6: Decision Framework Validation

Test the interactive recommendation tool against held-out scenarios:
- If a user inputs "50K classifications/month, 500 labelled examples, <200ms latency requirement" — does the tool's recommendation match what the experimental results show?
- Edge cases: what happens at the crossover boundaries?

---

## 8. Deployment Plan

### Streamlit Dashboard (Primary)

An interactive dashboard with three tabs:

**Tab 1: Learning Curves**
- Interactive plot (Plotly/Altair) with toggleable model lines
- Hover for exact values
- Shaded confidence bands (±1 std across seeds)
- Horizontal reference lines for zero-shot/few-shot baselines
- Dropdown to switch between macro-F1, weighted-F1, per-class views

**Tab 2: Cost Calculator**
- Sliders: monthly classification volume, available labelled data, max latency
- Input: current API provider and tier (GPT-4, Claude, etc.)
- Output: recommended approach, estimated monthly cost, accuracy estimate
- Cost crossover chart that updates dynamically with slider changes
- Table: side-by-side comparison of all approaches for the selected parameters

**Tab 3: Deep Dive**
- Per-class performance heatmap (model × class × metric)
- Side-by-side confusion matrices
- Latency distribution plots
- Failure examples (anonymised complaint text → predicted vs. actual label)

### FastAPI Endpoint (Secondary)

```
POST /recommend
{
  "monthly_volume": 50000,
  "labelled_examples": 500,
  "max_latency_ms": 200,
  "accuracy_priority": "high"  // or "balanced" or "cost_priority"
}

→ 200 OK
{
  "recommendation": "fine_tune_deberta",
  "estimated_monthly_cost_usd": 12.50,
  "estimated_accuracy_macro_f1": 0.87,
  "estimated_latency_p95_ms": 45,
  "alternative": "gpt4_zero_shot",
  "alternative_cost_usd": 1500.00,
  "alternative_accuracy": 0.84,
  "reasoning": "At 50K/month with 500 labelled examples, DeBERTa fine-tuning achieves higher accuracy than GPT-4 zero-shot at 99% lower inference cost."
}
```

### Docker + Cloud

- `Dockerfile` using `python:3.11-slim` + CUDA for inference
- Pre-trained model weights baked into image (or pulled from HuggingFace Hub)
- Deploy dashboard to Render / HuggingFace Spaces (free tier)
- All experimental results stored as JSON artifacts in the repo (no need to re-run training to use the dashboard)

---

## 9. Project Structure

```
fine-tune-vs-prompt-benchmark/
├── README.md                          # Business-first, < 500 words
├── Dockerfile
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/                           # CFPB download (gitignored)
│   ├── processed/                     # Cleaned + split data
│   ├── subsets/                       # Stratified subsamples at each N
│   └── sample/                        # Small sample for reviewers
│
├── src/
│   ├── data/
│   │   ├── download.py                # CFPB data download + extraction
│   │   ├── preprocess.py              # Cleaning, label mapping, splits
│   │   └── subsample.py               # Stratified subsampling at 6 sizes
│   ├── training/
│   │   ├── bert_trainer.py            # DistilBERT/DeBERTa fine-tuning (HF Trainer)
│   │   ├── bert_custom_loop.py        # Custom PyTorch training loop variant
│   │   ├── qlora_trainer.py           # Mistral-7B QLoRA fine-tuning (PEFT)
│   │   └── run_all_experiments.py     # Orchestrate: 3 models × 6 sizes × 3 seeds
│   ├── inference/
│   │   ├── local_inference.py         # Batch inference with fine-tuned models
│   │   ├── api_inference.py           # GPT-4 / Claude API calls
│   │   ├── response_parser.py         # Parse LLM responses to canonical labels
│   │   └── latency_benchmark.py       # Standardised latency measurement
│   ├── evaluation/
│   │   ├── metrics.py                 # Classification metrics computation
│   │   ├── cost_model.py              # Cost calculation engine
│   │   ├── learning_curves.py         # Generate learning curve data
│   │   ├── crossover_analysis.py      # Cost crossover computation
│   │   └── failure_analysis.py        # Error categorisation
│   └── api.py                         # FastAPI /recommend endpoint
│
├── app/
│   └── streamlit_app.py               # Interactive dashboard
│
├── results/
│   ├── experiments/                   # Raw results per experiment (JSON)
│   ├── aggregated/                    # Aggregated metrics, learning curves
│   ├── figures/                       # Static plots for README/blog
│   └── cost_model.json               # API pricing snapshot + assumptions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # CFPB data statistics, class distribution
│   ├── 02_training_experiments.ipynb  # Training runs (Colab-compatible)
│   ├── 03_api_evaluation.ipynb        # Zero-shot/few-shot LLM evaluation
│   ├── 04_results_analysis.ipynb      # Learning curves, cost analysis, failures
│   └── 05_qlora_deep_dive.ipynb       # QLoRA memory profiling, ablations
│
├── tests/
│   ├── test_preprocess.py
│   ├── test_subsample.py
│   ├── test_response_parser.py
│   ├── test_cost_model.py
│   └── test_metrics.py
│
└── docs/
    ├── architecture.md                # Pipeline diagram + design decisions
    ├── experiment_log.md              # All 54+ runs with hyperparameters
    ├── cost_assumptions.md            # API pricing, GPU pricing, amortisation
    └── evaluation_results.md          # Full results + failure analysis
```

---

## 10. Communication Plan

### README Structure (< 500 words)

1. **One-liner:** "When should you fine-tune a model vs. call an LLM API for text classification? This project answers with data: learning curves, cost crossover analysis, and an interactive decision tool."
2. **The problem:** Every ML team deploying NLP faces the build-vs-buy decision. Getting it wrong costs either engineering time (unnecessary fine-tuning) or money (unnecessary API spend).
3. **What this builds:** Fine-tuned DistilBERT, DeBERTa, and Mistral-7B (QLoRA) on CFPB consumer complaints at 6 training set sizes, benchmarked against GPT-4/Claude zero-shot. An interactive Streamlit dashboard that recommends the optimal approach given your volume, labelled data, and latency requirements.
4. **Key results:** (Filled after experiments.) "Fine-tuned DeBERTa matches GPT-4 zero-shot accuracy at [N] training examples. At [V] classifications/month, fine-tuning is [X]% cheaper. Fine-tuned models are [Y]x faster at inference."
5. **Try it:** Link to deployed Streamlit dashboard.
6. **Tech stack:** PyTorch, HuggingFace Transformers, PEFT/QLoRA, bitsandbytes, OpenAI/Anthropic APIs, Streamlit, FastAPI, Docker.

### Blog Post Outline

**Title:** "I Ran 54 Fine-Tuning Experiments to Answer the Most Common ML Engineering Question"

1. **Hook:** "Every ML team I've talked to asks the same thing: should we fine-tune or just use GPT-4? I ran the experiments to find out — and built a tool that answers it for your specific situation."
2. **The experiment design:** Why learning curves matter more than single-point benchmarks. How I structured 3 architectures × 6 training sizes × 3 seeds.
3. **Findings that surprised me:** (Expected: fine-tuning wins at scale. What was the crossover point? Were there cases where zero-shot won permanently? Did the 7B QLoRA model behave differently from BERT-class?)
4. **The cost math:** Real numbers from real experiments. When does the $8 training run pay for itself?
5. **Where zero-shot still wins:** The cases where fine-tuning doesn't make sense — low volume, rapidly changing labels, cold start with no labelled data.
6. **The interactive tool:** How I built the decision framework and what it recommends.
7. **Limitations and what I'd do differently:** Fixed API pricing, limited to 3 architectures, single dataset. What a production version would need.

### Interview Talking Points (STAR Format)

**S:** ML teams deploying NLP classification face a recurring build-vs-buy decision: fine-tune a small model or use LLM API calls. The answer depends on labelled data availability, volume, latency requirements, and budget — but nobody had published an interactive decision framework with real experimental data behind it.

**T:** Build a benchmarking suite that produces learning curves (accuracy vs. training set size), cost crossover analysis, and latency comparisons across fine-tuned and zero-shot approaches, then package it as a reusable decision tool.

**A:** Fine-tuned DistilBERT, DeBERTa-v3, and Mistral-7B (QLoRA) on CFPB consumer complaints at 6 training set sizes (10 to 100K), with 3 seeds each — 54 training runs. Benchmarked against GPT-4 and Claude zero-shot/few-shot on the same held-out test set. Built a cost model incorporating measured GPU training time, measured inference latency, and current API pricing. Deployed as an interactive Streamlit dashboard where users input their parameters and get a recommendation.

**R:** [Key finding about crossover point]. [Key finding about cost savings]. [Key finding about failure modes — e.g., which classes zero-shot LLMs systematically fail on]. The interactive dashboard was deployed at [URL] and the methodology was documented in a blog post.

---

## 11. Scope and Timeline

### MVP (Week 1): Data + First Fine-Tuning Runs

- [ ] Download and preprocess CFPB data
- [ ] Implement stratified subsampling at 6 sizes
- [ ] DistilBERT fine-tuning at all 6 sizes (1 seed) — validate pipeline works
- [ ] Basic evaluation metrics
- [ ] Test GPU memory on Colab T4

### Iteration 1 (Week 2): Full Experiment Suite

- [ ] All DistilBERT runs (3 seeds × 6 sizes = 18 runs)
- [ ] All DeBERTa runs (18 runs)
- [ ] QLoRA Mistral-7B runs (18 runs — may need Kaggle P100 for larger sizes)
- [ ] GPT-4 + Claude zero-shot evaluation on test set
- [ ] GPT-4 few-shot evaluation (k=5)
- [ ] Response parsing pipeline for LLM outputs

### Iteration 2 (Week 3): Analysis + Cost Framework

- [ ] Learning curve generation and visualisation
- [ ] Latency benchmarking (all models, standardised hardware)
- [ ] Cost model implementation (training cost, inference cost, crossover)
- [ ] Failure mode analysis
- [ ] Per-class performance deep dive

### Iteration 3 (Week 4): Dashboard + Deployment

- [ ] Streamlit dashboard (3 tabs)
- [ ] FastAPI /recommend endpoint
- [ ] Dockerise
- [ ] Deploy to Render / HuggingFace Spaces
- [ ] End-to-end testing of interactive tool

### Communication (Week 5): Write-Up

- [ ] Write README
- [ ] Write blog post
- [ ] Record 2-min demo video
- [ ] Prepare interview talking points
- [ ] Final code cleanup: docstrings, type hints, linting, tests

### Total: 4–5 weeks

---

## 12. Estimated Costs

| Item | Cost |
|---|---|
| CFPB data | Free (public data) |
| Google Colab GPU (T4, free tier) | $0 (15–30 hrs/week) |
| Kaggle GPU (P100/T4, 30 hrs/week) | $0 |
| OpenAI API (GPT-4 zero-shot + few-shot on ~10K test samples) | ~$15–30 |
| Anthropic API (Claude evaluation) | ~$5–10 |
| Render deployment | Free tier |
| **Total** | **~$20–40** |

---

## 13. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| QLoRA 7B doesn't fit on T4 with CFPB text lengths | Medium | Medium | Reduce sequence length to 512, use gradient checkpointing, reduce LoRA rank. Fall back to Phi-2 (2.7B) if 7B is unworkable. |
| LLM API costs exceed budget | Low | Medium | Evaluate on a 5K subsample instead of full test set. Use Claude Haiku (cheaper) as additional baseline. |
| Learning curves show no interesting crossover | Low | High | Even a "boring" result (fine-tuning always wins at all sizes, or never wins) is informative. Frame as: "here's the empirical answer to a question people debate without data." |
| CFPB saturation makes reviewers dismiss the project | Medium | High | **Lead with the decision framework, not the dataset.** README opens with the business question, not "I used CFPB data." Blog title references the framework, not the classification task. |
| Google Colab session limits block long training runs | Medium | Medium | Use Kaggle as backup. Checkpoint aggressively. Smaller models (DistilBERT) train in <1hr per run. Only the 7B QLoRA runs are time-intensive. |
| LLM response parsing fails at high rate | Medium | Medium | Iterative prompt refinement. Constrain output format. Document parse failure rate as a finding (it's a real production issue). |
| Results look like "obvious" conclusions | Medium | Medium | Focus the blog on surprising findings, edge cases, and failure modes. The interactive tool is the differentiator, not the conclusion that "fine-tuning is cheaper at scale." |

---

## 14. References and Prior Art to Cite

- **LoRA Land** (Predibase, arXiv 2405.00732, May 2024): 310 fine-tuned LoRA models across 31 tasks vs. GPT-4. The industrial-scale version of P1's comparison. Our contribution: learning curves (accuracy vs. N) and interactive decision tool, which LoRA Land does not provide.
- **"Fine-Tuned 'Small' LLMs (Still) Significantly Outperform..."** (arXiv 2406.08660, Oct 2025): Academic paper with learning curve analysis for RoBERTa/DeBERTa/ELECTRA vs. GPT-4/Claude. Our contribution: cost crossover analysis and interactive tool, which the paper does not provide. We cite and build on their finding.
- **Stanford RCPedia** (Nov 2025): Fine-tuned Qwen3-8B vs. GPT-4.1 mini with cost analysis. Blog post format, single dataset. We provide multi-architecture comparison with learning curves.
- **"Think Before You Classify"** (MDPI Electronics, Mar 2025): 14 LLMs zero-shot on CFPB complaints. Validates our dataset choice. We add the fine-tuning side of the comparison.
- **vilcek/fine-tuning-BERT-for-text-classification** (GitHub): DistilBERT on CFPB. Tutorial-style. We add LLM comparison, learning curves, cost analysis.
- **"Selecting Between BERT and GPT for Text Classification in Political Science Research"** (arXiv 2411.05050, Nov 2024): BERT fine-tuning vs. GPT prompting with learning curves on political text. Different domain, similar methodology. Worth citing.
- **Bioformer** (PMC, Feb 2023): Compact biomedical BERT with speed/accuracy tradeoffs. Relevant if using PubMed data.

---

## 15. Checklist (From Portfolio Principles Reference)

### Business Framing
- [x] Clear business question in first sentence
- [x] Identified stakeholder (ML teams deploying NLP in production)
- [x] Quantified cost of getting it wrong (API overspend vs. wasted engineering time)
- [x] Specific, actionable output (interactive decision tool with recommendation)

### Technical Execution
- [x] Real-world data (CFPB Consumer Complaints — messy, imbalanced, real consumer language)
- [ ] Multiple data sources joined together — **WEAK:** single source. Could strengthen by joining with product metadata.
- [x] Substantial data engineering (stratified subsampling, LLM response parsing, cost model)
- [x] Appropriate technique choice (justified: BERT-class for efficiency, 7B for LoRA demonstration, LLM APIs for zero-shot baseline)
- [x] Multi-layered evaluation framework (6 layers)
- [x] Honest limitation documentation (prior art acknowledged extensively)

### Modern Stack
- [x] Evaluation framework for model outputs (learning curves, statistical significance, failure modes)
- [x] Cost and latency analysis (the core contribution)
- [x] Failure mode documentation (per-class failures, parse failures, calibration)
- [x] Comparison against sensible baselines (zero-shot, few-shot, multiple fine-tuned architectures)

### Deployment & Engineering
- [x] Interactive demo (Streamlit dashboard with 3 tabs)
- [x] API endpoint (FastAPI /recommend)
- [x] Docker + cloud deployment (Render)
- [x] Professional code structure (src/, tests/, docs/)
- [x] Version control with meaningful commits planned

### Communication
- [x] README plan (< 500 words, business-first)
- [x] Blog post outline (focused on framework, not fine-tuning)
- [x] Resume line drafted
- [x] Interview talking points (STAR format)
- [ ] Architecture diagram — **TODO:** create visual pipeline diagram

### Novelty
- [x] Searched GitHub, Kaggle, Medium, arXiv — **extensively documented in Section 3**
- [x] Framing distinct from existing work (decision tool, not benchmark)
- [ ] Combination of data sources/techniques is original — **WEAK:** individual pieces exist; combination is the novelty
- [x] Prior art acknowledged and differentiation stated
- [⚠️] **Would not be confused with a tutorial follow-along** — this is the biggest risk. The CFPB + BERT combination specifically appears in tutorials. The framing must carry the project.

---

## 16. Honest Assessment

### Strengths
- The business question ("build or buy?") resonates with every ML hiring manager
- The interactive decision tool is a genuinely useful deliverable, not just a benchmark
- Deep PyTorch skills demonstrated (custom training loops, QLoRA, GPU memory management)
- Strong interview talking points about production trade-offs
- Cost analysis demonstrates the kind of thinking engineering managers actually need

### Weaknesses
- **Saturation is the primary risk.** LoRA Land, the arXiv learning curves paper, Stanford RCPedia, and CFPB + BERT tutorials collectively cover most of P1's territory. The novel contribution (interactive decision tool with combined learning curves + cost crossover) is real but narrow.
- **Business impact is indirect.** The output is a decision framework, not "this saves Company X €Y." Less visceral than L1's regulatory compliance or the Private Label project's market opportunity identification.
- **Single dataset.** A production version of this tool would test across multiple datasets/domains. With one dataset, the results may not generalise. (Mitigated by choosing a dataset with realistic properties.)
- **Overlap with B1.** If B1 uses FinBERT, there's moderate overlap in "PyTorch + HuggingFace + transformer" skills, though P1 goes deeper into fine-tuning mechanics.

### Verdict

P1 is a **viable but not outstanding** portfolio project. It fills the PyTorch gap competently, demonstrates production-relevant thinking, and produces a useful deliverable. But it requires careful framing to overcome the saturation risk — the README, blog, and interview narrative must lead with the decision framework and relegate "I fine-tuned BERT on CFPB" to implementation details. If a stronger PyTorch project (e.g., P2: Custom Embeddings) emerges from validation, P1 could be deprioritised.

---

*This document is the complete plan for P1. It should be compared against P2 validation results before committing to implementation.*
