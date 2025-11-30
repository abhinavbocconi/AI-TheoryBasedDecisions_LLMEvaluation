# Chat Analysis Pipeline

This repository contains scripts for processing, merging, and analyzing chat data from an AI-assisted decision-making experiment.

---

## ⚠️ Data Availability Notice

**The following data files are excluded from this repository (via `.gitignore`) to protect participant privacy:**

| File | Description | Availability |
|------|-------------|--------------|
| `gpt_threads.xlsx` | Raw OpenAI thread exports | Available on request* |
| `gpt_threads_clean.xlsx` | Cleaned thread data | Available on request* |
| `chats.xlsx` | All extracted messages | Available on request |
| `user_messages.xlsx` | Filtered user messages | Available on request |
| `user_messages_fulldata.xlsx` | Messages with participant data | Available on request |
| `user_messages_fulldata_classified.xlsx` | Final classified dataset | Available on request |
| `FullData/finalData_SS_981 (1).xlsx` | Participant survey data | Available on request |
| `FullData/finalData_SS_981_withThreadID.xlsx` | Survey data with thread IDs | Available on request |
| `FullData/threads_cleaned.xlsx` | Thread-participant mapping | Available on request |

*\*Raw thread files (`gpt_threads.xlsx`, `gpt_threads_clean.xlsx`) will have system prompts redacted before sharing to protect proprietary AI system design.*

**To request data access, please contact the authors.**

---

## Data Pipeline Overview

```
Raw Data                    Cleaned Data                 Analysis-Ready Data
─────────────────────────────────────────────────────────────────────────────

gpt_threads.xlsx     ──►    gpt_threads_clean.xlsx    (Step 1: cleanup.py)
                                  │
                                  ▼
                             chats.xlsx               (Step 2: json_extract.py)
                                  │
                                  ▼
                          user_messages.xlsx          (Step 3: user_message_extract.py)
                                  │
                                  ▼
FullData/                  user_messages_fulldata.xlsx (Step 4: merge_messages_fulldata.py)
├── finalData_SS_981.xlsx         │
└── threads_cleaned.xlsx          ▼
                          user_messages_fulldata_classified.xlsx  (Step 5: classify_messages.py)
                                 │
                                 ▼
                          user_messages_fulldata_classified.xlsx  (Step 6: combine_classifications.py)
                          (with 20-vote combined classification)
```

---

## Step-by-Step Replication Guide

### Prerequisites

```bash
pip install pandas openpyxl openai anthropic
```

### Step 1: Clean Raw Thread Data

**Script:** `cleanup.py`

**Input:** `gpt_threads.xlsx` (raw OpenAI thread exports with JSON blobs)

**Output:** `gpt_threads_clean.xlsx`

**What it does:**
- Removes rows with NULL `thread_json`
- Keeps only `thread_id` and `thread_json` columns
- Validates JSON format

```bash
python cleanup.py
```

---

### Step 2: Extract Messages from JSON

**Script:** `json_extract.py`

**Input:** `gpt_threads_clean.xlsx`

**Output:** `chats.xlsx`

**What it does:**
- Parses nested JSON from `thread_json` column
- Extracts individual messages (user and assistant)
- Maps assistant IDs to names (normalGPT, coordinator, creator, coach)
- Creates message-level dataset

```bash
python json_extract.py
```

---

### Step 3: Extract Real User Messages

**Script:** `user_message_extract.py`

**Input:** `chats.xlsx`

**Output:** `user_messages.xlsx`

**What it does:**
- Filters to user messages only
- Removes system-generated messages (`_input_1`, etc.)
- Removes duplicate messages (system copies sent to coach)
- Strips routing suffixes from message content
- Derives experimental condition (General AI vs Agentic AI) from assistant types
- Adds word count

```bash
python user_message_extract.py
```

---

### Step 4: Merge Thread IDs to Full Dataset

**Script:** `FullData/full_data_merge_threadID.py`

**Input:** 
- `FullData/finalData_SS_981 (1).xlsx` (participant survey data)
- `FullData/threads_cleaned.xlsx` (thread-participant mapping)

**Output:** `FullData/finalData_SS_981_withThreadID.xlsx`

**What it does:**
- Merges thread_id and prompt_count to participant data
- Only merges for Post=1 rows (when AI was used)
- Handles participants with multiple threads

```bash
cd FullData
python full_data_merge_threadID.py
```

---

### Step 5: Merge Full Data to Messages

**Script:** `merge_messages_fulldata.py`

**Input:**
- `user_messages.xlsx`
- `FullData/finalData_SS_981_withThreadID.xlsx`

**Output:** `user_messages_fulldata.xlsx`

**What it does:**
- Links messages to participants via thread_id
- Adds all 79 participant-level variables to each message
- **Filters out messages from participants not in the experimental sample**
- Result: 1,273 messages from 603 participants

```bash
python merge_messages_fulldata.py
```

---

### Step 6: Classify Messages with LLMs

**Script:** `classify_messages.py`

**Input:** `user_messages_fulldata.xlsx`

**Output:** `user_messages_fulldata_classified.xlsx`

**What it does:**
- Classifies each message into 7 categories using GPT-5.1 and Claude Sonnet 4.5
- Runs 10 classification repeats per message per model
- Takes mode (most common) classification for robustness
- Calculates confidence scores and model agreement

**Categories:**
1. Task Delegation
2. Refinement Request
3. Evaluation Seeking
4. Information Seeking
5. Clarification
6. Acknowledgment
7. Other

**Setup:**
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
python classify_messages.py
```

---

### Step 7: Combine Classifications (20-Vote Majority)

**Script:** `combine_classifications.py`

**Input:** `user_messages_fulldata_classified.xlsx`

**Output:** `user_messages_fulldata_classified.xlsx` (updated with combined columns)

**What it does:**
- Combines all 20 votes (10 GPT + 10 Claude) for each message
- Takes the mode (most common) classification across all 20 votes
- Provides a single robust classification that leverages both models
- Calculates combined confidence score

**Why this approach:**
- GPT and Claude agree on ~63% of classifications
- Main disagreement: GPT uses "Other" more often, Claude is more specific
- The 20-vote majority resolves disagreements by true consensus
- If GPT votes 6x "Other" but Claude votes 8x "Task Delegation", 
  the combined result is "Task Delegation" (8 > 6)

```bash
python combine_classifications.py
```

---

### Step 8: Analyze Chat Behavior Patterns

**Script:** `analyze_chat_behavior.py`

**Input:** `user_messages_fulldata_classified.xlsx`

**Output:** Updated Excel with `message_order` and `round_group` columns

**What it does:**
- Derives message order within each thread (1st, 2nd, 3rd... query)
- Creates round groups (1st Query, 2nd Query, 3rd+ Query)
- Analyzes query count distribution
- Computes classification distribution by round
- Shows autopilot→copilot transition pattern

```bash
python analyze_chat_behavior.py
```

---

### Step 9: Subsample × Condition Analysis

**Script:** `subsample_condition_analysis.py`

**Input:** `user_messages_fulldata_classified.xlsx`

**Output:** Console statistics for paper

**What it does:**
- Analyzes PhD vs Non-PhD behavior by condition
- Analyzes Experienced vs Less Experienced by condition
- Computes single-query (autopilot) rates
- Computes evaluation seeking rates
- Identifies User-System-Problem Fit patterns

**Key findings:**
- PhD + Agentic AI: +29pp single-query rate (autopilot)
- Experienced + Agentic AI: +9pp evaluation seeking (validation)

```bash
python subsample_condition_analysis.py
```

---

### Step 10: Generate Analysis Figures

**Script:** `generate_figures.py`

**Input:** `user_messages_fulldata_classified.xlsx`

**Output:** 
- `figure_intent_by_round.png` (Figure 1: Intent by query round)
- `figure_subsample_condition.png` (Figure 2: Subsample × condition)

**What it does:**
- Generates publication-ready figures with Bocconi colors
- Computes autopilot (Task Delegation) and copilot (Eval + Refine) rates
- Figure 1: Shows intent evolution across query rounds
- Figure 2: Shows autopilot/copilot behavior by PhD/Experience × Condition

**Metrics:**
- Autopilot = Task Delegation rate (intent-based)
- Copilot = Evaluation Seeking + Refinement Request rate (intent-based)

```bash
python generate_figures.py
```

---

### Step 11: Single-Query User Analysis

**Script:** `single_query_analysis.py`

**Input:** `user_messages_fulldata_classified.xlsx`

**Output:** Console statistics for paper

**What it does:**
- Analyzes intent distribution of single-query (autopilot) users
- Compares with multi-query users
- Shows single-query users had +11pp task delegation, −7pp evaluation seeking

```bash
python single_query_analysis.py
```

---

## Output Variables

### Message Classification Columns (added by Steps 6-7)

| Column | Description |
|--------|-------------|
| `category_gpt` | GPT's mode classification (10 repeats) |
| `gpt_mode_count` | Times mode appeared (out of 10) |
| `gpt_confidence` | Confidence ratio (0.0-1.0) |
| `gpt_high_confidence` | True if mode ≥ 5/10 |
| `category_claude` | Claude's mode classification (10 repeats) |
| `claude_mode_count` | Times mode appeared (out of 10) |
| `claude_confidence` | Confidence ratio (0.0-1.0) |
| `claude_high_confidence` | True if mode ≥ 5/10 |
| `models_agree` | True if GPT and Claude agree |
| **`category_combined`** | **Final classification (20-vote majority)** |
| `combined_mode_count` | Times mode appeared (out of 20) |
| `combined_confidence` | Confidence ratio (0.0-1.0) |

---

## Key Variables in Final Dataset

### Experimental Conditions
- `aristotle`: 0 = Human Only, 1 = General AI, 2 = Agentic AI
- `Post`: 0 = Pre-intervention, 1 = Post-intervention

### Message Content
- `content_clean`: Cleaned message text
- `word_count`: Number of words in message

### Participant Demographics
- `age`, `gender`, `educationLevel`, `educationField`
- `industry`, `jobFunction`, `experience`

### Psychological Measures
- `algoAversion`, `algoLove`: Algorithmic attitudes
- `GPTFamiliarity`, `promptSkill`: AI experience
- `automationBias`, `confidence`: Cognitive biases

### Outcomes
- `value`, `sd`: Theory quality scores
- `value_diff`, `sd_diff`: Pre-post changes

---

## File Structure

```
ChatAnalysis/
├── README.md                              # This file
├── participant_log_coverage.md            # Coverage analysis
├── .gitignore                             # Excludes data files
│
├── # Step 1-3: Data extraction
├── cleanup.py
├── json_extract.py
├── user_message_extract.py
│
├── # Step 4-5: Data merging
├── merge_messages_fulldata.py
│
├── # Step 6-7: Classification
├── classify_messages.py
├── combine_classifications.py
│
├── # Step 8-11: Analysis
├── analyze_chat_behavior.py               # Query patterns & round analysis
├── subsample_condition_analysis.py        # PhD/Experience × Condition
├── single_query_analysis.py               # Autopilot user analysis
├── generate_figures.py                    # Generates figures
│
├── # Figures for paper
├── figure_intent_by_round.png             # Figure 1
├── figure_subsample_condition.png         # Figure 2
│
└── FullData/
    └── full_data_merge_threadID.py
```

**Note:** Data files (`*.xlsx`) are excluded via `.gitignore`. See [Data Availability Notice](#️-data-availability-notice) above.

---

## Key Findings Summary

### Overall Engagement Patterns
- **1,273 messages** from **603 participants** across **609 threads**
- Mean queries per user: **2.1** (range: 1–10)
- **39% single-query users**

### Autopilot → Copilot Transition (Intent-Based)
| Round | Autopilot (Task Del.) | Copilot (Eval + Refine) |
|-------|----------------------|------------------------|
| 1st Query | 32% | 27% |
| 2nd Query | 23% | 33% |
| 3rd+ Query | 19% | 37% |

### Single-Query vs Multi-Query Users
| Metric | Single-Query | Multi-Query (1st) | Diff |
|--------|--------------|-------------------|------|
| Autopilot (Task Del.) | 39% | 27% | +12pp |
| Copilot (Eval + Refine) | 22% | 30% | −8pp |

### User-System-Problem Fit (Condition × Subsample)
| Subsample | Metric | General AI | Agentic AI | Diff |
|-----------|--------|------------|------------|------|
| **PhD** | Autopilot (Task Del.) | 22% | 34% | +12pp |
| **PhD** | Copilot (Eval + Refine) | 41% | 36% | −5pp |
| **Experienced** | Copilot (Eval + Refine) | 26% | 36% | +9pp |
