# Participant Chat Log Coverage Analysis

## Overview

This document summarizes how many participants from the experimental conditions have available chat logs for analysis.

---

## Experimental Design Recap

| Condition | aristotle | Description | Total Participants |
|-----------|-----------|-------------|-------------------|
| Human Only | 0 | No AI assistance | 329 |
| General AI | 1 | GPT-4o (standard ChatGPT) | 325 |
| Agentic AI | 2 | Aristotle (multi-agent system) | 327 |

**Note:** Chat logs are only available for AI conditions (aristotle = 1 or 2). Human-only participants did not interact with any AI system.

---

## Chat Log Coverage by Condition

### General AI Condition (aristotle = 1)

| Metric | Count | Percentage |
|--------|-------|------------|
| Total participants | 325 | 100% |
| With chat logs | 309 | 95.1% |
| Without chat logs | 16 | 4.9% |

### Agentic AI Condition (aristotle = 2)

| Metric | Count | Percentage |
|--------|-------|------------|
| Total participants | 327 | 100% |
| With chat logs | 304 | 93.0% |
| Without chat logs | 23 | 7.0% |

---

## Message-Level Summary

After merging chat logs with the full experimental dataset and filtering to only participants in our sample:

| Condition | Participants | Total Messages | Avg Messages/User |
|-----------|--------------|----------------|-------------------|
| General AI | 303 | 642 | 2.1 |
| Agentic AI | 300 | 631 | 2.1 |
| **Total** | **603** | **1,273** | **2.1** |

---

## Notes

1. **Missing logs (39 participants):** Some participants in the AI conditions do not have associated chat logs. This could be due to:
   - Technical issues during data collection
   - Participants not engaging with the AI
   - Matching issues between survey and chat data

2. **Balanced conditions:** Both conditions have nearly identical message volumes and per-user message counts, enabling fair comparison.

3. **Analysis sample:** The final analysis dataset (`user_messages_fulldata.xlsx`) contains 1,273 messages from 603 participants.

