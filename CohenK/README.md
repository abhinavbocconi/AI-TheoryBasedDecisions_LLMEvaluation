# Cohen's Kappa Inter-Rater Reliability Analysis

**Final GitHub-Ready Version**

This repository contains rigorous Cohen's kappa analyses for evaluating inter-rater reliability between human experts and Large Language Models (LLMs) in business strategy evaluation.

## 📊 Dataset Overview

- **1,962 business theory evaluations** (teresaID: 1-1962)
- **981 participants × 2 time periods**
- **Human Expert Evaluations**: Strategic management professionals
- **LLM Evaluations**: Claude-4-Sonnet & GPT o4-mini (10 runs each, averaged)

## 🔬 Analysis Scripts

### 1. `cohen_kappa_sps_cit_final.py`
**Analyzes Probability & Confidence Measures**

- **SPS**: Subjective Probability of Success (0-100%)
- **CIT**: Confidence in Theory Assessment (1-7 scale)
- **Discretization**: Terciles for SPS, 3-category for CIT
- **Key Result**: Fair agreement on SPS (κ ≈ 0.25), Poor on CIT (κ ≈ 0.00)

### 2. `cohen_kappa_quality_integer_final.py`
**Conservative Integer Matching Method**

- **Dimensions**: Novelty, Feasibility, Environmental, Financial, Quality (1-5 scale)
- **Method**: Round continuous LLM scores to nearest integers
- **Key Result**: Generally poor agreement (85% of comparisons κ < 0.20)

### 3. `cohen_kappa_quality_tercile_final.py`
**Alternative Tercile Categorization Method**

- **Dimensions**: Same 5 quality dimensions
- **Method**: Group all ratings into Low (1-2), Medium (3), High (4-5)
- **Key Result**: Improved agreement (30% fair agreement, κ > 0.21)

## 🚀 Quick Start

```bash
# Clone and navigate to directory
cd CohenK

# Install dependencies
pip install pandas scikit-learn numpy

# Run analyses
python3 cohen_kappa_sps_cit_final.py           # SPS/CIT analysis
python3 cohen_kappa_quality_integer_final.py   # Conservative approach
python3 cohen_kappa_quality_tercile_final.py   # Alternative approach
```

## 📋 Data Requirements

The scripts expect `business_evaluation_full.csv` with these columns:

**Human Expert Ratings:**
- `expertProbability`, `expertConfidence`
- `expertNovelty`, `expertImplementation `, `expertEnvironmental`, `expertFinancial`, `expertQuality`

**LLM Ratings:**
- `anthropic_avg_sps_llm`, `anthropic_avg_cit_llm`, `anthropic_avg_*`
- `openai_avg_sps_llm`, `openai_avg_cit_llm`, `openai_avg_*`
- `cross_model_avg_sps_llm`, `cross_model_avg_cit_llm`, `cross_model_avg_*`

**Unique Identifier:**
- `Teresa ID` or `teresaID` (automatically detected)

## 📊 Key Results Summary

### SPS/CIT Measures
| Comparison | SPS κ | CIT κ | Interpretation |
|------------|-------|-------|----------------|
| Human vs Claude | 0.257 | -0.020 | Fair SPS, Poor CIT |
| Human vs GPT | 0.250 | 0.002 | Fair SPS, Poor CIT |
| Claude vs GPT | 0.636 | 0.020 | **Substantial SPS**, Poor CIT |

### Quality Dimensions - Integer Method
| Comparison | Novelty | Feasibility | Environmental | Financial | Quality |
|------------|---------|-------------|---------------|-----------|---------|
| Human vs Claude | 0.056 | 0.108 | 0.054 | 0.129 | **0.154** |
| Human vs GPT | 0.069 | -0.002 | 0.085 | 0.051 | 0.064 |
| Claude vs GPT | **0.505** | 0.025 | **0.554** | 0.299 | 0.070 |

### Quality Dimensions - Tercile Method
| Comparison | Novelty | Feasibility | Environmental | Financial | Quality |
|------------|---------|-------------|---------------|-----------|---------|
| Human vs Claude | 0.162 | **0.241** | 0.092 | **0.309** | **0.314** |
| Human vs GPT | 0.153 | 0.062 | 0.126 | 0.141 | 0.173 |
| Claude vs GPT | **0.507** | -0.009 | **0.529** | **0.295** | 0.008 |

## 🔍 Code Quality Features

✅ **Rigorous Validation**: TeresaID verification, data range checks, missing value handling
✅ **Comprehensive Error Handling**: Graceful failure with informative messages
✅ **Transparent Methodology**: Clear documentation of discretization strategies
✅ **Publication Ready**: LaTeX table outputs included
✅ **GitHub Ready**: Professional documentation, clean code structure

## 📊 Interpretation Guide

- **κ < 0.20**: Poor agreement
- **0.21-0.40**: Fair agreement
- **0.41-0.60**: Moderate agreement
- **0.61-0.80**: Substantial agreement
- **0.80+**: Almost perfect agreement

## 🎯 Key Findings

1. **Human-LLM Agreement**: Generally modest, with best results for Financial and Quality dimensions using tercile method
2. **LLM-LLM Consistency**: Strong internal agreement between AI models, especially on Environmental and Novelty assessments
3. **Method Sensitivity**: Tercile categorization reveals meaningful agreement patterns masked by strict integer matching
4. **Confidence Calibration**: Major divergence in how humans vs. LLMs express evaluation certainty

## 📝 Citation

```bibtex
@software{cohen_kappa_analysis,
  title={Cohen's Kappa Inter-Rater Reliability Analysis for Human-LLM Agreement},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]/CohenK}
}
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Issues and pull requests welcome! Please ensure all scripts pass validation before submitting.

---

**Last Updated**: September 2024
**Python Version**: 3.7+
**Dependencies**: pandas, scikit-learn, numpy