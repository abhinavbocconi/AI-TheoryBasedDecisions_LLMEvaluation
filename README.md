# RnR LLM Evaluation Pipeline

A comprehensive evaluation system comparing human expert assessments with Large Language Model (LLM) evaluations of business theories for startup success prediction.

## 📊 Overview

This research pipeline evaluates **1,962 business theories** from **981 participants** across **2 time periods**, comparing human expert judgments with AI model assessments on multiple dimensions.

### Key Features
- **Dual LLM Evaluation**: Claude-4-Sonnet & GPT-o4-mini
- **10-Run Reliability**: Each theory evaluated 10 times per model for statistical robustness
- **Multi-Dimensional Assessment**: 5 quality dimensions + probability/confidence measures
- **Parallel Processing**: High-performance async evaluation (30+ theories simultaneously)
- **Statistical Analysis**: Cohen's kappa inter-rater reliability analysis included

## 🏗️ System Architecture

```
Input Data (finalData_SS_981.csv)
    ↓
[ParallelAnalysis_performance.py] → 5 Quality Dimensions Evaluation
[ParallelAnalysis_sps_conf.py]    → SPS & Confidence Assessment
    ↓
Output CSVs with aggregated results (10 runs per model)
    ↓
[CohenK/ Analysis] → Inter-rater reliability statistics
```

## 📋 Requirements

### Python Dependencies
```bash
pip install anthropic openai pandas numpy asyncio aiohttp
```

### API Keys Required
- **Anthropic API Key** (Claude-4-Sonnet access)
- **OpenAI API Key** (GPT-o4-mini access)

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd RnRLLMEvaluation

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
# Set environment variables
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
export OPENAI_API_KEY="your-openai-api-key-here"

# Or create .env file (copy from .env.example)
echo "ANTHROPIC_API_KEY=your-anthropic-api-key-here" > .env
echo "OPENAI_API_KEY=your-openai-api-key-here" >> .env
```

### 3. Run Evaluations

#### Quality Dimensions Evaluation (5 metrics)
```bash
python3 ParallelAnalysis_performance.py
```
**Output**: `business_evaluation_PARALLEL_YYYYMMDD_HHMMSS.csv`

#### SPS & Confidence Evaluation
```bash
python3 ParallelAnalysis_sps_conf.py
```
**Output**: `sps_confidence_evaluation_PARALLEL_YYYYMMDD_HHMMSS.csv`

### 4. Statistical Analysis
```bash
cd CohenK/
python3 cohen_kappa_quality_integer_final.py
python3 cohen_kappa_quality_tercile_final.py
python3 cohen_kappa_sps_cit_final.py
```

## 📊 Evaluation Framework

### Performance Evaluation (5 Dimensions)
| Dimension | Scale | Description |
|-----------|-------|-------------|
| **Novelty** | 1-5 | How different from existing solutions? |
| **Feasibility & Scalability** | 1-5 | Likelihood to succeed and scale? |
| **Environmental Impact** | 1-5 | Planet-positive benefit level? |
| **Financial Impact** | 1-5 | Business value creation potential? |
| **Quality** | 1-5 | Overall solution quality? |

### SPS & Confidence Evaluation
| Metric | Scale | Description |
|--------|-------|-------------|
| **SPS** (Subjective Probability of Success) | 0-100% | Probability of 50% food waste reduction in 5 years |
| **CIT** (Confidence in Theory Assessment) | 1-7 | Confidence level in the assessment |

## 🔧 Configuration

### Performance Settings
```python
MAX_CONCURRENT_THEORIES = 30      # Theories processed in parallel
ANTHROPIC_SEMAPHORE_SIZE = 50     # Max concurrent Anthropic API calls
OPENAI_SEMAPHORE_SIZE = 100       # Max concurrent OpenAI API calls
```

### Testing Mode
To test with limited theories (recommended for initial runs):
```python
# In either script, uncomment this line:
df_filtered = df_filtered.head(30)  # Test with 30 theories
```

## 📁 File Structure

```
RnRLLMEvaluation/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
├── .env.example                                # Environment variables template
│
├── finalData_SS_981.csv                       # Input: 1,962 business theories
├── system_prompt_performance.txt              # LLM prompt for quality evaluation
├── system_prompt_sps_confidence.txt           # LLM prompt for SPS/confidence
│
├── ParallelAnalysis_performance.py             # Main: Quality dimensions evaluation
├── ParallelAnalysis_sps_conf.py               # Main: SPS & confidence evaluation
│
├── business_evaluation_PARALLEL_*.csv         # Output: Quality evaluation results
├── sps_confidence_evaluation_PARALLEL_*.csv   # Output: SPS/confidence results
│
└── CohenK/                                     # Statistical analysis folder
    ├── README.md                               # Cohen's kappa analysis documentation
    ├── business_evaluation_full.csv            # Combined dataset for analysis
    ├── cohen_kappa_quality_integer_final.py    # Conservative integer matching
    ├── cohen_kappa_quality_tercile_final.py    # Tercile categorization method
    └── cohen_kappa_sps_cit_final.py           # SPS/CIT reliability analysis
```

## ⚡ Performance

### Typical Performance Metrics
- **Processing Speed**: ~20-25 seconds per theory
- **Parallel Efficiency**: ~1200x faster than sequential
- **API Calls per Theory**: 20 total (2 models × 10 runs each)
- **Full Dataset Runtime**: ~11 hours for 1,962 theories

### Checkpoint System
- Automatic progress saving every completed theory
- Resume capability from interruptions
- Checkpoint files: `checkpoint_*_TIMESTAMP.pkl`

## 📈 Output Format

### Quality Evaluation Output
```csv
teresaID,theory,processing_timestamp,
anthropic_avg_novelty,anthropic_avg_feasibility_and_scalability,...,
openai_avg_novelty,openai_avg_feasibility_and_scalability,...,
cross_model_avg_novelty,cross_model_avg_feasibility_and_scalability,...
```

### SPS/Confidence Output
```csv
teresaID,theory,processing_timestamp,
anthropic_avg_sps_llm,anthropic_avg_cit_llm,anthropic_valid_runs,
openai_avg_sps_llm,openai_avg_cit_llm,openai_valid_runs,
cross_model_avg_sps_llm,cross_model_avg_cit_llm
```

## 🔬 Research Results Summary

### Key Findings from Cohen's Kappa Analysis
- **Human-LLM Agreement**: Generally modest (κ < 0.40)
- **LLM-LLM Consistency**: Strong internal agreement (κ > 0.50 on some dimensions)
- **Best Agreement**: Financial Impact and Quality dimensions (tercile method)
- **Confidence Calibration**: Major divergence between human vs. LLM confidence expression

### Statistical Significance
- **SPS Agreement**: Fair agreement (κ ≈ 0.25) between humans and LLMs
- **Quality Dimensions**: 30% show fair agreement using tercile categorization
- **Inter-LLM Reliability**: Substantial agreement on Environmental and Novelty assessments

## 🚨 Important Notes

### Security
- **Never commit API keys** to version control
- Use environment variables or `.env` files only
- Rotate API keys regularly

### Data Integrity
- **teresaID range**: 1-1962 (verified consistent across all files)
- **Missing data handling**: Automatic filtering of empty theory entries
- **Validation**: Built-in data range and format checks

### Cost Considerations
- **API Costs**: ~$50-100 for full dataset evaluation (varies by model pricing)
- **Rate Limits**: Built-in semaphore controls prevent API limit violations
- **Monitoring**: Progress tracking with estimated completion times

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Ensure all tests pass with small dataset first
4. Submit pull request with clear description

## 📄 License

MIT License - See LICENSE file for details

## 📚 Citation

```bibtex
@software{rnr_llm_evaluation,
  title={Human vs. LLM Business Theory Evaluation Pipeline},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]/RnRLLMEvaluation}
}
```

## 💡 Troubleshooting

### Common Issues
1. **API Key Errors**: Ensure environment variables are set correctly
2. **Rate Limiting**: Reduce semaphore sizes if hitting API limits
3. **Memory Issues**: Use smaller batch sizes for limited RAM systems
4. **Timeout Errors**: Check internet connection and API status

### Support
- Review error messages in console output
- Check checkpoint files for recovery points
- Validate input data format matches expected structure

---

**Last Updated**: September 2024 | **Python**: 3.7+ | **Status**: Production Ready