"""
Step 4: Classify User Messages using LLMs (Parallel + Mode)
============================================================

This script classifies user messages into categories using:
1. OpenAI GPT-5.1 (with reasoning_effort="none" for speed)
2. Anthropic Claude Sonnet 4.5 (without extended thinking)

Features:
- 300 concurrent API calls for speed
- 10 repeats per message, takes MODE for robustness
- Both GPT and Claude classifications

Input:  user_messages_fulldata.xlsx (1,273 messages with all participant data)
Output: user_messages_fulldata_classified.xlsx (adds classification columns)

New columns added:
- category_gpt: GPT's mode classification
- gpt_mode_count: How many times mode appeared (out of 10)
- gpt_confidence: Confidence score (0.0 to 1.0)
- gpt_high_confidence: True if mode >= 5/10
- category_claude: Claude's mode classification
- claude_mode_count, claude_confidence, claude_high_confidence: Same for Claude
- models_agree: True if GPT and Claude agree

Requirements:
    pip install openai anthropic pandas openpyxl aiohttp

    Set environment variables:
    export OPENAI_API_KEY="your-key"
    export ANTHROPIC_API_KEY="your-key"
"""

import pandas as pd
import os
import asyncio
from collections import Counter
from typing import Optional, List
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = 'user_messages_fulldata.xlsx'
OUTPUT_FILE = 'user_messages_fulldata_classified.xlsx'

# Parallelization settings
MAX_CONCURRENT = 300  # Max concurrent API calls
NUM_REPEATS = 10      # Number of classification repeats per message

# Test mode - set to True to only process first 30 messages
TEST_MODE = False  # FULL RUN - processing all 1,273 messages
TEST_SAMPLE_SIZE = 30

# Classification categories
CATEGORIES = [
    "Task Delegation",
    "Refinement Request",
    "Evaluation Seeking",
    "Information Seeking",
    "Clarification",
    "Acknowledgment",
    "Other"
]

# The actual JSON schema (shared structure)
JSON_SCHEMA_DEFINITION = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": [
                "Task Delegation",
                "Refinement Request", 
                "Evaluation Seeking",
                "Information Seeking",
                "Clarification",
                "Acknowledgment",
                "Other"
            ],
            "description": "The category that best describes the user's message intent"
        }
    },
    "required": ["category"],
    "additionalProperties": False
}

# GPT format (uses json_schema wrapper with name/strict)
GPT_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "classification_result",
        "strict": True,
        "schema": JSON_SCHEMA_DEFINITION
    }
}

# Claude format (schema directly under output_format)
CLAUDE_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "schema": JSON_SCHEMA_DEFINITION
}

# Shared classification prompt (used by both models)
CLASSIFICATION_PROMPT = """You are a research assistant classifying user messages from an AI-assisted task.

Category definitions:
1. **Task Delegation**: User is handing off or delegating the core task to the AI (e.g., "Create a theory of value for...", "Write me a 300-word theory...")
2. **Refinement Request**: User is asking to improve, modify, shorten, lengthen, or combine existing output (e.g., "Make it shorter", "Improve this", "Combine these ideas")
3. **Evaluation Seeking**: User is asking for the AI's opinion or feedback (e.g., "What do you think?", "Is this good?")
4. **Information Seeking**: User is asking for facts, explanations, or knowledge (e.g., "What is the labor theory of value?")
5. **Clarification**: User is asking follow-up questions to understand something better
6. **Acknowledgment**: Short responses like thanks, greetings, or confirmations (e.g., "Thanks", "Hello", "OK")
7. **Other**: Message doesn't clearly fit any category

Classify the message into exactly one category."""


# =============================================================================
# API CLIENTS (Async)
# =============================================================================

def get_openai_client():
    """Initialize async OpenAI client."""
    try:
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    except ImportError:
        print("Error: openai package not installed. Run: pip install openai")
        return None


def get_anthropic_client():
    """Initialize async Anthropic client."""
    try:
        import anthropic
        return anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    except ImportError:
        print("Error: anthropic package not installed. Run: pip install anthropic")
        return None


# =============================================================================
# CLASSIFICATION FUNCTIONS (Async)
# =============================================================================

import json


def parse_json_response(response_text: str) -> str:
    """Parse category from JSON response (shared by GPT and Claude)."""
    try:
        data = json.loads(response_text.strip())
        category = data.get("category", "Other")
        # Validate it's a known category
        if category in CATEGORIES:
            return category
        return "Other"
    except json.JSONDecodeError:
        # Fallback: try to find a category in the text
        response_text = response_text.strip().lower()
        for cat in CATEGORIES:
            if cat.lower() in response_text:
                return cat
        return "Other"


async def classify_with_gpt_single(client, message: str, semaphore) -> Optional[str]:
    """Single GPT classification call with strict JSON schema output."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": CLASSIFICATION_PROMPT},
                    {"role": "user", "content": f"Classify this message:\n\n{message}"}
                ],
                response_format=GPT_OUTPUT_SCHEMA,  # Strict JSON schema (same as Claude)
                reasoning_effort="none",
                max_completion_tokens=500,  # GPT-5.1 uses max_completion_tokens, not max_tokens
                temperature=0.1  # Slight randomness for diversity in repeats
            )
            return parse_json_response(response.choices[0].message.content)
        except Exception as e:
            print(f"    GPT error: {e}")
            return None


async def classify_with_claude_single(client, message: str, semaphore) -> Optional[str]:
    """Single Claude classification call with native structured output (Nov 2025)."""
    async with semaphore:
        try:
            response = await client.beta.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=500,
                temperature=0.1,  # Match GPT for consistency
                betas=["structured-outputs-2025-11-13"],  # Enable structured outputs beta
                output_format=CLAUDE_OUTPUT_SCHEMA,  # Enforce JSON schema (same as GPT)
                messages=[
                    {
                        "role": "user",
                        "content": f"{CLASSIFICATION_PROMPT}\n\nClassify this message:\n\n{message}"
                    }
                ]
            )
            # Parse JSON response (same parser as GPT)
            return parse_json_response(response.content[0].text)
        except Exception as e:
            print(f"    Claude error: {e}")
            return None


async def classify_with_repeats(client, message: str, semaphore, classifier_func, num_repeats: int) -> tuple:
    """
    Classify a message multiple times and return mode + count + all results.
    
    Returns
    -------
    tuple
        (mode_category, mode_count, list_of_all_classifications)
    """
    tasks = [classifier_func(client, message, semaphore) for _ in range(num_repeats)]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        return ("Other", 0, [])
    
    # Get mode (most common) and its count
    counter = Counter(valid_results)
    mode_category, mode_count = counter.most_common(1)[0]
    
    return (mode_category, mode_count, valid_results)


async def classify_all_messages(df: pd.DataFrame, openai_client, anthropic_client):
    """Classify all messages in parallel with repeats."""
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # Prepare results storage
    gpt_results = []
    gpt_counts = []
    claude_results = []
    claude_counts = []
    gpt_all_classifications = []
    claude_all_classifications = []
    
    messages = df['content_clean'].tolist()
    total = len(messages)
    
    print(f"\nClassifying {total} messages...")
    print(f"  Concurrent calls: {MAX_CONCURRENT}")
    print(f"  Repeats per message: {NUM_REPEATS}")
    print(f"  Total API calls: {total * NUM_REPEATS * 2} (GPT + Claude)")
    
    # Process in batches for progress reporting
    batch_size = 100
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_messages = messages[batch_start:batch_end]
        
        batch_start_time = time.time()
        print(f"\n  Processing messages {batch_start + 1}-{batch_end}...")
        
        # GPT classifications
        if openai_client:
            print(f"    → GPT-5.1 ({NUM_REPEATS} repeats each)...")
            gpt_tasks = [
                classify_with_repeats(
                    openai_client, 
                    str(msg)[:2000], 
                    semaphore, 
                    classify_with_gpt_single, 
                    NUM_REPEATS
                )
                for msg in batch_messages
            ]
            batch_gpt_results = await asyncio.gather(*gpt_tasks)
            gpt_results.extend([r[0] for r in batch_gpt_results])
            gpt_counts.extend([r[1] for r in batch_gpt_results])
            gpt_all_classifications.extend([r[2] for r in batch_gpt_results])
        
        # Claude classifications
        if anthropic_client:
            print(f"    → Claude Sonnet 4.5 ({NUM_REPEATS} repeats each)...")
            claude_tasks = [
                classify_with_repeats(
                    anthropic_client,
                    str(msg)[:2000],
                    semaphore,
                    classify_with_claude_single,
                    NUM_REPEATS
                )
                for msg in batch_messages
            ]
            batch_claude_results = await asyncio.gather(*claude_tasks)
            claude_results.extend([r[0] for r in batch_claude_results])
            claude_counts.extend([r[1] for r in batch_claude_results])
            claude_all_classifications.extend([r[2] for r in batch_claude_results])
        
        batch_elapsed = time.time() - batch_start_time
        print(f"    ✓ Batch complete in {batch_elapsed:.1f}s")
    
    return (gpt_results, gpt_counts, gpt_all_classifications, 
            claude_results, claude_counts, claude_all_classifications)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("STEP 4: CLASSIFY USER MESSAGES (PARALLEL + MODE)")
    print("=" * 60)
    
    # Check API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    print("\nConfiguration:")
    print(f"  Max concurrent calls: {MAX_CONCURRENT}")
    print(f"  Repeats per message: {NUM_REPEATS}")
    print(f"\nAPI Key Status:")
    print(f"  OPENAI_API_KEY: {'Set' if openai_key else 'NOT SET'}")
    print(f"  ANTHROPIC_API_KEY: {'Set' if anthropic_key else 'NOT SET'}")
    
    if not openai_key and not anthropic_key:
        print("\nERROR: No API keys found. Please set at least one:")
        print("  export OPENAI_API_KEY='your-key'")
        print("  export ANTHROPIC_API_KEY='your-key'")
        return
    
    # Initialize clients
    openai_client = get_openai_client() if openai_key else None
    anthropic_client = get_anthropic_client() if anthropic_key else None
    
    # Load data
    print(f"\nLoading {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE)
    print(f"  Total messages in file: {len(df)}")
    
    # Test mode - only use first N messages
    if TEST_MODE:
        df = df.head(TEST_SAMPLE_SIZE).copy()
        print(f"  TEST MODE: Using only first {TEST_SAMPLE_SIZE} messages")
    
    # Run async classification
    start_time = time.time()
    
    (gpt_results, gpt_counts, gpt_all, 
     claude_results, claude_counts, claude_all) = asyncio.run(
        classify_all_messages(df, openai_client, anthropic_client)
    )
    
    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.1f} seconds")
    
    # Add results to dataframe
    if gpt_results:
        df['category_gpt'] = gpt_results
        df['gpt_mode_count'] = gpt_counts  # How many times mode appeared (out of 10)
        df['gpt_confidence'] = [count / NUM_REPEATS for count in gpt_counts]  # 0.0 to 1.0
        df['gpt_high_confidence'] = df['gpt_mode_count'] >= 5  # At least 5/10
        df['gpt_all_classifications'] = [str(x) for x in gpt_all]
    
    if claude_results:
        df['category_claude'] = claude_results
        df['claude_mode_count'] = claude_counts
        df['claude_confidence'] = [count / NUM_REPEATS for count in claude_counts]
        df['claude_high_confidence'] = df['claude_mode_count'] >= 5
        df['claude_all_classifications'] = [str(x) for x in claude_all]
    
    # Calculate agreement if both available
    if gpt_results and claude_results:
        # Raw agreement (all messages)
        df['models_agree'] = df['category_gpt'] == df['category_claude']
        agreement_pct = df['models_agree'].mean() * 100
        
        # High-confidence agreement (both models have mode >= 5/10)
        high_conf_mask = df['gpt_high_confidence'] & df['claude_high_confidence']
        high_conf_df = df[high_conf_mask]
        high_conf_agreement = high_conf_df['models_agree'].mean() * 100 if len(high_conf_df) > 0 else 0
        
        print(f"\n  GPT-Claude Agreement (all): {agreement_pct:.1f}%")
        print(f"  GPT-Claude Agreement (high confidence only): {high_conf_agreement:.1f}%")
        print(f"  Messages with both models high confidence: {len(high_conf_df)} ({len(high_conf_df)/len(df)*100:.1f}%)")
    
    # Summary
    print("\n" + "-" * 40)
    print("CLASSIFICATION SUMMARY")
    print("-" * 40)
    
    if gpt_results:
        print("\nGPT-5.1 Classification (Mode of 10 repeats):")
        print(df['category_gpt'].value_counts())
        print(f"\n  Mean confidence: {df['gpt_confidence'].mean():.1%}")
        print(f"  High confidence (>=5/10): {df['gpt_high_confidence'].sum()} ({df['gpt_high_confidence'].mean():.1%})")
        print(f"  Low confidence (<5/10): {(~df['gpt_high_confidence']).sum()}")
    
    if claude_results:
        print("\nClaude Sonnet 4.5 Classification (Mode of 10 repeats):")
        print(df['category_claude'].value_counts())
        print(f"\n  Mean confidence: {df['claude_confidence'].mean():.1%}")
        print(f"  High confidence (>=5/10): {df['claude_high_confidence'].sum()} ({df['claude_high_confidence'].mean():.1%})")
        print(f"  Low confidence (<5/10): {(~df['claude_high_confidence']).sum()}")
    
    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    df.to_excel(OUTPUT_FILE, index=False)
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Total time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
