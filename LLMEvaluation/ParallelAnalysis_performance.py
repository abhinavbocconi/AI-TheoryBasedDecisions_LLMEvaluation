import anthropic
import openai
import pandas as pd
import numpy as np
import json
import os
import csv
from datetime import datetime
import asyncio
import aiohttp
import time
from typing import Dict, List, Tuple, Any
import pickle
import threading

# Initialize API clients
anthropic_client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

openai_client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Parallelization settings - Level 2 (Optimized)
MAX_CONCURRENT_THEORIES = 30
ANTHROPIC_SEMAPHORE_SIZE = 50  # Allow 50 concurrent Anthropic calls
OPENAI_SEMAPHORE_SIZE = 100    # Allow 100 concurrent OpenAI calls

def read_system_prompt(filename: str) -> str:
    """Read system prompt from file"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"❌ Error: {filename} not found!")
        return None

async def call_anthropic_api_async(system_prompt: str, user_content: str, semaphore: asyncio.Semaphore) -> Dict:
    """Async call to Anthropic Claude API with rate limiting"""
    async with semaphore:
        try:
            # Using the sync client in a thread since Anthropic doesn't have native async yet
            loop = asyncio.get_event_loop()
            
            def sync_anthropic_call():
                with anthropic_client.messages.stream(
                    model="claude-sonnet-4-20250514",
                    max_tokens=32000,
                    temperature=1,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 8000
                    }
                ) as stream:
                    full_response = ""
                    for text in stream.text_stream:
                        full_response += text
                
                # Extract JSON from response
                json_start = full_response.find('{')
                json_end = full_response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = full_response[json_start:json_end]
                    return json.loads(json_str)
                else:
                    return {"error": "No JSON found in response", "raw_response": full_response}
            
            result = await loop.run_in_executor(None, sync_anthropic_call)
            return result
            
        except Exception as e:
            return {"error": str(e)}

async def call_openai_api_async(system_prompt: str, user_content: str, semaphore: asyncio.Semaphore) -> Dict:
    """Async call to OpenAI API with rate limiting"""
    async with semaphore:
        try:
            # Using the sync client in a thread
            loop = asyncio.get_event_loop()
            
            def sync_openai_call():
                response = openai_client.chat.completions.create(
                    model="o4-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    max_completion_tokens=4000
                )
                
                content = response.choices[0].message.content
                
                # Extract JSON from response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    return json.loads(json_str)
                else:
                    return {"error": "No JSON found in response", "raw_response": content}
            
            result = await loop.run_in_executor(None, sync_openai_call)
            return result
            
        except Exception as e:
            return {"error": str(e)}

async def run_parallel_analyses(system_prompt: str, theory_text: str, theory_id: int, 
                               anthropic_sem: asyncio.Semaphore, openai_sem: asyncio.Semaphore) -> Dict:
    """Run 10 analyses for each model in parallel"""
    
    print(f"    🔄 Starting parallel evaluation for Theory {theory_id}")
    
    # Create tasks for all API calls
    anthropic_tasks = [
        call_anthropic_api_async(system_prompt, theory_text, anthropic_sem)
        for _ in range(10)
    ]
    
    openai_tasks = [
        call_openai_api_async(system_prompt, theory_text, openai_sem)
        for _ in range(10)
    ]
    
    # Run all tasks concurrently
    start_time = time.time()
    anthropic_results, openai_results = await asyncio.gather(
        asyncio.gather(*anthropic_tasks),
        asyncio.gather(*openai_tasks),
        return_exceptions=True
    )
    end_time = time.time()
    
    print(f"    ✅ Theory {theory_id} completed in {end_time - start_time:.1f}s")
    
    return {
        "anthropic_results": anthropic_results,
        "openai_results": openai_results
    }

def calculate_evaluation_averages(results: List[Dict]) -> Dict:
    """Calculate averages for 5-dimension business evaluation results"""
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        return {
            "avg_novelty": 0,
            "avg_feasibility_and_scalability": 0,
            "avg_environmental_impact": 0,
            "avg_financial_impact": 0,
            "avg_quality": 0,
            "valid_runs": 0,
            "all_results": results
        }
    
    # Calculate averages for each dimension
    avg_novelty = np.mean([r.get("novelty", 0) for r in valid_results])
    avg_feasibility = np.mean([r.get("feasibility_and_scalability", 0) for r in valid_results])
    avg_environmental = np.mean([r.get("environmental_impact", 0) for r in valid_results])
    avg_financial = np.mean([r.get("financial_impact", 0) for r in valid_results])
    avg_quality = np.mean([r.get("quality", 0) for r in valid_results])
    
    return {
        "avg_novelty": round(avg_novelty, 2),
        "avg_feasibility_and_scalability": round(avg_feasibility, 2),
        "avg_environmental_impact": round(avg_environmental, 2),
        "avg_financial_impact": round(avg_financial, 2),
        "avg_quality": round(avg_quality, 2),
        "valid_runs": len(valid_results),
        "all_results": results
    }

async def process_single_theory_parallel(theory_data: Tuple, evaluation_prompt: str, 
                                       anthropic_sem: asyncio.Semaphore, openai_sem: asyncio.Semaphore) -> Dict:
    """Process single theory with parallel API calls"""
    theory_index, row = theory_data
    tid = row['teresaID']
    theory_text = str(row['theory']).strip()
    
    try:
        # Run parallel analyses for both models
        results = await run_parallel_analyses(
            evaluation_prompt, theory_text, tid, anthropic_sem, openai_sem
        )
        
        # Calculate averages
        anthropic_avg = calculate_evaluation_averages(results["anthropic_results"])
        openai_avg = calculate_evaluation_averages(results["openai_results"])
        
        # Calculate cross-model averages
        cross_model_averages = {
            "avg_novelty": round((anthropic_avg["avg_novelty"] + openai_avg["avg_novelty"]) / 2, 2),
            "avg_feasibility_and_scalability": round((anthropic_avg["avg_feasibility_and_scalability"] + openai_avg["avg_feasibility_and_scalability"]) / 2, 2),
            "avg_environmental_impact": round((anthropic_avg["avg_environmental_impact"] + openai_avg["avg_environmental_impact"]) / 2, 2),
            "avg_financial_impact": round((anthropic_avg["avg_financial_impact"] + openai_avg["avg_financial_impact"]) / 2, 2),
            "avg_quality": round((anthropic_avg["avg_quality"] + openai_avg["avg_quality"]) / 2, 2)
        }
        
        analysis_results = {
            "anthropic": anthropic_avg,
            "openai": openai_avg,
            "cross_model_averages": cross_model_averages
        }
        
        # Create output row
        output_row = create_output_row(tid, theory_text, analysis_results)
        
        print(f"✅ COMPLETED TID {tid} - Quality: {cross_model_averages['avg_quality']}/5")
        
        return output_row
        
    except Exception as e:
        print(f"❌ ERROR processing TID {tid}: {str(e)}")
        return None

def create_output_row(teresaID: int, theory_text: str, analysis_results: Dict) -> Dict:
    """Create output row for CSV with all required fields for 5-dimension evaluation"""
    row = {
        "teresaID": teresaID,
        "theory": theory_text,
        "processing_timestamp": datetime.now().isoformat(),
        
        # Anthropic Evaluation (10 runs) - All 5 dimensions
        "anthropic_avg_novelty": analysis_results["anthropic"]["avg_novelty"],
        "anthropic_avg_feasibility_and_scalability": analysis_results["anthropic"]["avg_feasibility_and_scalability"],
        "anthropic_avg_environmental_impact": analysis_results["anthropic"]["avg_environmental_impact"],
        "anthropic_avg_financial_impact": analysis_results["anthropic"]["avg_financial_impact"],
        "anthropic_avg_quality": analysis_results["anthropic"]["avg_quality"],
        "anthropic_valid_runs": analysis_results["anthropic"]["valid_runs"],
        
        # OpenAI Evaluation (10 runs) - All 5 dimensions
        "openai_avg_novelty": analysis_results["openai"]["avg_novelty"],
        "openai_avg_feasibility_and_scalability": analysis_results["openai"]["avg_feasibility_and_scalability"],
        "openai_avg_environmental_impact": analysis_results["openai"]["avg_environmental_impact"],
        "openai_avg_financial_impact": analysis_results["openai"]["avg_financial_impact"],
        "openai_avg_quality": analysis_results["openai"]["avg_quality"],
        "openai_valid_runs": analysis_results["openai"]["valid_runs"],
        
        # Cross-model averages - All 5 dimensions
        "cross_model_avg_novelty": analysis_results["cross_model_averages"]["avg_novelty"],
        "cross_model_avg_feasibility_and_scalability": analysis_results["cross_model_averages"]["avg_feasibility_and_scalability"],
        "cross_model_avg_environmental_impact": analysis_results["cross_model_averages"]["avg_environmental_impact"],
        "cross_model_avg_financial_impact": analysis_results["cross_model_averages"]["avg_financial_impact"],
        "cross_model_avg_quality": analysis_results["cross_model_averages"]["avg_quality"]
    }
    
    return row

class ThreadSafeCheckpoint:
    """Thread-safe checkpoint system for parallel processing"""
    def __init__(self, timestamp: str):
        self._lock = threading.Lock()
        self._results = []
        self._processed_teresaIDs = []
        self._timestamp = timestamp
    
    def add_result(self, result: Dict):
        with self._lock:
            if result is not None:
                self._results.append(result)
                self._processed_teresaIDs.append(result['teresaID'])
    
    def save_checkpoint(self):
        with self._lock:
            checkpoint_data = {
                'output_rows': self._results,
                'timestamp': self._timestamp,
                'processed_teresaIDs': self._processed_teresaIDs,
                'last_saved': datetime.now().isoformat()
            }
            checkpoint_file = f'checkpoint_parallel_{self._timestamp}.pkl'
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"💾 Checkpoint saved: {len(self._results)} theories completed")
    
    def get_results(self):
        with self._lock:
            return self._results.copy(), self._processed_teresaIDs.copy()

async def process_theory_batch(batch: List[Tuple], evaluation_prompt: str, 
                              checkpoint: ThreadSafeCheckpoint) -> List[Dict]:
    """Process a batch of theories in parallel"""
    
    # Create semaphores for rate limiting
    anthropic_semaphore = asyncio.Semaphore(ANTHROPIC_SEMAPHORE_SIZE)
    openai_semaphore = asyncio.Semaphore(OPENAI_SEMAPHORE_SIZE)
    
    print(f"🚀 Processing batch of {len(batch)} theories...")
    
    # Create tasks for all theories in the batch
    tasks = [
        process_single_theory_parallel(theory_data, evaluation_prompt, 
                                     anthropic_semaphore, openai_semaphore)
        for theory_data in batch
    ]
    
    # Execute all tasks concurrently
    batch_start = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    batch_end = time.time()
    
    # Add results to checkpoint
    valid_results = []
    for result in results:
        if result is not None and not isinstance(result, Exception):
            checkpoint.add_result(result)
            valid_results.append(result)
    
    # Save checkpoint after each batch
    checkpoint.save_checkpoint()
    
    print(f"✅ Batch completed: {len(valid_results)}/{len(batch)} successful in {batch_end - batch_start:.1f}s")
    return valid_results

async def main():
    """Main async function for parallel business theory evaluation"""
    print("🚀 STARTING PARALLEL BUSINESS THEORY EVALUATION")
    print(f"⚡ Level 2: {MAX_CONCURRENT_THEORIES} theories in parallel")
    print("="*80)

    # Check for required API keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set!")
        print("Please set your Anthropic API key: export ANTHROPIC_API_KEY='your-key-here'")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Read evaluation prompt
    print("📖 Loading evaluation prompt...")
    evaluation_prompt = read_system_prompt('system_prompt_performance.txt')
    
    if not evaluation_prompt:
        print("❌ Cannot proceed without evaluation prompt!")
        return
    
    print("✅ Evaluation prompt loaded successfully")
    
    # Read business theory data
    try:
        df = pd.read_csv('../data/finalData_SS_981.csv')
        print(f"✅ Theory data loaded: {len(df)} total theories")
    except FileNotFoundError:
        print("❌ Error: ../data/finalData_SS_981.csv not found!")
        return
    
    # Filter out rows with empty theory text
    df_filtered = df.dropna(subset=['theory'])
    df_filtered = df_filtered[df_filtered['theory'].str.strip() != '']
    print(f"✅ Filtered data: {len(df_filtered)} theories with content")
    
    # FULL RUN - Processing all theories
    # To test with limited theories, uncomment the line below:
    # df_filtered = df_filtered.head(30)
    print(f"🚀 FULL RUN: Processing ALL {len(df_filtered)} theories")
    
    # Initialize checkpoint system
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint = ThreadSafeCheckpoint(timestamp)
    
    # Create theory batches
    all_theories = [(i, row) for i, (_, row) in enumerate(df_filtered.iterrows(), 1)]
    batches = []
    
    for i in range(0, len(all_theories), MAX_CONCURRENT_THEORIES):
        batch = all_theories[i:i + MAX_CONCURRENT_THEORIES]
        batches.append(batch)
    
    print(f"📦 Created {len(batches)} batches of up to {MAX_CONCURRENT_THEORIES} theories each")
    
    # Process all batches
    start_time = datetime.now()
    all_results = []
    
    for batch_num, batch in enumerate(batches, 1):
        print(f"\n📋 PROCESSING BATCH {batch_num}/{len(batches)}")
        print(f"🎯 Theories {batch[0][0]}-{batch[-1][0]} of {len(all_theories)}")
        
        try:
            batch_results = await process_theory_batch(batch, evaluation_prompt, checkpoint)
            all_results.extend(batch_results)
            
            print(f"✅ Batch {batch_num} completed: {len(batch_results)} theories")
            
        except Exception as e:
            print(f"❌ Batch {batch_num} failed: {str(e)}")
            continue
    
    # Save final results
    output_rows, processed_teresaIDs = checkpoint.get_results()
    
    if output_rows:
        output_filename = f'business_evaluation_PARALLEL_{timestamp}.csv'
        
        # Define CSV headers for 5-dimension evaluation
        headers = [
            'teresaID', 'theory', 'processing_timestamp',
            'anthropic_avg_novelty', 'anthropic_avg_feasibility_and_scalability', 'anthropic_avg_environmental_impact',
            'anthropic_avg_financial_impact', 'anthropic_avg_quality', 'anthropic_valid_runs',
            'openai_avg_novelty', 'openai_avg_feasibility_and_scalability', 'openai_avg_environmental_impact',
            'openai_avg_financial_impact', 'openai_avg_quality', 'openai_valid_runs',
            'cross_model_avg_novelty', 'cross_model_avg_feasibility_and_scalability', 
            'cross_model_avg_environmental_impact', 'cross_model_avg_financial_impact', 'cross_model_avg_quality'
        ]
        
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for row in output_rows:
                writer.writerow(row)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n" + "="*80)
        print(f"🎉 PARALLEL EVALUATION COMPLETE!")
        print(f"📁 Results saved to: {output_filename}")
        print(f"📊 Theories processed: {len(output_rows)}/{len(all_theories)}")
        print(f"🔢 Total API calls made: {len(output_rows) * 20}")
        print(f"⏱️  Total processing time: {duration}")
        print(f"⚡ Average time per theory: {duration.total_seconds() / len(output_rows):.1f} seconds")
        print(f"🚀 Speedup achieved: ~{(22 * 3600) / duration.total_seconds():.1f}x faster than sequential")
        print("="*80)
        
        # Show quick summary
        if len(output_rows) > 0:
            print("📈 QUICK RESULTS PREVIEW:")
            for i, row in enumerate(output_rows[:5], 1):  # Show first 5
                print(f"  Theory {i} (teresaID {row['teresaID']}):")
                print(f"    Cross-model Quality: {row['cross_model_avg_quality']}/5")
                print(f"    Cross-model Novelty: {row['cross_model_avg_novelty']}/5")
    else:
        print("❌ No data to export!")

if __name__ == "__main__":
    asyncio.run(main())
