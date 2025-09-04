#!/usr/bin/env python3
"""
Visualize NIAH (Needle in a Haystack) evaluation results as a heatmap.

Usage:
    python visualize_niah.py \
        --data_file path/to/data.jsonl \
        --output_file path/to/model_output.jsonl \
        --save_path path/to/output_plot.png
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd


def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def calculate_score(outputs_gold, output_pred):
    """
    Calculate score based on how many gold answers are in the prediction.
    
    Args:
        outputs_gold: List of gold answers
        output_pred: Model's prediction string
    
    Returns:
        Score between 0 and 1
    """
    if not outputs_gold:
        return 0.0
    
    # Convert everything to strings for comparison
    outputs_gold = [str(ans).lower() for ans in outputs_gold]
    output_pred = str(output_pred).lower()
    
    # Count how many gold answers are found in the prediction
    found = sum(1 for answer in outputs_gold if answer in output_pred)
    
    # Return the fraction of answers found
    return found / len(outputs_gold)


def process_results(data_file, output_file):
    """Process NIAH results and calculate scores"""
    # Load data
    data = load_jsonl(data_file)
    outputs = load_jsonl(output_file)
    
    # Create mapping from index to output
    output_map = {item['index']: item for item in outputs}
    
    # Organize results by length and depth
    results = defaultdict(lambda: defaultdict(list))
    
    for item in data:
        idx = item['index']
        if idx not in output_map:
            print(f"Warning: No output found for index {idx}")
            continue
            
        # Get model output
        model_output = output_map[idx].get('output', '')
        
        # Calculate score
        score = calculate_score(item['outputs'], model_output)
        
        # Get target length and depth
        target_length = item.get('target_length', item.get('length'))
        depth_percent = item.get('depth_percent', 50.0)  # Default to 50% if not specified
        
        # Store result
        results[target_length][depth_percent].append(score)
    
    return results


def create_heatmap(results, save_path=None, title="NIAH Performance Heatmap"):
    """Create heatmap visualization of NIAH results"""
    # Get all unique lengths and depths
    lengths = sorted(list(results.keys()))
    depths = sorted(list(set(depth for length_results in results.values() 
                            for depth in length_results.keys())))
    
    # Create matrix for heatmap
    score_matrix = np.zeros((len(depths), len(lengths)))
    
    for i, depth in enumerate(depths):
        for j, length in enumerate(lengths):
            if depth in results[length]:
                scores = results[length][depth]
                score_matrix[i, j] = np.mean(scores) if scores else 0.0
            else:
                score_matrix[i, j] = np.nan
    
    # Create DataFrame for better labeling
    df = pd.DataFrame(
        score_matrix,
        index=[f"{d:.1f}%" for d in depths],
        columns=[f"{l//1000}K" if l >= 1000 else str(l) for l in lengths]
    )
    
    # Create figure - single performance heatmap only
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(
        df,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Accuracy Score'},
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Needle Depth Position', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance heatmap saved to {save_path}")
    
    plt.show()
    
    return df


def print_statistics(results):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("NIAH Evaluation Summary")
    print("="*60)
    
    all_scores = []
    for length, depth_results in results.items():
        for depth, scores in depth_results.items():
            all_scores.extend(scores)
    
    if all_scores:
        print(f"Overall Performance:")
        print(f"  Mean Score: {np.mean(all_scores):.3f}")
        print(f"  Std Dev: {np.std(all_scores):.3f}")
        print(f"  Min Score: {np.min(all_scores):.3f}")
        print(f"  Max Score: {np.max(all_scores):.3f}")
        print(f"  Total Samples: {len(all_scores)}")
        
        # Performance by length
        print(f"\nPerformance by Context Length:")
        for length in sorted(results.keys()):
            length_scores = []
            for scores in results[length].values():
                length_scores.extend(scores)
            if length_scores:
                display_length = f"{length//1000}K" if length >= 1000 else str(length)
                print(f"  {display_length:>6} tokens: {np.mean(length_scores):.3f} (n={len(length_scores)})")
        
        # Performance by depth
        print(f"\nPerformance by Depth Position:")
        depth_scores = defaultdict(list)
        for length_results in results.values():
            for depth, scores in length_results.items():
                depth_scores[depth].extend(scores)
        
        for depth in sorted(depth_scores.keys()):
            if depth_scores[depth]:
                print(f"  {depth:>5.1f}%: {np.mean(depth_scores[depth]):.3f} (n={len(depth_scores[depth])})")
    else:
        print("No valid scores found!")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize NIAH evaluation results')
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to the data JSONL file')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to the model output JSONL file')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save the heatmap image')
    parser.add_argument('--title', type=str, default='NIAH Performance Heatmap',
                       help='Title for the heatmap')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display the plot (only save)')
    
    args = parser.parse_args()
    
    # Process results
    print(f"Loading data from {args.data_file}")
    print(f"Loading outputs from {args.output_file}")
    results = process_results(args.data_file, args.output_file)
    
    # Print statistics
    print_statistics(results)
    
    # Create and save heatmap
    if args.no_show:
        plt.ioff()  # Turn off interactive mode
    
    df = create_heatmap(results, args.save_path, args.title)
    
    # Save scores to CSV for further analysis
    if args.save_path:
        csv_path = Path(args.save_path).with_suffix('.csv')
        df.to_csv(csv_path)
        print(f"Scores saved to {csv_path}")
    
    # Also save detailed results to JSON
    if args.save_path:
        json_path = Path(args.save_path).with_suffix('.json')
        detailed_results = {}
        for length, depth_results in results.items():
            detailed_results[str(length)] = {}
            for depth, scores in depth_results.items():
                detailed_results[str(length)][str(depth)] = {
                    'scores': scores,
                    'mean': np.mean(scores) if scores else 0.0,
                    'std': np.std(scores) if scores else 0.0,
                    'count': len(scores)
                }
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"Detailed results saved to {json_path}")


if __name__ == '__main__':
    main()