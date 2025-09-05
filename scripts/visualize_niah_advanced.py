#!/usr/bin/env python3
"""
Advanced NIAH visualization with multiple plot types and analysis options.

Usage:
    python visualize_niah_advanced.py \
        --data_file path/to/data.jsonl \
        --output_file path/to/model_output.jsonl \
        --save_dir path/to/output_directory \
        --plot_types heatmap line scatter
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.gridspec as gridspec


def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def calculate_score(outputs_gold, output_pred, partial_credit=False):
    """
    Calculate score with optional partial credit.
    
    Args:
        outputs_gold: List of gold answers
        output_pred: Model's prediction string
        partial_credit: If True, give partial credit for each found answer
    
    Returns:
        Score between 0 and 1
    """
    if not outputs_gold:
        return 0.0
    
    # Convert everything to strings for comparison
    outputs_gold = [str(ans).lower().strip() for ans in outputs_gold]
    output_pred = str(output_pred).lower()
    
    if partial_credit:
        # Count how many gold answers are found in the prediction
        found = sum(1 for answer in outputs_gold if answer in output_pred)
        return found / len(outputs_gold)
    else:
        # All or nothing scoring
        return 1.0 if all(answer in output_pred for answer in outputs_gold) else 0.0


def process_results(data_file, output_file, partial_credit=False):
    """Process NIAH results with detailed tracking"""
    data = load_jsonl(data_file)
    outputs = load_jsonl(output_file)
    
    # Create mapping from index to output
    output_map = {item['index']: item for item in outputs}
    
    # Organize results with detailed information
    results = defaultdict(lambda: defaultdict(list))
    detailed_results = []
    
    for item in data:
        idx = item['index']
        if idx not in output_map:
            print(f"Warning: No output found for index {idx}")
            continue
            
        # Get model output
        model_output = output_map[idx].get('output', '')
        
        # Calculate score
        score = calculate_score(item['outputs'], model_output, partial_credit)
        
        # Get metadata
        target_length = item.get('target_length', item.get('length'))
        depth_percent = item.get('depth_percent', 50.0)
        needle_position = item.get('token_position_answer', -1)
        
        # Store results
        results[target_length][depth_percent].append(score)
        
        # Store detailed result
        detailed_results.append({
            'index': idx,
            'score': score,
            'target_length': target_length,
            'depth_percent': depth_percent,
            'needle_position': needle_position,
            'gold_answers': item['outputs'],
            'model_output': model_output[:200],  # Truncate for storage
            'exact_match': score == 1.0
        })
    
    return results, detailed_results


def create_heatmap(results, save_dir=None, title="NIAH Performance Heatmap", 
                   figsize=(12, 8), annotate=True):
    """Create separate performance and sample count heatmaps"""
    lengths = sorted(list(results.keys()))
    depths = sorted(list(set(depth for length_results in results.values() 
                            for depth in length_results.keys())))
    
    # Create matrices
    score_matrix = np.zeros((len(depths), len(lengths)))
    sample_counts = np.zeros((len(depths), len(lengths)))
    
    for i, depth in enumerate(depths):
        for j, length in enumerate(lengths):
            if depth in results[length]:
                scores = results[length][depth]
                score_matrix[i, j] = np.mean(scores) if scores else 0.0
                sample_counts[i, j] = len(scores)
            else:
                score_matrix[i, j] = np.nan
    
    # Create DataFrames
    df = pd.DataFrame(
        score_matrix,
        index=[f"{d:.1f}%" for d in depths],
        columns=[f"{l//1000}K" if l >= 1000 else str(l) for l in lengths]
    )
    
    df_counts = pd.DataFrame(
        sample_counts,
        index=[f"{d:.1f}%" for d in depths],
        columns=[f"{l//1000}K" if l >= 1000 else str(l) for l in lengths]
    )
    
    # Create performance heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        df,
        annot=annotate,
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
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance heatmap saved to {save_path}")
    
    plt.show()
    
    # Create sample count heatmap
    plt.figure(figsize=(figsize[0]*0.8, figsize[1]*0.8))
    sns.heatmap(
        df_counts,
        annot=True,
        fmt='.0f',
        cmap='Blues',
        cbar_kws={'label': 'Sample Count'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Sample Distribution', fontsize=14)
    plt.xlabel('Context Length (tokens)', fontsize=12)
    plt.ylabel('Needle Depth Position', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_dir:
        count_path = Path(save_dir) / 'sample_counts.png'
        plt.savefig(count_path, dpi=300, bbox_inches='tight')
        print(f"Sample count heatmap saved to {count_path}")
    
    plt.show()
    
    return df, df_counts


def create_line_plot(results, save_path=None, title="Performance by Context Length"):
    """Create line plot showing performance across lengths"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Performance by length (averaged across depths)
    lengths = sorted(results.keys())
    mean_scores = []
    std_scores = []
    
    for length in lengths:
        all_scores = []
        for scores in results[length].values():
            all_scores.extend(scores)
        mean_scores.append(np.mean(all_scores) if all_scores else 0)
        std_scores.append(np.std(all_scores) if all_scores else 0)
    
    length_labels = [f"{l//1000}K" if l >= 1000 else str(l) for l in lengths]
    
    ax1.errorbar(range(len(lengths)), mean_scores, yerr=std_scores, 
                marker='o', linestyle='-', capsize=5, capthick=2)
    ax1.set_xlabel('Context Length', fontsize=12)
    ax1.set_ylabel('Mean Accuracy', fontsize=12)
    ax1.set_title('Performance by Context Length', fontsize=14)
    ax1.set_xticks(range(len(lengths)))
    ax1.set_xticklabels(length_labels, rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Performance by depth (averaged across lengths)
    all_depths = sorted(set(depth for length_results in results.values() 
                           for depth in length_results.keys()))
    depth_mean_scores = []
    depth_std_scores = []
    
    for depth in all_depths:
        all_scores = []
        for length_results in results.values():
            if depth in length_results:
                all_scores.extend(length_results[depth])
        depth_mean_scores.append(np.mean(all_scores) if all_scores else 0)
        depth_std_scores.append(np.std(all_scores) if all_scores else 0)
    
    ax2.errorbar(all_depths, depth_mean_scores, yerr=depth_std_scores,
                marker='s', linestyle='-', capsize=5, capthick=2, color='orange')
    ax2.set_xlabel('Needle Depth Position (%)', fontsize=12)
    ax2.set_ylabel('Mean Accuracy', fontsize=12)
    ax2.set_title('Performance by Needle Depth', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Line plot saved to {save_path}")
    
    return fig


def create_scatter_plot(detailed_results, save_path=None, title="Needle Position vs Score"):
    """Create scatter plot of needle position vs score"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by length for different colors
    df = pd.DataFrame(detailed_results)
    df = df[df['needle_position'] >= 0]  # Filter valid positions
    
    # Create scatter plot with different colors for different lengths
    unique_lengths = sorted(df['target_length'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_lengths)))
    
    for i, length in enumerate(unique_lengths):
        length_data = df[df['target_length'] == length]
        label = f"{length//1000}K" if length >= 1000 else str(length)
        ax.scatter(length_data['needle_position'], length_data['score'],
                  alpha=0.6, s=30, color=colors[i], label=label)
    
    ax.set_xlabel('Needle Token Position', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(title='Context Length', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {save_path}")
    
    return fig


def create_distribution_plot(detailed_results, save_path=None):
    """Create distribution plots for scores"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    df = pd.DataFrame(detailed_results)
    
    # Overall score distribution
    axes[0, 0].hist(df['score'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Score', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Overall Score Distribution', fontsize=12)
    axes[0, 0].axvline(df['score'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["score"].mean():.3f}')
    axes[0, 0].legend()
    
    # Score distribution by length bins
    length_bins = pd.cut(df['target_length'], bins=5)
    df['length_bin'] = length_bins
    
    for bin_name, group in df.groupby('length_bin'):
        axes[0, 1].hist(group['score'], alpha=0.5, label=str(bin_name), bins=10)
    axes[0, 1].set_xlabel('Score', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('Score Distribution by Length Range', fontsize=12)
    axes[0, 1].legend()
    
    # Score distribution by depth bins
    depth_bins = pd.cut(df['depth_percent'], bins=5)
    df['depth_bin'] = depth_bins
    
    for bin_name, group in df.groupby('depth_bin'):
        axes[1, 0].hist(group['score'], alpha=0.5, label=str(bin_name), bins=10)
    axes[1, 0].set_xlabel('Score', fontsize=11)
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('Score Distribution by Depth Range', fontsize=12)
    axes[1, 0].legend()
    
    # Exact match percentage by length
    exact_match_by_length = df.groupby('target_length')['exact_match'].mean()
    lengths = exact_match_by_length.index
    length_labels = [f"{l//1000}K" if l >= 1000 else str(l) for l in lengths]
    
    axes[1, 1].bar(range(len(lengths)), exact_match_by_length.values)
    axes[1, 1].set_xlabel('Context Length', fontsize=11)
    axes[1, 1].set_ylabel('Exact Match Rate', fontsize=11)
    axes[1, 1].set_title('Exact Match Rate by Context Length', fontsize=12)
    axes[1, 1].set_xticks(range(len(lengths)))
    axes[1, 1].set_xticklabels(length_labels, rotation=45)
    axes[1, 1].set_ylim([0, 1])
    
    plt.suptitle('NIAH Score Distributions', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
    
    return fig


def save_detailed_report(results, detailed_results, save_path):
    """Save detailed text report"""
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("NIAH EVALUATION DETAILED REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Overall statistics
        all_scores = [r['score'] for r in detailed_results]
        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"  Total Samples: {len(all_scores)}\n")
        f.write(f"  Mean Score: {np.mean(all_scores):.4f}\n")
        f.write(f"  Std Dev: {np.std(all_scores):.4f}\n")
        f.write(f"  Min Score: {np.min(all_scores):.4f}\n")
        f.write(f"  Max Score: {np.max(all_scores):.4f}\n")
        f.write(f"  Exact Match Rate: {sum(r['exact_match'] for r in detailed_results)/len(detailed_results):.4f}\n")
        f.write("\n")
        
        # Performance by length
        f.write("PERFORMANCE BY CONTEXT LENGTH:\n")
        for length in sorted(results.keys()):
            length_scores = []
            for scores in results[length].values():
                length_scores.extend(scores)
            if length_scores:
                display_length = f"{length//1000}K" if length >= 1000 else str(length)
                f.write(f"  {display_length:>8} tokens: ")
                f.write(f"Mean={np.mean(length_scores):.4f}, ")
                f.write(f"Std={np.std(length_scores):.4f}, ")
                f.write(f"N={len(length_scores)}\n")
        f.write("\n")
        
        # Performance by depth
        f.write("PERFORMANCE BY NEEDLE DEPTH:\n")
        depth_scores = defaultdict(list)
        for length_results in results.values():
            for depth, scores in length_results.items():
                depth_scores[depth].extend(scores)
        
        for depth in sorted(depth_scores.keys()):
            if depth_scores[depth]:
                f.write(f"  {depth:>6.1f}%: ")
                f.write(f"Mean={np.mean(depth_scores[depth]):.4f}, ")
                f.write(f"Std={np.std(depth_scores[depth]):.4f}, ")
                f.write(f"N={len(depth_scores[depth])}\n")
        f.write("\n")
        
        # Performance matrix
        f.write("PERFORMANCE MATRIX (Mean Scores):\n")
        f.write("-"*70 + "\n")
        
        lengths = sorted(results.keys())
        depths = sorted(set(depth for length_results in results.values() 
                          for depth in length_results.keys()))
        
        # Header
        f.write("Depth\\Length  ")
        for length in lengths:
            display_length = f"{length//1000}K" if length >= 1000 else str(length)
            f.write(f"{display_length:>8} ")
        f.write("\n")
        
        # Data rows
        for depth in depths:
            f.write(f"{depth:>6.1f}%      ")
            for length in lengths:
                if depth in results[length] and results[length][depth]:
                    score = np.mean(results[length][depth])
                    f.write(f"{score:>8.3f} ")
                else:
                    f.write(f"{'N/A':>8} ")
            f.write("\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"Detailed report saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Advanced NIAH visualization')
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to the data JSONL file')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to the model output JSONL file')
    parser.add_argument('--save_dir', type=str, default='./niah_results',
                       help='Directory to save all outputs')
    parser.add_argument('--plot_types', nargs='+', 
                       default=['heatmap', 'line', 'scatter', 'distribution'],
                       choices=['heatmap', 'line', 'scatter', 'distribution', 'all'],
                       help='Types of plots to generate')
    parser.add_argument('--partial_credit', action='store_true',
                       help='Use partial credit scoring')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display plots')
    parser.add_argument('--title_prefix', type=str, default='',
                       help='Prefix for plot titles')
    
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Process results
    print(f"Loading data from {args.data_file}")
    print(f"Loading outputs from {args.output_file}")
    print(f"Using {'partial' if args.partial_credit else 'exact match'} scoring")
    
    results, detailed_results = process_results(
        args.data_file, args.output_file, args.partial_credit
    )
    
    if args.no_show:
        plt.ioff()
    
    # Generate plots based on user selection
    if 'all' in args.plot_types:
        args.plot_types = ['heatmap', 'line', 'scatter', 'distribution']
    
    prefix = args.title_prefix + ' ' if args.title_prefix else ''
    
    if 'heatmap' in args.plot_types:
        df, df_counts = create_heatmap(
            results, 
            save_dir / 'heatmap.png',
            title=f'{prefix}NIAH Performance Heatmap'
        )
        # Save data to CSV
        df.to_csv(save_dir / 'scores_matrix.csv')
        df_counts.to_csv(save_dir / 'counts_matrix.csv')
    
    if 'line' in args.plot_types:
        create_line_plot(
            results,
            save_dir / 'line_plot.png',
            title=f'{prefix}Performance Trends'
        )
    
    if 'scatter' in args.plot_types:
        create_scatter_plot(
            detailed_results,
            save_dir / 'scatter_plot.png',
            title=f'{prefix}Needle Position vs Score'
        )
    
    if 'distribution' in args.plot_types:
        create_distribution_plot(
            detailed_results,
            save_dir / 'distributions.png'
        )
    
    # Save detailed results
    save_detailed_report(results, detailed_results, save_dir / 'report.txt')
    
    # Save detailed results to JSON
    with open(save_dir / 'detailed_results.json', 'w') as f:
        json.dump({
            'summary': {
                'mean_score': float(np.mean([r['score'] for r in detailed_results])),
                'std_score': float(np.std([r['score'] for r in detailed_results])),
                'total_samples': len(detailed_results),
                'exact_match_rate': sum(r['exact_match'] for r in detailed_results) / len(detailed_results)
            },
            'results_by_config': {
                str(length): {
                    str(depth): {
                        'scores': results[length][depth],
                        'mean': float(np.mean(results[length][depth])) if results[length][depth] else 0,
                        'count': len(results[length][depth])
                    }
                    for depth in results[length]
                }
                for length in results
            }
        }, f, indent=2)
    
    print(f"\nAll results saved to {save_dir}/")
    print("Files generated:")
    print(f"  - heatmap.png: Main performance heatmap")
    print(f"  - line_plot.png: Performance trends")
    print(f"  - scatter_plot.png: Needle position analysis")
    print(f"  - distributions.png: Score distributions")
    print(f"  - report.txt: Detailed text report")
    print(f"  - scores_matrix.csv: Score matrix data")
    print(f"  - counts_matrix.csv: Sample count matrix")
    print(f"  - detailed_results.json: Complete results in JSON")
    
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()