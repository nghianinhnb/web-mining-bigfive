#!/usr/bin/env python3
"""
Visualize IR and Explanation evaluation results.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

from src.config import RESULTS_DIR, LABELS_DIR

# Set style
sns.set_style("whitegrid")
rcParams['figure.figsize'] = (14, 8)
rcParams['font.size'] = 11


def load_ir_results(results_path: Path) -> pd.DataFrame:
    """Load IR evaluation results."""
    if not results_path.exists():
        print(f"IR results not found at {results_path}")
        return None
    
    df = pd.read_csv(results_path)
    return df


def load_explanation_results(summary_path: Path) -> pd.DataFrame:
    """Load explanation evaluation summary."""
    if not summary_path.exists():
        print(f"Explanation summary not found at {summary_path}")
        return None
    
    df = pd.read_csv(summary_path)
    return df


def plot_ir_results(ir_df: pd.DataFrame, output_path: Path):
    """Plot IR evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('IR Evidence Retrieval Evaluation Results', fontsize=16, fontweight='bold')
    
    # Extract metrics
    traits = ["open", "conscientious", "extroverted", "agreeable", "stable"]
    trait_labels = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Stability"]
    
    p_at_5 = [ir_df[f"p@5_{t}"].iloc[0] for t in traits]
    ndcg_at_5 = [ir_df[f"ndcg@5_{t}"].iloc[0] for t in traits]
    
    # 1. Precision@5 by trait
    ax1 = axes[0, 0]
    bars1 = ax1.bar(trait_labels, p_at_5, color=['#f59e0b', '#10b981', '#f43f5e', '#3b82f6', '#8b5cf6'])
    ax1.set_ylabel('Precision@5', fontsize=12)
    ax1.set_title('Precision@5 by Trait', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars1, p_at_5)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. nDCG@5 by trait
    ax2 = axes[0, 1]
    bars2 = ax2.bar(trait_labels, ndcg_at_5, color=['#f59e0b', '#10b981', '#f43f5e', '#3b82f6', '#8b5cf6'])
    ax2.set_ylabel('nDCG@5', fontsize=12)
    ax2.set_title('nDCG@5 by Trait', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars2, ndcg_at_5)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Combined comparison
    ax3 = axes[1, 0]
    x = np.arange(len(trait_labels))
    width = 0.35
    ax3.bar(x - width/2, p_at_5, width, label='P@5', color='#6366f1', alpha=0.8)
    ax3.bar(x + width/2, ndcg_at_5, width, label='nDCG@5', color='#8b5cf6', alpha=0.8)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('P@5 vs nDCG@5 Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(trait_labels, rotation=45, ha='right')
    ax3.set_ylim([0, 1])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Overall metrics
    ax4 = axes[1, 1]
    overall_metrics = {
        'Avg P@5': ir_df['avg_p@5'].iloc[0],
        'Avg nDCG@5': ir_df['avg_ndcg@5'].iloc[0]
    }
    bars4 = ax4.bar(overall_metrics.keys(), overall_metrics.values(), 
                    color=['#6366f1', '#8b5cf6'])
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Overall IR Metrics', fontsize=13, fontweight='bold')
    ax4.set_ylim([0, 1])
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars4, overall_metrics.values()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved IR visualization to {output_path}")


def plot_explanation_results(explain_df: pd.DataFrame, output_path: Path):
    """Plot explanation evaluation results."""
    if explain_df is None or len(explain_df) == 0:
        print("No explanation results to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Explanation Evaluation Results', fontsize=16, fontweight='bold')
    
    # Filter to criteria only (exclude overall)
    criteria_df = explain_df[explain_df['criterion'] != 'overall'].copy()
    
    if len(criteria_df) == 0:
        print("No criteria data found")
        return
    
    criteria = criteria_df['criterion'].tolist()
    means = criteria_df['mean'].tolist()
    stds = criteria_df['std'].tolist()
    
    criterion_labels = [c.capitalize() for c in criteria]
    
    # 1. Mean scores by criterion (left plot)
    ax1 = axes[0]
    # Only show error bars if std > 0
    has_errors = any(std > 0 for std in stds)
    if has_errors:
        bars1 = ax1.bar(criterion_labels, means, yerr=stds, 
                        color=['#f59e0b', '#10b981', '#f43f5e'], 
                        capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)
    else:
        bars1 = ax1.bar(criterion_labels, means, 
                        color=['#f59e0b', '#10b981', '#f43f5e'], 
                        alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Mean Rating (1-5)', fontsize=12)
    ax1.set_title('Mean Ratings by Criterion', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 5.5])
    ax1.grid(axis='y', alpha=0.3)
    # Only show text labels - don't show ±0.00
    for bar, mean, std in zip(bars1, means, stds):
        if std > 0:
            label_text = f'{mean:.2f}±{std:.2f}'
        else:
            label_text = f'{mean:.2f}'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                label_text, ha='center', va='bottom', fontweight='bold')
    
    # 2. Overall summary (right plot)
    ax2 = axes[1]
    overall_row = explain_df[explain_df['criterion'] == 'overall']
    if len(overall_row) > 0:
        overall_mean = overall_row['mean'].iloc[0]
        overall_std = overall_row['std'].iloc[0]
        # Only show error bar if std > 0
        if overall_std > 0:
            ax2.barh(['Overall'], [overall_mean], xerr=[overall_std],
                    color='#6366f1', capsize=10, alpha=0.8, edgecolor='black', linewidth=1.5)
            label_text = f'{overall_mean:.2f}±{overall_std:.2f}'
            label_x = overall_mean + overall_std + 0.1
        else:
            ax2.barh(['Overall'], [overall_mean],
                    color='#6366f1', alpha=0.8, edgecolor='black', linewidth=1.5)
            label_text = f'{overall_mean:.2f}'
            label_x = overall_mean + 0.1
        ax2.set_xlabel('Mean Rating (1-5)', fontsize=12)
        ax2.set_title('Overall Explanation Quality', fontsize=13, fontweight='bold')
        ax2.set_xlim([0, 5.5])
        ax2.grid(axis='x', alpha=0.3)
        ax2.text(label_x, 0, label_text, va='center', fontweight='bold', fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'No overall data', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Overall Explanation Quality', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved explanation visualization to {output_path}")


def plot_combined_summary(ir_df: pd.DataFrame, explain_df: pd.DataFrame, output_path: Path):
    """Create combined summary visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Evaluation Results Summary', fontsize=16, fontweight='bold')
    
    # Left: IR Results
    ax1 = axes[0]
    ir_metrics = {
        'Avg P@5': ir_df['avg_p@5'].iloc[0],
        'Avg nDCG@5': ir_df['avg_ndcg@5'].iloc[0]
    }
    bars1 = ax1.bar(ir_metrics.keys(), ir_metrics.values(), 
                   color=['#6366f1', '#8b5cf6'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('IR Evidence Retrieval', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, ir_metrics.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Right: Explanation Results
    ax2 = axes[1]
    if explain_df is not None and len(explain_df) > 0:
        overall_row = explain_df[explain_df['criterion'] == 'overall']
        if len(overall_row) > 0:
            overall_mean = overall_row['mean'].iloc[0]
            overall_std = overall_row['std'].iloc[0]
            ax2.barh(['Overall Quality'], [overall_mean], xerr=[overall_std],
                    color='#10b981', capsize=10, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax2.set_xlabel('Mean Rating (1-5)', fontsize=12)
            ax2.set_title('Explanation Quality', fontsize=13, fontweight='bold')
            ax2.set_xlim([0, 5.5])
            ax2.grid(axis='x', alpha=0.3)
            ax2.text(overall_mean + overall_std + 0.1, 0,
                    f'{overall_mean:.2f}±{overall_std:.2f}', 
                    va='center', fontweight='bold', fontsize=11)
        else:
            ax2.text(0.5, 0.5, 'No overall data', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Explanation Quality', fontsize=13, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No explanation data', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Explanation Quality', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--ir_results", type=str, default=None,
                       help="Path to IR results CSV (default: results/ir_eval.csv)")
    parser.add_argument("--explain_results", type=str, default=None,
                       help="Path to explanation summary CSV (default: results/explain_eval_summary.csv)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: results/)")
    args = parser.parse_args()
    
    # Set paths
    ir_path = Path(args.ir_results) if args.ir_results else RESULTS_DIR / "ir_eval.csv"
    explain_path = Path(args.explain_results) if args.explain_results else RESULTS_DIR / "explain_eval_summary.csv"
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading evaluation results...")
    ir_df = load_ir_results(ir_path)
    explain_df = load_explanation_results(explain_path)
    
    if ir_df is None:
        print("Cannot proceed without IR results")
        return
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    if ir_df is not None:
        plot_ir_results(ir_df, output_dir / "ir_eval_visualization.png")
    
    if explain_df is not None and len(explain_df) > 0:
        plot_explanation_results(explain_df, output_dir / "explain_eval_visualization.png")
    
    # Combined summary
    plot_combined_summary(ir_df, explain_df, output_dir / "eval_summary_combined.png")
    
    print("\n=== Visualization Complete ===")
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    if ir_df is not None:
        print("  - ir_eval_visualization.png")
    if explain_df is not None and len(explain_df) > 0:
        print("  - explain_eval_visualization.png")
    print("  - eval_summary_combined.png")


if __name__ == "__main__":
    main()
