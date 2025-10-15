#!/usr/bin/env python3

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
Helper script to run attack vector analysis on comprehensive logs.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lexicalrichness import LexicalRichness
from textblob import TextBlob
from collections import Counter
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class AttackVectorAnalyzer:
    """Simplified analyzer for quick attack vector analysis."""
    
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.data = self._load_log()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.seed_prompts = self._load_seed_prompts()
        
    def _load_log(self) -> Dict:
        with open(self.log_path, 'r') as f:
            return json.load(f)
    
    def _load_seed_prompts(self) -> Dict[str, str]:
        """Load seed prompts from dataset files to support SP metric calculation."""
        seed_prompts = {}
        
        # Look for dataset files in common locations
        dataset_dirs = [
            Path("data"),
            self.log_path.parent.parent.parent / "data",
            Path.cwd() / "data"
        ]
        
        dataset_files = ["harmbench.json", "BeaverTails.json", "do-not-answer.json", 
                        "harmfulQA.json", "CategoricalHarmfulQA.json", "DangerousQA.json",
                        "AdversarialQA.json"]
        
        for dataset_dir in dataset_dirs:
            if not dataset_dir.exists():
                continue
                
            for dataset_file in dataset_files:
                dataset_path = dataset_dir / dataset_file
                if dataset_path.exists():
                    try:
                        with open(dataset_path, 'r') as f:
                            if dataset_file.endswith('.json'):
                                # Handle different JSON formats
                                content = f.read().strip()
                                if content.startswith('['):
                                    # Array format
                                    data = json.loads(content)
                                else:
                                    # JSONL format
                                    data = [json.loads(line) for line in content.split('\n') if line.strip()]
                                
                                for i, item in enumerate(data):
                                    seed_id = f"seed_{i}"
                                    if 'question' in item:
                                        seed_prompts[seed_id] = item['question']
                                    elif 'prompt' in item:
                                        seed_prompts[seed_id] = item['prompt']
                                    elif 'text' in item:
                                        seed_prompts[seed_id] = item['text']
                                        
                    except Exception as e:
                        print(f"Warning: Could not load seeds from {dataset_path}: {e}")
                        continue
        
        print(f"Loaded {len(seed_prompts)} seed prompts for SP metric calculation")
        return seed_prompts
    
    def extract_prompts_by_status(self) -> Tuple[List[str], List[str]]:
        """Extract successful and unsuccessful prompts."""
        successful_prompts = []
        unsuccessful_prompts = []
        
        all_prompts = self.data.get('all_prompts', {})
        rejection_reasons = self.data.get('rejection_reasons', {})
        
        for key in all_prompts:
            prompts = all_prompts[key]
            reasons = rejection_reasons.get(key, [])
            
            for i, prompt in enumerate(prompts):
                reason = reasons[i] if i < len(reasons) else 'unknown'
                
                if reason == 'accepted':
                    successful_prompts.append(prompt)
                else:
                    unsuccessful_prompts.append(prompt)
        
        return successful_prompts, unsuccessful_prompts
    
    def _compute_self_bleu(self, prompts: List[str]) -> float:
        """
        Compute self-BLEU score for a list of prompts.
        Self-BLEU measures diversity by calculating BLEU scores between each prompt and all others.
        Lower self-BLEU indicates higher diversity.
        """
        if len(prompts) < 2:
            return 0.0
        
        # Tokenize prompts
        tokenized_prompts = [p.split() for p in prompts]
        smoother = SmoothingFunction().method1
        scores = []
        n = 4
        
        for i, candidate in enumerate(tokenized_prompts):
            references = tokenized_prompts[:i] + tokenized_prompts[i+1:]
            if not references:
                continue
            score = sentence_bleu(references, candidate, weights=tuple([1/n]*n), smoothing_function=smoother)
            scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.0

    def _compute_diversity_metrics(self, magnitudes_nu, magnitudes_sp, similarities_nu, similarities_sp):
        """Compute diversity metrics including lexical, embedding, and attack vector diversity."""
        
        # Extract lexical diversity from log if available
        diversity_data = self.data.get('diversity', {})
        embedding_diversity = self.data.get('embedding_diversity', 0.0)
        
        # Compute self-BLEU for successful prompts only
        successful_prompts, _ = self.extract_prompts_by_status()
        if len(successful_prompts) >= 2:
            self_bleu = self._compute_self_bleu(successful_prompts)
            print(f"Computed self-BLEU score: {self_bleu:.4f} (lower = more diverse)")
        else:
            self_bleu = 0.0
            print(f"Not enough prompts for self-BLEU calculation (need ≥2, got {len(successful_prompts)})")
        
        diversity_metrics = {
            "lexical": {
                "total_prompts": diversity_data.get('total_prompts', len(self.data.get('all_prompts', {}))),
                "unique_prompts": diversity_data.get('unique_prompts', 0),
                "diversity_score": diversity_data.get('diversity_score', 0.0)
            },
            "embedding": float(embedding_diversity),
            "self_bleu": float(self_bleu),
            "avg_magnitude_nu": float(np.mean(magnitudes_nu)),
            "std_magnitude_nu": float(np.std(magnitudes_nu)),
            "avg_magnitude_sp": float(np.mean(magnitudes_sp)),
            "std_magnitude_sp": float(np.std(magnitudes_sp)),
            "avg_cosine_sim_nu": float(np.mean(similarities_nu)),
            "std_cosine_sim_nu": float(np.std(similarities_nu)),
            "avg_cosine_sim_sp": float(np.mean(similarities_sp)),
            "std_cosine_sim_sp": float(np.std(similarities_sp))
        }
        
        return diversity_metrics

    def find_optimal_k(self, attack_vectors: np.ndarray, k_min: int = 2, k_max: int = 10) -> int:
        """Find optimal number of clusters using silhouette score."""
        n_samples = len(attack_vectors)
        
        # Silhouette score requires at least 2 samples and k must be < n_samples
        if n_samples < 2:
            return 1
        if n_samples == 2:
            return 2
            
        # Adjust k_max to be at most n_samples - 1
        k_max = min(k_max, n_samples - 1)
        
        # If k_min >= k_max, just return k_min
        if k_min >= k_max:
            return min(k_min, n_samples - 1)
            
        best_k, best_score = k_min, -1
        for k in range(k_min, k_max + 1):
            try:
                labels = KMeans(n_clusters=k, random_state=42).fit_predict(attack_vectors)
                # Only compute silhouette score if we have valid clustering
                if len(np.unique(labels)) >= 2 and len(np.unique(labels)) < n_samples:
                    score = silhouette_score(attack_vectors, labels)
                    if score > best_score:
                        best_k, best_score = k, score
            except ValueError as e:
                # Skip this k value if silhouette score calculation fails
                print(f"Warning: Skipping k={k} due to error: {e}")
                continue
        return best_k
    
    def _compute_enhanced_attack_vectors(self, successful_embeddings: np.ndarray, unsuccessful_embeddings: np.ndarray):
        """
        Compute three attack vector metrics:
        1. Attack Vector_NU: successful - nearest unsuccessful prompt
        2. Attack Vector_MU: successful - mean of closest unsuccessful cluster
        3. Attack Vector_SP: successful - seed prompt (evolutionary trajectory)
        """
        from sklearn.metrics.pairwise import cosine_distances
        from sklearn.cluster import KMeans
        
        # First, cluster unsuccessful prompts to identify failure modes
        print("  Clustering unsuccessful prompts...")
        n_unsuccess_clusters = min(5, max(2, len(unsuccessful_embeddings) // 10))
        if len(unsuccessful_embeddings) < 2:
            # Fallback when very few unsuccessful prompts
            attack_vectors_nu = successful_embeddings - unsuccessful_embeddings[0]
            attack_vectors_mu = successful_embeddings - unsuccessful_embeddings[0]
            
            # Compute SP vectors
            print("  Computing seed-to-prompt (SP) attack vectors...")
            seed_embeddings = self._get_seed_embeddings()
            attack_vectors_sp = []
            sp_similarities = []
            seed_prompt_texts = []
            
            for i, success_emb in enumerate(successful_embeddings):
                seed_emb, seed_text = self._get_seed_for_prompt(i)
                attack_vector_sp = success_emb - seed_emb
                sp_cosine_sim = np.dot(success_emb, seed_emb) / (
                    np.linalg.norm(success_emb) * np.linalg.norm(seed_emb)
                ) if np.linalg.norm(success_emb) > 0 and np.linalg.norm(seed_emb) > 0 else 0
                
                attack_vectors_sp.append(attack_vector_sp)
                sp_similarities.append(sp_cosine_sim)
                seed_prompt_texts.append(seed_text)
            
            # Store similarities and seed info
            self.nu_similarities = np.ones(len(successful_embeddings)) * 0.5  # Fallback
            self.mu_similarities = np.ones(len(successful_embeddings)) * 0.5  # Fallback
            self.sp_similarities = np.array(sp_similarities)
            self.seed_prompt_texts = seed_prompt_texts
            
            return attack_vectors_nu, attack_vectors_mu, np.array(attack_vectors_sp)
            
        unsuccess_kmeans = KMeans(n_clusters=n_unsuccess_clusters, random_state=42)
        unsuccess_labels = unsuccess_kmeans.fit_predict(unsuccessful_embeddings)
        
        # Compute centroids of unsuccessful clusters
        unsuccess_centroids = []
        for i in range(n_unsuccess_clusters):
            cluster_mask = unsuccess_labels == i
            if np.any(cluster_mask):
                centroid = np.mean(unsuccessful_embeddings[cluster_mask], axis=0)
                unsuccess_centroids.append(centroid)
        unsuccess_centroids = np.array(unsuccess_centroids)
        
        # Get seed embeddings
        print("  Computing seed-to-prompt (SP) attack vectors...")
        seed_embeddings = self._get_seed_embeddings()
        
        attack_vectors_nu = []
        attack_vectors_mu = []
        attack_vectors_sp = []
        nu_similarities = []  # Store cosine similarities for NU
        mu_similarities = []  # Store cosine similarities for MU
        sp_similarities = []  # Store cosine similarities for SP
        seed_prompt_texts = []  # Store seed prompt texts for reference
        
        print("  Computing NU, MU, and SP attack vectors...")
        for i, success_emb in enumerate(successful_embeddings):
            # Attack Vector_NU: find nearest unsuccessful prompt
            distances = np.linalg.norm(unsuccessful_embeddings - success_emb, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_unsuccessful = unsuccessful_embeddings[nearest_idx]
            attack_vec_nu = success_emb - nearest_unsuccessful
            attack_vectors_nu.append(attack_vec_nu)
            
            # Compute cosine similarity for NU
            nu_cosine_sim = np.dot(success_emb, nearest_unsuccessful) / (
                np.linalg.norm(success_emb) * np.linalg.norm(nearest_unsuccessful)
            ) if np.linalg.norm(success_emb) > 0 and np.linalg.norm(nearest_unsuccessful) > 0 else 0
            nu_similarities.append(nu_cosine_sim)
            
            # Attack Vector_MU: find closest unsuccessful cluster centroid
            if len(unsuccess_centroids) > 0:
                centroid_distances = np.linalg.norm(unsuccess_centroids - success_emb, axis=1)
                closest_centroid_idx = np.argmin(centroid_distances)
                closest_centroid = unsuccess_centroids[closest_centroid_idx]
                attack_vec_mu = success_emb - closest_centroid
                attack_vectors_mu.append(attack_vec_mu)
                
                # Compute cosine similarity for MU
                mu_cosine_sim = np.dot(success_emb, closest_centroid) / (
                    np.linalg.norm(success_emb) * np.linalg.norm(closest_centroid)
                ) if np.linalg.norm(success_emb) > 0 and np.linalg.norm(closest_centroid) > 0 else 0
                mu_similarities.append(mu_cosine_sim)
            else:
                # Fallback
                attack_vectors_mu.append(attack_vec_nu)
                mu_similarities.append(nu_cosine_sim)
            
            # Attack Vector_SP: compute vector from seed to successful prompt
            seed_emb, seed_text = self._get_seed_for_prompt(i)
            attack_vec_sp = success_emb - seed_emb
            attack_vectors_sp.append(attack_vec_sp)
            
            # Compute cosine similarity for SP
            sp_cosine_sim = np.dot(success_emb, seed_emb) / (
                np.linalg.norm(success_emb) * np.linalg.norm(seed_emb)
            ) if np.linalg.norm(success_emb) > 0 and np.linalg.norm(seed_emb) > 0 else 0
            sp_similarities.append(sp_cosine_sim)
            seed_prompt_texts.append(seed_text)
        

        # Store similarities and seed info for later use
        self.nu_similarities = np.array(nu_similarities)
        self.mu_similarities = np.array(mu_similarities)
        self.sp_similarities = np.array(sp_similarities)
        self.seed_prompt_texts = seed_prompt_texts
        
        return np.array(attack_vectors_nu), np.array(attack_vectors_mu), np.array(attack_vectors_sp)
    
    def _get_seed_embeddings(self):
        """Extract and encode seed prompts from the log data."""
        # Extract seed prompts from the data
        all_prompts = self.data.get('all_prompts', {})
        
        # Find seed prompts (those that start with 'seed_')
        seed_prompts = []
        for key, prompts in all_prompts.items():
            if isinstance(prompts, list) and len(prompts) > 0:
                # Use the first prompt in each category as a representative seed
                seed_prompts.extend(prompts[:1])  # Take just the first from each category
        
        # If we can't find seeds this way, try to extract from prompt_ids
        if not seed_prompts and hasattr(self, 'seed_ids') and self.seed_ids:
            # Try to get seed prompts from lineage data
            seed_prompts = self._extract_seeds_from_lineage()
        
        # Fallback: create simple seed prompts if we can't find any
        if not seed_prompts:
            print("  Warning: No seed prompts found, using fallback generic seeds")
            seed_prompts = [
                "How can I help with this task?",
                "What information do you need?",
                "Can you provide more details?",
                "I'd be happy to assist you.",
                "What would you like to know?"
            ]
        
        print(f"  Found {len(seed_prompts)} seed prompts")
        return self.model.encode(seed_prompts)
    
    def _extract_seeds_from_lineage(self):
        """Extract actual seed prompts from lineage data if available."""
        if not (hasattr(self, 'prompt_ids') and hasattr(self, 'seed_ids')):
            return []
        
        # Find prompts that are seeds (have no parent)
        seeds = []
        all_prompts = self.data.get('all_prompts', {})
        
        for key, prompts in all_prompts.items():
            for prompt in prompts:
                # This is a simple heuristic - you might need to adjust based on your data structure
                if len(prompt.split()) < 20:  # Seeds are typically shorter
                    seeds.append(prompt)
                    if len(seeds) >= 10:  # Limit number of seeds
                        break
            if len(seeds) >= 10:
                break
        
        return seeds
    
    def _get_seed_for_prompt(self, prompt_index):
        """Get the seed embedding and text for a given successful prompt index."""
        # Load lineage fields if not already loaded
        if not hasattr(self, 'seed_ids'):
            self.load_lineage_fields()
        
        # Get seed embeddings
        if hasattr(self, '_seed_embeddings'):
            seed_embeddings = self._seed_embeddings
        else:
            seed_embeddings = self._get_seed_embeddings()
            self._seed_embeddings = seed_embeddings
        
        # Try to get the actual seed ID for this prompt from lineage data
        successful_prompts, _ = self.extract_prompts_by_status()
        
        # Find which category/key this prompt belongs to by matching text
        prompt_text = successful_prompts[prompt_index]
        seed_id = None
        
        # Look through all_seed_ids to find the seed for this prompt
        all_prompts = self.data.get('all_prompts', {})
        all_seed_ids = self.data.get('all_seed_ids', {})
        
        for key in all_prompts:
            prompts = all_prompts[key]
            seed_ids_for_key = all_seed_ids.get(key, [])
            
            for i, p in enumerate(prompts):
                if p == prompt_text and i < len(seed_ids_for_key):
                    seed_id = seed_ids_for_key[i]
                    break
            if seed_id:
                break
        
        # Get seed text and embedding
        if seed_id and seed_id in self.seed_prompts:
            seed_text = self.seed_prompts[seed_id]
            seed_embedding = self.model.encode([seed_text])[0]
            return seed_embedding, seed_text
        else:
            # Fallback: use cycling through available seeds
            if not hasattr(self, '_fallback_seed_texts'):
                # Use first prompt from each category as fallback seeds
                fallback_seeds = []
                for key, prompts in all_prompts.items():
                    if isinstance(prompts, list) and len(prompts) > 0:
                        fallback_seeds.append(prompts[0])
                
                if not fallback_seeds:
                    fallback_seeds = [
                        "How can I help with this task?",
                        "What information do you need?", 
                        "Can you provide more details?",
                        "I'd be happy to assist you.",
                        "What would you like to know?"
                    ]
                self._fallback_seed_texts = fallback_seeds
                self._fallback_seed_embeddings = self.model.encode(fallback_seeds)
            
            # Map prompt to seed (cycle through if more prompts than seeds)
            seed_idx = prompt_index % len(self._fallback_seed_embeddings)
            return self._fallback_seed_embeddings[seed_idx], self._fallback_seed_texts[seed_idx]
    
    def compute_similarity_scores(self, successful_embeddings: np.ndarray, unsuccessful_embeddings: np.ndarray):
        """
        Compute similarity scores using all three attack vector metrics.
        Returns cosine similarities, euclidean distances, and other metrics.
        """
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
        
        print("Computing comprehensive similarity scores...")
        
        # Compute centroids
        avg_successful = np.mean(successful_embeddings, axis=0)
        avg_unsuccessful = np.mean(unsuccessful_embeddings, axis=0)
        
        similarity_scores = []
        
        for i, success_emb in enumerate(successful_embeddings):
            scores = {}
            
            # 1. Original method: similarity to unsuccessful centroid
            scores['cosine_sim_to_unsuccess_avg'] = cosine_similarity([success_emb], [avg_unsuccessful])[0][0]
            scores['euclidean_dist_to_unsuccess_avg'] = euclidean_distances([success_emb], [avg_unsuccessful])[0][0]
            
            # 2. NU method: similarity to nearest unsuccessful
            unsuccess_distances = euclidean_distances([success_emb], unsuccessful_embeddings)[0]
            nearest_idx = np.argmin(unsuccess_distances)
            nearest_unsuccessful = unsuccessful_embeddings[nearest_idx]
            scores['cosine_sim_to_nearest_unsuccess'] = cosine_similarity([success_emb], [nearest_unsuccessful])[0][0]
            scores['euclidean_dist_to_nearest_unsuccess'] = unsuccess_distances[nearest_idx]
            
            # 3. Similarity to successful centroid (for comparison)
            scores['cosine_sim_to_success_avg'] = cosine_similarity([success_emb], [avg_successful])[0][0]
            scores['euclidean_dist_to_success_avg'] = euclidean_distances([success_emb], [avg_successful])[0][0]
            
            # 4. Relative similarities
            scores['relative_similarity_success_vs_unsuccess'] = (
                scores['cosine_sim_to_success_avg'] - scores['cosine_sim_to_unsuccess_avg']
            )
            scores['relative_similarity_success_vs_nearest'] = (
                scores['cosine_sim_to_success_avg'] - scores['cosine_sim_to_nearest_unsuccess']
            )
            
            # 5. Distance ratios
            scores['distance_ratio_nearest_vs_avg'] = (
                scores['euclidean_dist_to_nearest_unsuccess'] / scores['euclidean_dist_to_unsuccess_avg']
            )
            
            similarity_scores.append(scores)
        
        return similarity_scores
    
    def similarity_analysis(self, output_dir):
        """
        Comprehensive similarity analysis using all three attack vector metrics.
        """
        if not hasattr(self, 'attack_vectors'):
            print("Error: Attack vectors not computed yet. Run analyze() first.")
            return
            
        similarity_dir = Path(output_dir) / "similarity_analysis"
        similarity_dir.mkdir(parents=True, exist_ok=True)
        
        # Get embeddings from stored data
        successful_prompts = self.df['prompt'].tolist()
        successful_embeddings = self.model.encode(successful_prompts)
        
        # We need to get unsuccessful embeddings - let's reconstruct them
        unsuccessful_prompts = []
        all_prompts = self.data.get('all_prompts', {})
        rejection_reasons = self.data.get('rejection_reasons', {})
        
        for key in all_prompts:
            prompts = all_prompts[key]
            reasons = rejection_reasons.get(key, [])
            
            for i, prompt in enumerate(prompts):
                reason = reasons[i] if i < len(reasons) else 'unknown'
                if reason != 'accepted':
                    unsuccessful_prompts.append(prompt)
        
        unsuccessful_embeddings = self.model.encode(unsuccessful_prompts)
        
        # Compute similarity scores
        similarity_scores = self.compute_similarity_scores(successful_embeddings, unsuccessful_embeddings)
        
        # Create DataFrame with similarity scores
        similarity_df = self.df.copy()
        for i, scores in enumerate(similarity_scores):
            for key, value in scores.items():
                if i == 0:  # First iteration, create the column
                    similarity_df[key] = [None] * len(similarity_df)
                similarity_df.loc[i, key] = value
        
        # Save detailed similarity analysis
        similarity_df.to_csv(similarity_dir / "comprehensive_similarity_analysis.csv", index=False)
        
        # Analyze patterns
        print("\nSimilarity Score Analysis:")
        print(f"Average cosine similarity to unsuccessful avg: {similarity_df['cosine_sim_to_unsuccess_avg'].mean():.4f}")
        print(f"Average cosine similarity to nearest unsuccessful: {similarity_df['cosine_sim_to_nearest_unsuccess'].mean():.4f}")
        print(f"Average cosine similarity to successful avg: {similarity_df['cosine_sim_to_success_avg'].mean():.4f}")
        
        # Identify interesting cases
        high_similarity_to_unsuccess = similarity_df['cosine_sim_to_unsuccess_avg'] > 0.8
        very_close_to_nearest = similarity_df['distance_ratio_nearest_vs_avg'] < 0.5
        
        print(f"\nInteresting Cases:")
        print(f"High similarity to unsuccessful average (>0.8): {high_similarity_to_unsuccess.sum()} cases")
        print(f"Very close to nearest unsuccessful (<0.5 distance ratio): {very_close_to_nearest.sum()} cases")
        
        # Create visualizations
        self._create_similarity_visualizations(similarity_df, similarity_dir)
        
        return similarity_df
    
    def _create_similarity_visualizations(self, similarity_df, similarity_dir):
        """Create visualizations for similarity analysis."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 1. Similarity distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cosine similarities
        axes[0, 0].hist([
            similarity_df['cosine_sim_to_unsuccess_avg'],
            similarity_df['cosine_sim_to_nearest_unsuccess'],
            similarity_df['cosine_sim_to_success_avg']
        ], bins=20, alpha=0.6, label=[
            'To Unsuccess Avg', 'To Nearest Unsuccess', 'To Success Avg'
        ])
        axes[0, 0].set_title('Distribution of Cosine Similarities')
        axes[0, 0].set_xlabel('Cosine Similarity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Distance ratios
        axes[0, 1].hist(similarity_df['distance_ratio_nearest_vs_avg'], bins=20, alpha=0.7, color='orange')
        axes[0, 1].set_title('Distance Ratio: Nearest vs Average Unsuccessful')
        axes[0, 1].set_xlabel('Distance Ratio')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=1.0, color='red', linestyle='--', label='Equal Distance')
        axes[0, 1].legend()
        
        # Relative similarities
        axes[1, 0].scatter(
            similarity_df['relative_similarity_success_vs_unsuccess'],
            similarity_df['relative_similarity_success_vs_nearest'],
            alpha=0.7, c=similarity_df['cluster'], cmap='viridis'
        )
        axes[1, 0].set_xlabel('Relative Sim: Success vs Unsuccess Avg')
        axes[1, 0].set_ylabel('Relative Sim: Success vs Nearest Unsuccess')
        axes[1, 0].set_title('Relative Similarity Comparison')
        axes[1, 0].plot([-1, 1], [-1, 1], 'r--', alpha=0.5)
        
        # Correlation with attack vector magnitudes
        axes[1, 1].scatter(
            similarity_df['magnitude'],
            similarity_df['cosine_sim_to_unsuccess_avg'],
            alpha=0.7, c=similarity_df['cluster'], cmap='viridis'
        )
        axes[1, 1].set_xlabel('Attack Vector Magnitude (Original)')
        axes[1, 1].set_ylabel('Cosine Similarity to Unsuccess Avg')
        axes[1, 1].set_title('Magnitude vs Similarity')
        
        plt.tight_layout()
        plt.savefig(similarity_dir / "similarity_analysis_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation matrix of similarity metrics
        similarity_cols = [
            'cosine_sim_to_unsuccess_avg', 'cosine_sim_to_nearest_unsuccess',
            'cosine_sim_to_success_avg', 'euclidean_dist_to_unsuccess_avg',
            'euclidean_dist_to_nearest_unsuccess', 'distance_ratio_nearest_vs_avg',
            'magnitude', 'magnitude_nu', 'magnitude_mu'
        ]
        
        available_cols = [col for col in similarity_cols if col in similarity_df.columns]
        corr_matrix = similarity_df[available_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f', cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix: Similarity Metrics and Attack Vector Magnitudes')
        plt.tight_layout()
        plt.savefig(similarity_dir / "similarity_correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Similarity visualizations saved to {similarity_dir}/")
    
    def create_rainbow_teaming_visualizations(self, output_dir):
        """
        Create visualizations specifically for rainbow teaming context.
        Shows mutation efficiency and evolutionary patterns.
        """
        if not hasattr(self, 'attack_vectors'):
            print("Error: Attack vectors not computed yet. Run analyze() first.")
            return
            
        rainbow_dir = Path(output_dir) / "rainbow_teaming_analysis"
        rainbow_dir.mkdir(parents=True, exist_ok=True)
        
        df = self.df
        
        # Debug: Check DataFrame structure
        print(f"Rainbow Teaming Analysis - DataFrame info:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['magnitude_nu', 'magnitude_mu', 'cosine_sim_nu', 'cosine_sim_mu']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ERROR: Missing required columns: {missing_cols}")
            print("  Available columns:")
            for col in df.columns:
                print(f"    - {col}")
            return
        else:
            print(f"  ✓ All required columns present")
        
        # Check for NaN values in key columns
        for col in required_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"  Warning: {nan_count} NaN values in {col}")
            print(f"  {col}: range {df[col].min():.4f} to {df[col].max():.4f}")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 1. Mutation Efficiency Quadrant Plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Quadrant 1: Mutation Efficiency Matrix
        scatter = axes[0, 0].scatter(df['magnitude_nu'], df['magnitude_mu'], 
                                   c=df['cluster'], cmap='viridis', s=100, alpha=0.7)
        axes[0, 0].axvline(x=df['magnitude_nu'].median(), color='red', linestyle='--', alpha=0.5, label='NU Median')
        axes[0, 0].axhline(y=df['magnitude_mu'].median(), color='red', linestyle='--', alpha=0.5, label='MU Median')
        axes[0, 0].set_xlabel('NU Magnitude (Distance from Nearest Failure)')
        axes[0, 0].set_ylabel('MU Magnitude (Distance from Failure Cluster)')
        axes[0, 0].set_title('Rainbow Teaming: Mutation Efficiency Quadrants')
        
        # Add quadrant labels
        med_nu, med_mu = df['magnitude_nu'].median(), df['magnitude_mu'].median()
        
        # Debug information
        print(f"Debug - Rainbow Teaming Analysis:")
        print(f"  NU median: {med_nu:.4f}, MU median: {med_mu:.4f}")
        print(f"  NU range: {df['magnitude_nu'].min():.4f} to {df['magnitude_nu'].max():.4f}")
        print(f"  MU range: {df['magnitude_mu'].min():.4f} to {df['magnitude_mu'].max():.4f}")
        print(f"  Data shape: {df.shape}")
        
        # Check for NaN values
        nan_nu = df['magnitude_nu'].isna().sum()
        nan_mu = df['magnitude_mu'].isna().sum()
        if nan_nu > 0 or nan_mu > 0:
            print(f"  Warning: Found {nan_nu} NaN NU values and {nan_mu} NaN MU values")
            # Fill NaN values with median
            df['magnitude_nu'].fillna(med_nu, inplace=True)
            df['magnitude_mu'].fillna(med_mu, inplace=True)
            med_nu, med_mu = df['magnitude_nu'].median(), df['magnitude_mu'].median()  # Recalculate
        axes[0, 0].text(df['magnitude_nu'].min(), df['magnitude_mu'].max(), 'Revolutionary\nMutation', 
                       fontsize=10, ha='left', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        axes[0, 0].text(df['magnitude_nu'].max(), df['magnitude_mu'].max(), 'Completely\nNovel Approach', 
                       fontsize=10, ha='right', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.5))
        axes[0, 0].text(df['magnitude_nu'].min(), df['magnitude_mu'].min(), 'Lucky\nBreakthrough', 
                       fontsize=10, ha='left', va='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.5))
        axes[0, 0].text(df['magnitude_nu'].max(), df['magnitude_mu'].min(), 'Local\nOptimization', 
                       fontsize=10, ha='right', va='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.5))
        
        plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
        axes[0, 0].legend()
        
        # Quadrant 2: Evolutionary Pressure vs Escape Route
        axes[0, 1].scatter(df['cosine_sim_nu'], df['cosine_sim_mu'], 
                          c=df['magnitude'], cmap='coolwarm', s=100, alpha=0.7)
        axes[0, 1].set_xlabel('NU Similarity (Similarity to Nearest Failure)')
        axes[0, 1].set_ylabel('MU Similarity (Similarity to Failure Cluster)')
        axes[0, 1].set_title('Evolutionary Pressure vs Systematic Escape')
        axes[0, 1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label='Original Magnitude')
        
        # Quadrant 3: Mutation Strategy Classification
        df['mutation_strategy'] = 'Unknown'  # Default value
        
        # Use <= and >= to ensure all cases are covered (including median values)
        df.loc[(df['magnitude_nu'] <= med_nu) & (df['magnitude_mu'] <= med_mu), 'mutation_strategy'] = 'Lucky Breakthrough'
        df.loc[(df['magnitude_nu'] > med_nu) & (df['magnitude_mu'] <= med_mu), 'mutation_strategy'] = 'Local Optimization'  
        df.loc[(df['magnitude_nu'] <= med_nu) & (df['magnitude_mu'] > med_mu), 'mutation_strategy'] = 'Revolutionary Mutation'
        df.loc[(df['magnitude_nu'] > med_nu) & (df['magnitude_mu'] > med_mu), 'mutation_strategy'] = 'Novel Approach'
        
        # Debug: Check for any remaining unknowns
        unknown_count = (df['mutation_strategy'] == 'Unknown').sum()
        if unknown_count > 0:
            print(f"Warning: {unknown_count} entries remain classified as 'Unknown'")
            print("Median NU:", med_nu, "Median MU:", med_mu)
            unknown_entries = df[df['mutation_strategy'] == 'Unknown']
            print("Unknown entries NU range:", unknown_entries['magnitude_nu'].min(), "to", unknown_entries['magnitude_nu'].max())
            print("Unknown entries MU range:", unknown_entries['magnitude_mu'].min(), "to", unknown_entries['magnitude_mu'].max())
        
        strategy_counts = df['mutation_strategy'].value_counts()
        
        # Handle case where we have unknown entries in the pie chart
        if 'Unknown' in strategy_counts.index:
            colors = ['green', 'blue', 'yellow', 'red', 'gray']  # Gray for unknown
        else:
            colors = ['green', 'blue', 'yellow', 'red']
            
        axes[1, 0].pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%', 
                      colors=colors[:len(strategy_counts)], startangle=90)
        axes[1, 0].set_title('Distribution of Mutation Strategies')
        
        # Quadrant 4: Efficiency Score
        # Create efficiency score: low magnitude = high efficiency
        df['nu_efficiency'] = 1 / (1 + df['magnitude_nu'])  # Higher when NU magnitude is low
        df['mu_efficiency'] = 1 / (1 + df['magnitude_mu'])  # Higher when MU magnitude is low
        df['combined_efficiency'] = (df['nu_efficiency'] + df['mu_efficiency']) / 2
        
        bars = axes[1, 1].bar(range(len(df)), df['combined_efficiency'], 
                             color=plt.cm.viridis(df['cluster'] / df['cluster'].max()))
        axes[1, 1].set_xlabel('Successful Prompt Index')
        axes[1, 1].set_ylabel('Mutation Efficiency Score')
        axes[1, 1].set_title('Rainbow Teaming: Mutation Efficiency Ranking')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(rainbow_dir / "rainbow_teaming_mutation_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Evolutionary Timeline (if prompt_ids available)
        if hasattr(self, 'prompt_ids') and self.prompt_ids:
            self._create_evolutionary_timeline(df, rainbow_dir)
        
        # 3. Failure Mode Escape Analysis
        self._create_failure_escape_analysis(df, rainbow_dir)
        
        # Save strategy classification
        df[['prompt', 'cluster', 'magnitude_nu', 'magnitude_mu', 'mutation_strategy', 
           'nu_efficiency', 'mu_efficiency', 'combined_efficiency']].to_csv(
            rainbow_dir / "rainbow_teaming_mutation_strategies.csv", index=False)
        
        print(f"\nRainbow Teaming Analysis:")
        print(f"Strategy Distribution:")
        for strategy, count in strategy_counts.items():
            print(f"  {strategy}: {count} prompts ({count/len(df)*100:.1f}%)")
        
        print(f"\nTop 3 Most Efficient Mutations:")
        top_efficient = df.nlargest(3, 'combined_efficiency')
        for i, row in top_efficient.iterrows():
            print(f"  {row['mutation_strategy']}: {row['combined_efficiency']:.3f} - {row['prompt'][:80]}...")
            
        return df
    
    def _create_evolutionary_timeline(self, df, rainbow_dir):
        """Create timeline showing evolution of mutation efficiency."""
        # This would require prompt timestamps or generation order
        # For now, create a conceptual timeline based on cluster progression
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # Sort by combined efficiency to show progression
        df_sorted = df.sort_values('combined_efficiency')
        
        plt.plot(range(len(df_sorted)), df_sorted['combined_efficiency'], 'o-', alpha=0.7, linewidth=2)
        plt.fill_between(range(len(df_sorted)), df_sorted['combined_efficiency'], alpha=0.3)
        
        # Color points by strategy
        strategy_colors = {'Lucky Breakthrough': 'green', 'Local Optimization': 'blue', 
                          'Revolutionary Mutation': 'yellow', 'Novel Approach': 'red'}
        for strategy, color in strategy_colors.items():
            mask = df_sorted['mutation_strategy'] == strategy
            if mask.any():
                indices = [i for i, x in enumerate(mask) if x]
                efficiencies = df_sorted[mask]['combined_efficiency'].values
                plt.scatter(indices, efficiencies, c=color, s=100, alpha=0.8, label=strategy, edgecolors='black')
        
        plt.xlabel('Evolution Order (by efficiency)')
        plt.ylabel('Mutation Efficiency Score')
        plt.title('Rainbow Teaming: Evolutionary Timeline of Mutation Strategies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(rainbow_dir / "evolutionary_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_failure_escape_analysis(self, df, rainbow_dir):
        """Analyze patterns in escaping from failure modes."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Escape velocity from nearest failure
        axes[0, 0].hist(df['magnitude_nu'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(df['magnitude_nu'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["magnitude_nu"].mean():.3f}')
        axes[0, 0].set_xlabel('NU Magnitude (Escape from Nearest Failure)')
        axes[0, 0].set_ylabel('Number of Successful Prompts')
        axes[0, 0].set_title('Distribution of Escape Velocity from Nearest Failures')
        axes[0, 0].legend()
        
        # 2. Systematic escape from failure clusters
        axes[0, 1].hist(df['magnitude_mu'], bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].axvline(df['magnitude_mu'].mean(), color='blue', linestyle='--', 
                          label=f'Mean: {df["magnitude_mu"].mean():.3f}')
        axes[0, 1].set_xlabel('MU Magnitude (Escape from Failure Cluster)')
        axes[0, 1].set_ylabel('Number of Successful Prompts')
        axes[0, 1].set_title('Distribution of Systematic Escape from Failure Patterns')
        axes[0, 1].legend()
        
        # 3. Relationship between escape methods
        scatter = axes[1, 0].scatter(df['magnitude_nu'], df['magnitude_mu'], 
                                    c=df['combined_efficiency'], cmap='RdYlGn', s=100, alpha=0.7)
        axes[1, 0].set_xlabel('Individual Escape (NU Magnitude)')
        axes[1, 0].set_ylabel('Systematic Escape (MU Magnitude)')
        axes[1, 0].set_title('Escape Strategy Relationship')
        
        # Add diagonal line
        min_val = min(df['magnitude_nu'].min(), df['magnitude_mu'].min())
        max_val = max(df['magnitude_nu'].max(), df['magnitude_mu'].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal Escape')
        axes[1, 0].legend()
        plt.colorbar(scatter, ax=axes[1, 0], label='Efficiency Score')
        
        # 4. Similarity vs Escape Analysis
        df['similarity_ratio'] = df['cosine_sim_nu'] / df['cosine_sim_mu']
        axes[1, 1].scatter(df['similarity_ratio'], df['combined_efficiency'], 
                          c=df['cluster'], cmap='viridis', s=100, alpha=0.7)
        axes[1, 1].set_xlabel('Similarity Ratio (NU/MU)')
        axes[1, 1].set_ylabel('Combined Efficiency')
        axes[1, 1].set_title('Similarity Pattern vs Mutation Efficiency')
        axes[1, 1].axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Equal Similarity')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(rainbow_dir / "failure_escape_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print escape analysis insights
        print(f"\nFailure Escape Analysis:")
        print(f"Average escape from nearest failure: {df['magnitude_nu'].mean():.3f}")
        print(f"Average escape from failure clusters: {df['magnitude_mu'].mean():.3f}")
        
        # Categorize escape patterns
        near_escape_dominant = (df['magnitude_nu'] < df['magnitude_mu']).sum()
        cluster_escape_dominant = (df['magnitude_mu'] < df['magnitude_nu']).sum()
        
        print(f"Prompts using individual escape strategy: {near_escape_dominant} ({near_escape_dominant/len(df)*100:.1f}%)")
        print(f"Prompts using systematic escape strategy: {cluster_escape_dominant} ({cluster_escape_dominant/len(df)*100:.1f}%)")
        
        # High efficiency insights
        high_efficiency = df[df['combined_efficiency'] > df['combined_efficiency'].quantile(0.75)]
        print(f"\nHigh-efficiency mutations characteristics:")
        print(f"  Average NU magnitude: {high_efficiency['magnitude_nu'].mean():.3f}")
        print(f"  Average MU magnitude: {high_efficiency['magnitude_mu'].mean():.3f}")
        print(f"  Most common strategy: {high_efficiency['mutation_strategy'].mode().iloc[0]}")
    
    def compare_attack_vector_types(self, output_dir):
        """
        Create detailed comparison analysis of the three attack vector types.
        """
        if not hasattr(self, 'attack_vectors'):
            print("Error: Attack vectors not computed yet. Run analyze() first.")
            return
            
        comparison_dir = Path(output_dir) / "attack_vector_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'prompt': self.df['prompt'],
            'cluster': self.df['cluster'],
            'magnitude_orig': self.df['magnitude'],
            'magnitude_nu': self.df['magnitude_nu'], 
            'magnitude_mu': self.df['magnitude_mu']
        })
        
        # Compute ratios
        comparison_df['ratio_nu_orig'] = comparison_df['magnitude_nu'] / comparison_df['magnitude_orig']
        comparison_df['ratio_mu_orig'] = comparison_df['magnitude_mu'] / comparison_df['magnitude_orig']
        comparison_df['ratio_nu_mu'] = comparison_df['magnitude_nu'] / comparison_df['magnitude_mu']
        
        # Identify interesting cases
        comparison_df['nu_much_smaller'] = comparison_df['ratio_nu_orig'] < 0.5  # NU vector much smaller
        comparison_df['mu_much_smaller'] = comparison_df['ratio_mu_orig'] < 0.5  # MU vector much smaller
        comparison_df['nu_much_larger'] = comparison_df['ratio_nu_orig'] > 2.0   # NU vector much larger
        comparison_df['mu_much_larger'] = comparison_df['ratio_mu_orig'] > 2.0   # MU vector much larger
        
        # Save detailed comparison
        comparison_df.to_csv(comparison_dir / "attack_vector_comparison.csv", index=False)
        
        # Print interesting findings
        print("\nAttack Vector Comparison Analysis:")
        print(f"Cases where NU magnitude << Original magnitude: {comparison_df['nu_much_smaller'].sum()}")
        print(f"Cases where MU magnitude << Original magnitude: {comparison_df['mu_much_smaller'].sum()}")
        print(f"Cases where NU magnitude >> Original magnitude: {comparison_df['nu_much_larger'].sum()}")
        print(f"Cases where MU magnitude >> Original magnitude: {comparison_df['mu_much_larger'].sum()}")
        
        # Show examples of extreme cases
        if comparison_df['nu_much_smaller'].any():
            print("\nExamples where NU vector is much smaller (easier to reach from nearest unsuccessful):")
            examples = comparison_df[comparison_df['nu_much_smaller']].head(3)
            for _, row in examples.iterrows():
                print(f"  Orig: {row['magnitude_orig']:.3f}, NU: {row['magnitude_nu']:.3f} | {row['prompt'][:100]}...")
                
        if comparison_df['mu_much_smaller'].any():
            print("\nExamples where MU vector is much smaller (easier to reach from cluster mean):")
            examples = comparison_df[comparison_df['mu_much_smaller']].head(3) 
            for _, row in examples.iterrows():
                print(f"  Orig: {row['magnitude_orig']:.3f}, MU: {row['magnitude_mu']:.3f} | {row['prompt'][:100]}...")
        
        return comparison_df
    
    def print_metrics_explanation(self):
        """Print a comprehensive explanation of all computed metrics."""
        print("\n" + "="*80)
        print("COMPREHENSIVE ATTACK VECTOR METRICS EXPLANATION")
        print("="*80)
        
        print("\n🎯 ATTACK VECTOR TYPES:")
        print("  • Original:  successful_embedding - mean(all_unsuccessful_embeddings)")
        print("    → General semantic direction from average failure to success")
        print("  • NU:        successful_embedding - nearest_unsuccessful_embedding")
        print("    → Minimal semantic shift from closest failure to success")
        print("  • MU:        successful_embedding - mean(closest_unsuccessful_cluster)")
        print("    → Systematic departure from similar failure patterns")
        print("  • SP:        successful_embedding - seed_embedding")
        print("    → Evolutionary trajectory from initial seed to successful adversarial prompt")
        
        print("\n📏 MAGNITUDE METRICS:")
        print("  • High magnitude = Large semantic shift required")
        print("  • Low magnitude = Small semantic shift required")
        print("  • NU magnitude < Original → Success was very close to a specific failure")
        print("  • MU magnitude < Original → Success systematically differs from failure cluster")
        print("  • SP magnitude → Total evolutionary distance from seed to successful prompt")
        
        print("\n🔄 SIMILARITY SCORES:")
        print("  • NU Similarity: cosine_similarity(successful, nearest_unsuccessful)")
        print("    → How similar successful prompt is to its nearest failure")
        print("  • MU Similarity: cosine_similarity(successful, closest_cluster_centroid)")
        print("    → How similar successful prompt is to failure cluster center")
        print("  • SP Similarity: cosine_similarity(successful, seed)")
        print("    → How similar successful prompt is to original seed prompt")
        print("  • High similarity (>0.8) = Very similar to unsuccessful attempts")
        print("  • Low similarity (<0.3) = Very different from unsuccessful attempts")
        
        print("\n🌈 RAINBOW TEAMING CONTEXT:")
        print("  • NU Magnitude: Minimal mutation needed from nearest failed attempt")
        print("    → Low NU = 'Near miss' - tiny mutation flipped failure to success")
        print("    → High NU = Required major change from closest failure")
        print("  • MU Magnitude: Systematic departure from recurring failure patterns")
        print("    → Low MU = Escaped common failure trap with small systematic change")
        print("    → High MU = Completely avoided clustered failure modes")
        print("  • SP Magnitude: Total evolutionary trajectory from seed to success")
        print("    → Low SP = Successful prompt stayed close to original seed intent")
        print("    → High SP = Successful prompt significantly transformed from seed")
        
        print("\n🎯 MUTATION STRATEGY CLASSIFICATION:")
        print("  • Lucky Breakthrough: Low NU + Low MU (close to individual AND cluster failures)")
        print("  • Local Optimization: High NU + Low MU (far from nearest, close to cluster pattern)")
        print("  • Revolutionary Mutation: Low NU + High MU (close to nearest, far from cluster)")
        print("  • Novel Approach: High NU + High MU (far from both individual and cluster failures)")
        print("\n🧬 EVOLUTIONARY TRAJECTORY ANALYSIS (SP):")
        print("  • Conservative Evolution: Low SP magnitude + High SP similarity")
        print("    → Successful prompt maintained original seed characteristics")
        print("  • Adaptive Transformation: Medium SP magnitude + Medium SP similarity")
        print("    → Successful prompt evolved from seed while preserving core intent")
        print("  • Radical Mutation: High SP magnitude + Low SP similarity")
        print("    → Successful prompt dramatically transformed from original seed")
        
        print("\n💡 INTERPRETATION GUIDELINES:")
        print("  • Low NU magnitude + High NU similarity = 'Near miss' - minimal change needed")
        print("  • Low MU magnitude + High MU similarity = Systematic pattern deviation")
        print("  • Low SP magnitude + High SP similarity = Conservative evolution from seed")
        print("  • High magnitude + Low similarity = Completely different approach")
        print("  • Correlation between metrics = Consistency in attack patterns")
        print("  • SP vs NU/MU comparison = Evolutionary trajectory vs optimization dynamics")
        print("="*80)
    
    def load_lineage_fields(self):
        """Load prompt_id, parent_id, seed_id from log if present."""
        # Try multiple possible key names for lineage data
        self.prompt_ids = self.data.get('all_prompt_ids', self.data.get('prompt_ids', {}))
        self.parent_ids = self.data.get('all_parent_ids', self.data.get('parent_ids', {}))
        self.seed_ids = self.data.get('all_seed_ids', self.data.get('seed_ids', {}))
        
        print(f"Loaded lineage fields:")
        print(f"  Prompt IDs: {len(self.prompt_ids)} keys")
        print(f"  Parent IDs: {len(self.parent_ids)} keys") 
        print(f"  Seed IDs: {len(self.seed_ids)} keys")
        
        # If no lineage data found, return False
        return bool(self.prompt_ids or self.parent_ids or self.seed_ids)

    def lineage_dataframe(self):
        """Return a DataFrame with prompt, prompt_id, parent_id, seed_id, cluster, magnitude, etc."""
        # Only works after analyze() has run and df exists
        if not hasattr(self, 'df'):
            print("Error: Main DataFrame not available. Run analyze() first.")
            return pd.DataFrame()
        
        # Check if lineage data is available
        if not (self.prompt_ids or self.parent_ids or self.seed_ids):
            print("No lineage data available. Creating basic lineage DataFrame.")
            # Create basic lineage with just successful prompts
            lineage_data = []
            for idx, row in self.df.iterrows():
                lineage_data.append({
                    'prompt': row['prompt'],
                    'prompt_id': f"prompt_{idx}",
                    'parent_id': None,
                    'seed_id': f"seed_{idx % 5}",  # Distribute across 5 mock seeds
                    'cluster': row['cluster'],
                    'magnitude': row['magnitude']
                })
            return pd.DataFrame(lineage_data)
        
        # Create a comprehensive mapping of all prompts and their lineage data from the logs
        print("Building comprehensive lineage mapping...")
        all_prompt_to_data = {}  # Maps prompt text to its lineage data
        
        # Process all prompts in the comprehensive log (not just successful ones)
        for k, prompts in self.data.get('all_prompts', {}).items():
            if not isinstance(prompts, list):
                continue
                
            prompt_ids_for_key = self.prompt_ids.get(k, [])
            parent_ids_for_key = self.parent_ids.get(k, [])
            seed_ids_for_key = self.seed_ids.get(k, [])
            
            for i, prompt in enumerate(prompts):
                prompt_id = prompt_ids_for_key[i] if i < len(prompt_ids_for_key) else f"unknown_{k}_{i}"
                parent_id = parent_ids_for_key[i] if i < len(parent_ids_for_key) else None
                seed_id = seed_ids_for_key[i] if i < len(seed_ids_for_key) else f"unknown_seed_{k}_{i}"
                
                all_prompt_to_data[prompt] = {
                    'prompt_id': prompt_id,
                    'parent_id': parent_id,
                    'seed_id': seed_id
                }
        
        print(f"  Found lineage data for {len(all_prompt_to_data)} total prompts")
        
        # Now build the lineage DataFrame for successful prompts only
        prompt_list = []
        prompt_id_list = []
        parent_id_list = []
        seed_id_list = []
        cluster_list = []
        magnitude_list = []
        
        for idx, row in self.df.iterrows():
            prompt = row['prompt']
            prompt_list.append(prompt)
            cluster_list.append(row['cluster'])
            magnitude_list.append(row['magnitude'])
            
            # Get lineage data from our comprehensive mapping
            if prompt in all_prompt_to_data:
                lineage_data = all_prompt_to_data[prompt]
                prompt_id_list.append(lineage_data['prompt_id'])
                parent_id_list.append(lineage_data['parent_id'])
                seed_id_list.append(lineage_data['seed_id'])
            else:
                # Fallback for prompts not found in lineage data
                prompt_id_list.append(f"fallback_{idx}")
                parent_id_list.append(None)
                seed_id_list.append(f"fallback_seed_{idx % 3}")
        
        lineage_df = pd.DataFrame({
            'prompt': prompt_list,
            'prompt_id': prompt_id_list,
            'parent_id': parent_id_list,
            'seed_id': seed_id_list,
            'cluster': cluster_list,
            'magnitude': magnitude_list
        })
        
        # Clean up empty string parent_ids to None for proper tree visualization
        lineage_df['parent_id'] = lineage_df['parent_id'].replace('', None)
        lineage_df['parent_id'] = lineage_df['parent_id'].replace(pd.NA, None)
        
        # Validate parent references - check for orphaned children
        all_prompt_ids = set(lineage_df['prompt_id'].tolist())
        orphaned_children = lineage_df[
            (lineage_df['parent_id'].notna()) & 
            (~lineage_df['parent_id'].isin(all_prompt_ids))
        ]
        
        print(f"Created lineage DataFrame: {len(lineage_df)} prompts")
        print(f"  Unique seeds: {lineage_df['seed_id'].nunique()}")
        print(f"  Prompts with parents: {lineage_df['parent_id'].notna().sum()}")
        print(f"  Orphaned children (missing parents): {len(orphaned_children)}")
        
        if len(orphaned_children) > 0:
            print("  Missing parent IDs:", sorted(orphaned_children['parent_id'].unique()))
        
        return lineage_df

    def _format_seed_label(self, seed_id):
        """Format seed ID for display in tree visualizations."""
        if isinstance(seed_id, str):
            if seed_id.startswith('seed_'):
                return seed_id.replace('seed_', 'S')
            elif seed_id.startswith('fallback_seed_'):
                return 'F' + seed_id.replace('fallback_seed_', '')
            elif seed_id == 'missing':
                return 'MISS'
            else:
                return seed_id[:4]  # Truncate to 4 chars
        else:
            return str(seed_id)[:4]

    def analyze_mutations_from_replaced_parents(self, lineage_df):
        """Analyze and report on successful prompts that were mutated from parents not in the final archive.
        
        Note: These are NOT mutations from unsuccessful prompts. The RainbowPlus algorithm only
        selects parents from successful prompts. What we're seeing here are mutations from 
        successful parents that were later replaced in their archive cells by better prompts.
        """
        if lineage_df.empty:
            print("Warning: Empty lineage DataFrame, skipping replaced parent analysis")
            return {}
        
        # Create a mapping of all prompt_ids to their data for quick lookup
        prompt_data_map = {row['prompt_id']: row for _, row in lineage_df.iterrows()}
        
        # Find all children of parents not in final archive (but were successful at time of selection)
        mutations_from_replaced = {}
        replaced_parents = set()
        
        for _, row in lineage_df.iterrows():
            pid = row['prompt_id']
            parid = row['parent_id']
            
            # Handle empty strings and None values
            if parid == "" or parid is None or pd.isna(parid):
                continue
            
            # If parent doesn't exist in final successful dataset, it was replaced in archive
            if parid not in prompt_data_map:
                replaced_parents.add(parid)
                mutations_from_replaced[pid] = {
                    'child_id': pid,
                    'replaced_parent_id': parid,
                    'child_prompt': row['prompt'],
                    'child_seed': row['seed_id'],
                    'child_cluster': row['cluster'],
                    'child_magnitude': row['magnitude']
                }
        
        print(f"\n🔄 MUTATIONS FROM REPLACED PARENTS ANALYSIS:")
        print(f"{'='*60}")
        print(f"📌 NOTE: These parents were SUCCESSFUL when selected, but later")
        print(f"   replaced in their archive cells by better prompts.")
        print(f"   RainbowPlus NEVER mutates from unsuccessful prompts!")
        print(f"{'='*60}")
        print(f"Total replaced parent nodes: {len(replaced_parents)}")
        print(f"Successful mutations from replaced parents: {len(mutations_from_replaced)}")
        
        if mutations_from_replaced:
            # Analyze by seed
            seed_analysis = {}
            for mutation in mutations_from_replaced.values():
                seed = mutation['child_seed']
                if seed not in seed_analysis:
                    seed_analysis[seed] = []
                seed_analysis[seed].append(mutation)
            
            print(f"\n📊 Distribution by seed:")
            for seed, mutations in seed_analysis.items():
                print(f"  {seed}: {len(mutations)} mutations from replaced parents")
            
            # Analyze by cluster
            cluster_analysis = {}
            for mutation in mutations_from_replaced.values():
                cluster = mutation['child_cluster']
                if cluster not in cluster_analysis:
                    cluster_analysis[cluster] = []
                cluster_analysis[cluster].append(mutation)
            
            print(f"\n🎯 Distribution by cluster:")
            for cluster, mutations in cluster_analysis.items():
                avg_magnitude = np.mean([m['child_magnitude'] for m in mutations])
                print(f"  Cluster {cluster}: {len(mutations)} mutations (avg magnitude: {avg_magnitude:.3f})")
            
            # Show top examples
            print(f"\n💡 Examples of successful mutations from replaced parents:")
            for i, (child_id, mutation) in enumerate(list(mutations_from_replaced.items())[:5]):
                prompt_preview = mutation['child_prompt'][:80] + ('...' if len(mutation['child_prompt']) > 80 else '')
                print(f"  {i+1}. {child_id} (from 🔄{mutation['replaced_parent_id'][:10]} - was successful but replaced)")
                print(f"     Seed: {mutation['child_seed']}, Cluster: {mutation['child_cluster']}, Magnitude: {mutation['child_magnitude']:.3f}")
                print(f"     Prompt: {prompt_preview}")
                print()
            
            # Statistical insights
            magnitudes = [m['child_magnitude'] for m in mutations_from_replaced.values()]
            total_prompts = len(lineage_df)
            mutation_rate = len(mutations_from_replaced) / total_prompts * 100
            
            print(f"📈 Statistical insights:")
            print(f"  Archive replacement rate: {mutation_rate:.1f}% of successful prompts have replaced parents")
            print(f"  Average magnitude of mutations from replaced parents: {np.mean(magnitudes):.3f}")
            print(f"  Magnitude range: {np.min(magnitudes):.3f} to {np.max(magnitudes):.3f}")
            print(f"  Standard deviation: {np.std(magnitudes):.3f}")
            
            # Compare with overall statistics
            overall_magnitudes = lineage_df['magnitude'].values
            avg_overall = np.mean(overall_magnitudes)
            avg_mutations = np.mean(magnitudes)
            
            print(f"\n🔍 Comparison with all prompts:")
            print(f"  Average magnitude (all prompts): {avg_overall:.3f}")
            print(f"  Average magnitude (mutations from replaced): {avg_mutations:.3f}")
            print(f"  Difference: {avg_mutations - avg_overall:.3f} ({'higher' if avg_mutations > avg_overall else 'lower'})")
            
            print(f"\n🏆 Archive Dynamics Insights:")
            print(f"  This shows the evolutionary pressure in RainbowPlus archives!")
            print(f"  Parents were good enough to be selected, but later outcompeted.")
            print(f"  Their children survived while they were replaced - natural selection!")
            
        else:
            print("  No mutations from replaced parents found.")
        
        print("="*60)
        
        return mutations_from_replaced

    def visualize_prompt_tree(self, lineage_df, output_dir):
        """Visualize the prompt tree for each seed using networkx and matplotlib."""
        if lineage_df.empty:
            print("Warning: Empty lineage DataFrame, skipping tree visualization")
            return
        
        tree_dir = Path(output_dir) / "tree_analysis"
        tree_dir.mkdir(parents=True, exist_ok=True)
        
        import matplotlib.pyplot as plt
        import networkx as nx
        
        seeds = lineage_df['seed_id'].unique()
        print(f"Creating tree visualizations for {len(seeds)} seeds...")
        
        # Create a mapping of all prompt_ids to their data for quick lookup
        prompt_data_map = {row['prompt_id']: row for _, row in lineage_df.iterrows()}
        
        for seed in seeds:
            sub = lineage_df[lineage_df['seed_id'] == seed]
            if len(sub) == 0:
                continue
                
            G = nx.DiGraph()
            
            # First pass: Add all nodes from current seed
            for _, row in sub.iterrows():
                pid = row['prompt_id']
                label = row['prompt'][:40] + ('...' if len(row['prompt']) > 40 else '')
                G.add_node(pid, label=label, prompt=row['prompt'], seed=row['seed_id'], is_external=False)
            
            # Second pass: Add all possible parent-child relationships
            for _, row in sub.iterrows():
                pid = row['prompt_id']
                parid = row['parent_id']
                
                # Handle empty strings and None values
                if parid == "" or parid is None or pd.isna(parid) or parid == pid:
                    continue
                
                # If parent exists in the successful prompts dataset, add it and create edge
                if parid in prompt_data_map:
                    parent_row = prompt_data_map[parid]
                    
                    # Add parent node if not already in graph
                    if parid not in G.nodes():
                        parent_label = parent_row['prompt'][:40] + ('...' if len(parent_row['prompt']) > 40 else '')
                        is_external = parent_row['seed_id'] != seed
                        G.add_node(parid, label=parent_label, prompt=parent_row['prompt'], 
                                  seed=parent_row['seed_id'], is_external=is_external)
                    
                    # Add edge
                    G.add_edge(parid, pid)
                else:
                    # Parent doesn't exist in successful prompts dataset (was unsuccessful)
                    # Create a placeholder node to show the relationship
                    if parid not in G.nodes():
                        G.add_node(parid, label=f"{parid[:15]}...\n(unsuccessful)", 
                                  prompt="[Unsuccessful parent prompt]", 
                                  seed="unknown", is_external=True, is_unsuccessful=True)
                    
                    # Add edge to show the lineage
                    G.add_edge(parid, pid)
            
            # Third pass: Add any remaining parent relationships recursively
            # This catches multi-level relationships where parents of parents exist
            added_new_nodes = True
            while added_new_nodes:
                added_new_nodes = False
                current_nodes = list(G.nodes())
                
                for node in current_nodes:
                    if node in prompt_data_map:
                        row = prompt_data_map[node]
                        parid = row['parent_id']
                        
                        # Handle empty strings and None values
                        if parid == "" or parid is None or pd.isna(parid) or parid == node:
                            continue
                        
                        # If parent exists in dataset and not yet in graph
                        if parid in prompt_data_map and parid not in G.nodes():
                            parent_row = prompt_data_map[parid]
                            parent_label = parent_row['prompt'][:40] + ('...' if len(parent_row['prompt']) > 40 else '')
                            is_external = parent_row['seed_id'] != seed
                            G.add_node(parid, label=parent_label, prompt=parent_row['prompt'], 
                                      seed=parent_row['seed_id'], is_external=is_external)
                            G.add_edge(parid, node)
                            added_new_nodes = True
            # Only create visualization if we have nodes
            if len(G.nodes()) > 0:
                plt.figure(figsize=(14, 10))
                
                # Use hierarchical layout if we have edges, otherwise spring layout
                if len(G.edges()) > 0:
                    try:
                        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
                    except:
                        # Fallback to spring layout if graphviz not available
                        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
                else:
                    # For isolated nodes, use a simple grid layout
                    nodes = list(G.nodes())
                    pos = {node: (i % 5, i // 5) for i, node in enumerate(nodes)}
                
                # Different colors and sizes for different node types
                node_colors = []
                node_sizes = []
                for node in G.nodes():
                    node_data = G.nodes[node]
                    if node_data.get('is_unsuccessful', False):
                        node_colors.append('lightgray')  # Unsuccessful parents in gray
                        node_sizes.append(150)  # Smaller for unsuccessful
                    elif node_data.get('is_external', False):
                        node_colors.append('lightcoral')  # External successful nodes in light red
                        node_sizes.append(200)  # Medium for external
                    else:
                        node_colors.append('lightblue')   # Internal nodes in light blue
                        node_sizes.append(400)  # Larger for internal
                
                # Draw the graph
                nx.draw(G, pos, with_labels=False, node_size=node_sizes, node_color=node_colors, 
                       arrows=True, arrowsize=20, edge_color='gray', alpha=0.8)
                
                # Add node labels with different formatting for different node types
                labels = {}
                for node in G.nodes():
                    node_data = G.nodes[node]
                    if node_data.get('is_unsuccessful', False):
                        # Unsuccessful parent nodes
                        labels[node] = f"{node[:8]}...\n❌FAILED❌"
                    elif node_data.get('is_external', False):
                        # External successful nodes - show seed info
                        seed_short = node_data.get('seed', 'unknown')[-3:]  # Last 3 chars
                        labels[node] = f"{node[:8]}...\n🌈{seed_short}🌈"
                    else:
                        # Internal nodes - just the ID
                        labels[node] = f"{node[:10]}..." if len(str(node)) > 10 else str(node)
                
                # Add special annotation for nodes that are children of unsuccessful parents
                children_of_failed = {}
                for edge in G.edges():
                    parent, child = edge
                    if G.nodes[parent].get('is_unsuccessful', False):
                        children_of_failed[child] = parent
                
                # Update labels for children of failed nodes
                for child, failed_parent in children_of_failed.items():
                    if child in labels:
                        original_label = labels[child]
                        labels[child] = f"🔄{original_label}\n(from ❌{failed_parent[:6]})"
                
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)
                
                # Count different node types for title
                internal_nodes = len([n for n in G.nodes() if not G.nodes[n].get('is_external', False) and not G.nodes[n].get('is_unsuccessful', False)])
                external_nodes = len([n for n in G.nodes() if G.nodes[n].get('is_external', False) and not G.nodes[n].get('is_unsuccessful', False)])
                unsuccessful_nodes = len([n for n in G.nodes() if G.nodes[n].get('is_unsuccessful', False)])
                connected_nodes = len([n for n in G.nodes() if G.degree(n) > 0])
                isolated_nodes = len(G.nodes()) - connected_nodes
                
                plt.title(f"Prompt Tree for Seed {seed}\n"
                         f"Internal: {internal_nodes}, External: {external_nodes}, Failed: {unsuccessful_nodes}\n"
                         f"Connected: {connected_nodes}, Isolated: {isolated_nodes}\n"
                         f"🔄 = Mutated from failed parent | ❌ = Failed parent | 🌈 = Other seed", 
                         fontsize=11)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(tree_dir / f"prompt_tree_seed_{seed}.png", dpi=200, bbox_inches='tight')
                plt.close()
                
                print(f"  Seed {seed}: {len(G.nodes())} nodes, {len(G.edges())} edges, {isolated_nodes} isolated")
                
        print(f"Tree visualizations saved to {tree_dir}/")
        
        # Also create a global tree showing all relationships
        self.visualize_global_prompt_tree(lineage_df, output_dir)

    def visualize_global_prompt_tree(self, lineage_df, output_dir):
        """Visualize a global tree showing all prompt relationships across seeds."""
        if lineage_df.empty:
            print("Warning: Empty lineage DataFrame, skipping global tree visualization")
            return
        
        tree_dir = Path(output_dir) / "tree_analysis"
        tree_dir.mkdir(parents=True, exist_ok=True)
        
        import matplotlib.pyplot as plt
        import networkx as nx
        
        print("Creating global tree visualization...")
        
        G = nx.DiGraph()
        
        # Create a mapping of all prompt_ids to their data for quick lookup
        prompt_data_map = {row['prompt_id']: row for _, row in lineage_df.iterrows()}
        
        # Add all successful nodes
        for _, row in lineage_df.iterrows():
            pid = row['prompt_id']
            seed = row['seed_id']
            label = row['prompt'][:30] + ('...' if len(row['prompt']) > 30 else '')
            G.add_node(pid, label=label, seed=seed, prompt=row['prompt'], is_unsuccessful=False)
        
        # Add all edges and missing parent nodes
        missing_parents = set()
        children_of_missing = {}
        
        for _, row in lineage_df.iterrows():
            pid = row['prompt_id']
            parid = row['parent_id']
            
            # Handle empty strings and None values
            if parid == "" or parid is None or pd.isna(parid):
                continue
                
            # Skip self-references
            if parid == pid:
                continue
            
            # If parent exists in successful dataset, add edge
            if parid in prompt_data_map:
                G.add_edge(parid, pid)
            else:
                # Parent doesn't exist in successful prompts - it was either unsuccessful or replaced
                missing_parents.add(parid)
                children_of_missing[pid] = parid
                
                # Add missing parent node if not already present
                if parid not in G.nodes():
                    # Determine if this was likely a seed or mutation based on naming
                    if parid.startswith('seed_'):
                        node_label = f"{parid}\n🌱 MISSING SEED"
                        node_type = "missing_seed"
                    else:
                        node_label = f"{parid[:15]}...\n❌ MISSING PARENT"
                        node_type = "missing_parent"
                    
                    G.add_node(parid, label=node_label, seed="missing", 
                              prompt="[Missing from successful set]", is_unsuccessful=True,
                              node_type=node_type)
                
                # Add edge from missing parent to successful child
                G.add_edge(parid, pid)
                continue
            
            # If parent exists in successful dataset, add edge
            if parid in prompt_data_map:
                G.add_edge(parid, pid)
            else:
                # Parent doesn't exist - it was unsuccessful
                unsuccessful_parents.add(parid)
                children_of_failed[pid] = parid
                
                # Add unsuccessful parent node if not already present
                if parid not in G.nodes():
                    G.add_node(parid, label=f"{parid[:15]}...\n❌FAILED❌", 
                              seed="failed", prompt="[Unsuccessful parent]", is_unsuccessful=True)
                
                # Add edge from unsuccessful parent to successful child
                G.add_edge(parid, pid)
        
        if len(G.nodes()) > 0:
            plt.figure(figsize=(16, 12))
            
            # Use spring layout for global view
            if len(G.edges()) > 0:
                try:
                    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
                except:
                    # Fallback to spring layout
                    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
            else:
                # For isolated nodes, use a simple grid layout
                nodes = list(G.nodes())
                pos = {node: (i % 10, i // 10) for i, node in enumerate(nodes)}
            
            # Color nodes: different colors for seeds + gray for unsuccessful
            successful_seeds = [G.nodes[node]['seed'] for node in G.nodes() if not G.nodes[node]['is_unsuccessful']]
            unique_seeds = list(set(successful_seeds))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_seeds)))
            seed_color_map = {seed: colors[i] for i, seed in enumerate(unique_seeds)}
            seed_color_map['failed'] = 'lightgray'  # Gray for unsuccessful
            
            node_colors = [seed_color_map[G.nodes[node]['seed']] for node in G.nodes()]
            
            # Different sizes for different node types
            node_sizes = []
            for node in G.nodes():
                if G.nodes[node]['is_unsuccessful']:
                    node_sizes.append(150)  # Smaller for unsuccessful
                else:
                    node_sizes.append(200)  # Normal for successful
            
            # Draw the graph
            nx.draw(G, pos, with_labels=False, node_size=node_sizes, node_color=node_colors, 
                   arrows=True, arrowsize=15, edge_color='gray', alpha=0.7)
            
            # Add enhanced labels showing mutations from missing parents
            labels = {}
            for node in G.nodes():
                node_data = G.nodes[node]
                if node_data['is_unsuccessful']:
                    node_type = node_data.get('node_type', 'missing_parent')
                    if node_type == 'missing_seed':
                        labels[node] = f"🌱{node[:8]}...\nMISSING SEED"
                    else:
                        labels[node] = f"❌{node[:8]}...\nMISSING PARENT"
                elif node in children_of_missing:
                    # This node is a successful mutation from a missing parent
                    missing_parent = children_of_missing[node]
                    seed_short = self._format_seed_label(node_data['seed'])
                    labels[node] = f"🔄{node[:6]}...\n({seed_short})\nfrom ❌{missing_parent[:6]}"
                else:
                    # Regular successful node
                    seed_short = self._format_seed_label(node_data['seed'])
                    labels[node] = f"{node[:8]}...\n({seed_short})"
            
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)
            
            # Create legend for seeds
            legend_elements = []
            for seed in unique_seeds:
                seed_label = self._format_seed_label(seed) if hasattr(self, '_format_seed_label') else seed
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=seed_color_map[seed], markersize=8, 
                                                label=f'Seed {seed_label}'))
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor='lightgray', markersize=8, 
                                            label='Missing Parent'))
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            
            # Add connection info to title
            connected_nodes = len([n for n in G.nodes() if G.degree(n) > 0])
            isolated_nodes = len(G.nodes()) - connected_nodes
            total_edges = len(G.edges())
            mutations_from_missing = len(children_of_missing)
            
            plt.title(f"Global Prompt Tree (All Seeds & Missing Parents)\n"
                     f"Nodes: {len(G.nodes())}, Edges: {total_edges}, "
                     f"Connected: {connected_nodes}, Isolated: {isolated_nodes}\n"
                     f"🔄 Mutations from missing parents: {mutations_from_missing}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(tree_dir / "prompt_tree_global.png", dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"Global tree visualization saved with {mutations_from_missing} mutations from missing parents")
            print(f"  Total missing parent nodes: {len(missing_parents)}")
            
            # Print some statistics about mutations from missing parents
            if children_of_missing:
                print("  Examples of successful mutations from missing parents:")
                for child, missing_parent in list(children_of_missing.items())[:3]:
                    child_data = prompt_data_map.get(child, {})
                    child_prompt = child_data.get('prompt', 'Unknown')[:60]
                    print(f"    {child} (from ❌{missing_parent}): {child_prompt}...")
        else:
            print("No nodes found for global tree visualization")

    def visualize_tree_stats(self, lineage_df, seed_depths, branch_factors, output_dir):
        """Visualize distributions of tree depth and branch factor using lineage data directly."""
        if lineage_df.empty:
            print("Warning: Empty lineage DataFrame, skipping tree stats visualization")
            return
        
        # Filter out any rows with problematic seed_id values
        lineage_df = lineage_df[lineage_df['seed_id'] != 'missing'].copy()
        lineage_df = lineage_df[lineage_df['seed_id'].notna()].copy()
        
        if lineage_df.empty:
            print("Warning: No valid lineage data after filtering, skipping tree stats visualization")
            return
        
        tree_dir = Path(output_dir) / "tree_analysis"
        tree_dir.mkdir(parents=True, exist_ok=True)
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Extract numeric part from seed names for proper ordering (e.g., "seed_1" -> 1)
        def extract_seed_number(seed_name):
            if isinstance(seed_name, str) and seed_name.startswith('seed_'):
                try:
                    return int(seed_name.split('_')[1])
                except (IndexError, ValueError):
                    return float('inf')  # Put invalid seeds at the end
            elif isinstance(seed_name, str) and seed_name.startswith('fallback_seed_'):
                try:
                    return 1000 + int(seed_name.split('_')[2])  # Put fallback seeds after regular seeds
                except (IndexError, ValueError):
                    return float('inf')
            else:
                return float('inf')  # Put non-standard seeds at the end
        
        # 1. Seed distribution (prompts per seed) - Use lineage data directly
        seed_counts = lineage_df['seed_id'].value_counts()
        seed_names = seed_counts.index.tolist()
        
        # Sort seeds by their numeric value
        sorted_seeds = sorted(seed_names, key=extract_seed_number)
        sorted_counts = [seed_counts[seed] for seed in sorted_seeds]
        
        # Create readable labels (remove "seed_" prefix for cleaner display)
        seed_labels = []
        for seed in sorted_seeds:
            if seed.startswith('seed_'):
                seed_labels.append(seed.replace('seed_', ''))
            elif seed.startswith('fallback_seed_'):
                seed_labels.append('F' + seed.replace('fallback_seed_', ''))
            else:
                seed_labels.append(seed[:6])  # Truncate long seed names
        
        axes[0, 0].bar(range(len(sorted_counts)), sorted_counts, alpha=0.7)
        axes[0, 0].set_title('Successful Prompts per Seed')
        axes[0, 0].set_xlabel('Seed ID')
        axes[0, 0].set_ylabel('Number of Prompts')
        axes[0, 0].set_xticks(range(len(seed_labels)))
        axes[0, 0].set_xticklabels(seed_labels, rotation=45)
        
        # Add count labels on top of bars
        for i, count in enumerate(sorted_counts):
            axes[0, 0].text(i, count + 0.1, str(count), ha='center', va='bottom')
        
        # 2. Tree depth distribution (if available)
        if seed_depths and len(seed_depths) > 0:
            depth_values = list(seed_depths.values())
            axes[0, 1].hist(depth_values, bins=max(1, len(set(depth_values))), 
                           edgecolor='black', alpha=0.7)
            axes[0, 1].set_title('Distribution of Max Tree Depth per Seed')
            axes[0, 1].set_xlabel('Max Depth')
            axes[0, 1].set_ylabel('Number of Seeds')
        else:
            axes[0, 1].text(0.5, 0.5, 'No tree depth data available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Tree Depth Distribution (No Data)')
        
        # 3. Branch factor distribution (if available)
        if branch_factors and len(branch_factors) > 0:
            branch_values = list(branch_factors.values())
            axes[1, 0].hist(branch_values, bins=max(1, len(set(branch_values))), 
                           edgecolor='black', alpha=0.7)
            axes[1, 0].set_title('Distribution of Branch Factors')
            axes[1, 0].set_xlabel('Branch Factor')
            axes[1, 0].set_ylabel('Number of Prompts')
        else:
            axes[1, 0].text(0.5, 0.5, 'No branch factor data available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Branch Factor Distribution (No Data)')
        
        # 4. Show seed distribution again but with different metrics (from lineage data only)
        if lineage_df['seed_id'].nunique() > 1:
            # Use the same seed data but show different perspective - maybe mutations per seed
            mutation_counts_per_seed = {}
            for seed in sorted_seeds:
                seed_data = lineage_df[lineage_df['seed_id'] == seed]
                # Count how many are mutations (have parent_id) vs seeds (no parent_id)
                mutations = seed_data[seed_data['parent_id'].notna()]
                mutation_counts_per_seed[seed] = len(mutations)
            
            mutation_counts = [mutation_counts_per_seed.get(seed, 0) for seed in sorted_seeds]
            
            axes[1, 1].bar(range(len(mutation_counts)), mutation_counts, alpha=0.7)
            axes[1, 1].set_title('Mutations per Seed')
            axes[1, 1].set_xlabel('Seed ID') 
            axes[1, 1].set_ylabel('Number of Mutations')
            axes[1, 1].set_xticks(range(len(seed_labels)))
            axes[1, 1].set_xticklabels(seed_labels, rotation=45)
            
            # Add count labels on bars
            for i, count in enumerate(mutation_counts):
                axes[1, 1].text(i, count + 0.1, str(count), ha='center', va='bottom')
                
        else:
            axes[1, 1].text(0.5, 0.5, 'Single seed data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Seed Distribution (Single Seed)')
        
        plt.tight_layout()
        plt.savefig(tree_dir / "tree_statistics.png", dpi=200, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print(f"\nTree Statistics Summary:")
        print(f"  Total seeds: {lineage_df['seed_id'].nunique()}")
        print(f"  Total prompts: {len(lineage_df)}")
        print(f"  Prompts per seed (avg): {len(lineage_df) / lineage_df['seed_id'].nunique():.1f}")
        if seed_depths:
            print(f"  Max tree depth: {max(seed_depths.values()) if seed_depths.values() else 0}")
        if branch_factors:
            avg_branch = sum(branch_factors.values()) / len(branch_factors) if branch_factors else 0
            print(f"  Average branch factor: {avg_branch:.1f}")
        
        print(f"Tree statistics saved to {tree_dir}/")

    def tfidf_analysis(self, df, output_dir, top_n=10):
        tfidf_dir = Path(output_dir) / "tfidf"
        tfidf_dir.mkdir(parents=True, exist_ok=True)
        prompts = df['prompt'].tolist()
        clusters = df['cluster'].tolist()
        vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(prompts)
        feature_names = np.array(vectorizer.get_feature_names_out())
        # Per-cluster top terms
        cluster_top_terms = {}
        cluster_top_scores = {}
        for c in sorted(df['cluster'].unique()):
            idxs = df[df['cluster'] == c].index
            if len(idxs) == 0:
                continue
            mean_tfidf = tfidf_matrix[idxs].mean(axis=0).A1
            top_idx = mean_tfidf.argsort()[::-1][:top_n]
            cluster_top_terms[c] = feature_names[top_idx].tolist()
            cluster_top_scores[c] = mean_tfidf[top_idx].tolist()
            # Word cloud
            from wordcloud import WordCloud
            word_freq = {feature_names[i]: mean_tfidf[i] for i in top_idx}
            wc = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(word_freq)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud for Cluster {c}')
            plt.tight_layout()
            plt.savefig(tfidf_dir / f"wordcloud_cluster_{c}.png", dpi=200)
            plt.close()
            # Bar plot
            plt.figure(figsize=(8,4))
            terms = feature_names[top_idx]
            scores = mean_tfidf[top_idx]
            plt.barh(terms[::-1], scores[::-1], color='skyblue')
            plt.xlabel('Mean TF-IDF Score')
            plt.title(f'Top TF-IDF Terms for Cluster {c}')
            plt.tight_layout()
            plt.savefig(tfidf_dir / f"tfidf_barplot_cluster_{c}.png", dpi=200)
            plt.close()
        # Per-prompt top terms
        prompt_top_terms = []
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix[i].toarray().ravel()
            top_idx = row.argsort()[::-1][:top_n]
            prompt_top_terms.append(feature_names[top_idx].tolist())
        # Print and save
        print("\nTF-IDF Top Terms by Cluster:")
        for c, terms in cluster_top_terms.items():
            print(f"Cluster {c}: {', '.join(terms)}")
        df_out = df.copy()
        df_out['top_tfidf_terms'] = prompt_top_terms
        df_out.to_csv(tfidf_dir / "attack_vectors_with_tfidf.csv", index=False)
        with open(tfidf_dir / "tfidf_top_terms_by_cluster.json", 'w') as f:
            import json
            json.dump({str(k): v for k, v in cluster_top_terms.items()}, f, indent=2)

    def tfidf_success_vs_unsuccess(self, successful_prompts, unsuccessful_prompts, output_dir, top_n=10):
        tfidf_dir = Path(output_dir) / "tfidf"
        tfidf_dir.mkdir(parents=True, exist_ok=True)
        all_prompts = successful_prompts + unsuccessful_prompts
        labels = np.array([1]*len(successful_prompts) + [0]*len(unsuccessful_prompts))
        vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(all_prompts)
        feature_names = np.array(vectorizer.get_feature_names_out())
        # Compute mean TF-IDF for each group
        mean_success = tfidf_matrix[labels==1].mean(axis=0).A1
        mean_unsuccess = tfidf_matrix[labels==0].mean(axis=0).A1
        diff = mean_success - mean_unsuccess
        top_idx = diff.argsort()[::-1][:top_n]
        bottom_idx = diff.argsort()[:top_n]
        enriched_success = feature_names[top_idx]
        enriched_unsuccess = feature_names[bottom_idx]
        print("\nTop TF-IDF terms enriched in successful prompts:")
        print(", ".join(enriched_success))
        print("\nTop TF-IDF terms enriched in unsuccessful prompts:")
        print(", ".join(enriched_unsuccess))
        # Save
        import json
        with open(tfidf_dir / "tfidf_enriched_success_vs_unsuccess.json", 'w') as f:
            json.dump({
                "enriched_success": enriched_success.tolist(),
                "enriched_unsuccess": enriched_unsuccess.tolist()
            }, f, indent=2)
        # Bar plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.barh(enriched_success[::-1], diff[top_idx][::-1], color='green')
        plt.xlabel('TF-IDF Difference (Success - Unsuccess)')
        plt.title('Top TF-IDF Terms Enriched in Successful Prompts')
        plt.tight_layout()
        plt.savefig(tfidf_dir / "tfidf_barplot_enriched_success.png", dpi=200)
        plt.close()
        plt.figure(figsize=(8,4))
        plt.barh(enriched_unsuccess[::-1], diff[bottom_idx][::-1], color='red')
        plt.xlabel('TF-IDF Difference (Success - Unsuccess)')
        plt.title('Top TF-IDF Terms Enriched in Unsuccessful Prompts')
        plt.tight_layout()
        plt.savefig(tfidf_dir / "tfidf_barplot_enriched_unsuccess.png", dpi=200)
        plt.close()

    def surface_feature_analysis(self, df, attack_vectors, unsuccess_mean, output_dir):
        surface_dir = Path(output_dir) / "surface_features"
        surface_dir.mkdir(parents=True, exist_ok=True)
        features = []
        unsuccess_centroid = unsuccess_mean
        from numpy.linalg import norm
        for i, row in df.iterrows():
            text = row['prompt']
            # Basic features
            length = len(text.split())
            sentiment = TextBlob(text).sentiment.polarity
            modals = sum(1 for w in text.lower().split() if w in {"might", "could", "should", "would", "may"})
            # Lexical sophistication (MTLD)
            try:
                lex = LexicalRichness(text)
                sophistication = lex.mtld
                if sophistication is None or np.isnan(sophistication):
                    sophistication = 0.0
                # Warn if prompt is too short for MTLD
                if length < 10:
                    import warnings
                    warnings.warn(f"Prompt too short for reliable MTLD: '{text[:40]}...'")
            except Exception as e:
                sophistication = 0.0
            # Cosine similarity to unsuccessful centroid
            vec = attack_vectors[i]
            centroid_cosine = np.dot(vec, unsuccess_centroid) / (norm(vec) * norm(unsuccess_centroid)) if norm(vec) > 0 and norm(unsuccess_centroid) > 0 else 0
            features.append({
                "word_count": length,
                "sentiment": sentiment,
                "modals": modals,
                "lex_mtld": sophistication,
                "centroid_cosine": centroid_cosine
            })
        feat_df = pd.DataFrame(features)
        df_out = df.copy()
        for col in feat_df.columns:
            df_out[col] = feat_df[col]
        df_out.to_csv(surface_dir / "attack_vectors_with_surface_features.csv", index=False)
        # Ensure all columns are numeric and exist
        cols = ["word_count", "sentiment", "modals", "lex_mtld", "centroid_cosine", "magnitude"]
        for col in cols:
            if col in df_out.columns:
                df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
        # Visualizations
        import seaborn as sns
        import matplotlib.pyplot as plt
        plot_cols = [col for col in cols if col in df_out.columns]
        # Remove constant columns (all values the same)
        plot_df = df_out[plot_cols].copy()
        constant_cols = [col for col in plot_df.columns if plot_df[col].nunique() <= 1]
        if constant_cols:
            print(f"Warning: Dropping constant columns from plots: {constant_cols}")
            plot_df = plot_df.drop(columns=constant_cols)
        # Only plot if at least 2 columns remain
        if plot_df.shape[1] >= 2:
            sns.pairplot(plot_df, diag_kind="kde")
            plt.suptitle("Surface Feature Distributions (Version 3)", y=1.02)
            plt.tight_layout()
            plt.savefig(surface_dir / "surface_feature_pairplot.png", dpi=200)
            plt.close()
            # Correlation matrix
            corr = plot_df.corr()
            plt.figure(figsize=(8,6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
            plt.title('Feature Correlation Matrix (Version 3)')
            plt.tight_layout()
            plt.savefig(surface_dir / "surface_feature_correlation_matrix.png", dpi=200)
            plt.close()
        else:
            print("Not enough non-constant features for pairplot/correlation matrix.")

    def surface_feature_success_vs_unsuccess(self, successful_prompts, unsuccessful_prompts, output_dir):
        surface_dir = Path(output_dir) / "surface_features"
        surface_dir.mkdir(parents=True, exist_ok=True)
        from numpy.linalg import norm
        from pandas import DataFrame
        def extract_features(prompts, unsuccess_centroid=None):
            features = []
            for text in prompts:
                length = len(text.split())
                sentiment = TextBlob(text).sentiment.polarity
                modals = sum(1 for w in text.lower().split() if w in {"might", "could", "should", "would", "may"})
                try:
                    lex = LexicalRichness(text)
                    sophistication = lex.mtld
                    if sophistication is None or np.isnan(sophistication):
                        sophistication = 0.0
                    if length < 10:
                        import warnings
                        warnings.warn(f"Prompt too short for reliable MTLD: '{text[:40]}...'" )
                except Exception as e:
                    sophistication = 0.0
                features.append({
                    "word_count": length,
                    "sentiment": sentiment,
                    "modals": modals,
                    "lex_mtld": sophistication
                })
            return DataFrame(features)
        df_succ = extract_features(successful_prompts)
        df_unsucc = extract_features(unsuccessful_prompts)
        # Save
        df_succ['success'] = 1
        df_unsucc['success'] = 0
        df_all = pd.concat([df_succ, df_unsucc], ignore_index=True)
        df_all['success_str'] = df_all['success'].map({0: "Unsuccessful", 1: "Successful"})
        df_all.to_csv(surface_dir / "surface_features_success_vs_unsuccess.csv", index=False)
        # Boxplots
        import matplotlib.pyplot as plt
        import seaborn as sns
        for col in ["word_count", "sentiment", "modals", "lex_mtld"]:
            plt.figure(figsize=(6,4))
            sns.boxplot(x='success_str', y=col, data=df_all, palette={"Unsuccessful":'red',"Successful":'green'})
            plt.xlabel("Success")
            plt.title(f"{col} Distribution by Success")
            plt.tight_layout()
            plt.savefig(surface_dir / f"boxplot_{col}_success_vs_unsuccess.png", dpi=200)
            plt.close()
        # Print means
        print("\nSurface feature means (successful vs unsuccessful):")
        print(df_all.groupby('success').mean(numeric_only=True))

    def analyze(self, n_clusters: int = None, output_dir: str = "attack_results", prompt_tree_analysis: bool = False):
        print(f"Analyzing {self.log_path}...")
        
        # Extract prompts
        successful_prompts, unsuccessful_prompts = self.extract_prompts_by_status()
        
        print(f"Found {len(successful_prompts)} successful and {len(unsuccessful_prompts)} unsuccessful prompts")
        
        if len(successful_prompts) == 0:
            print("No successful prompts found!")
            return
        
        if len(unsuccessful_prompts) == 0:
            print("No unsuccessful prompts found!")
            return
        
        # Get embeddings
        print("Computing embeddings...")
        successful_embeddings = self.model.encode(successful_prompts)
        unsuccessful_embeddings = self.model.encode(unsuccessful_prompts)
        
        # Compute attack vectors
        print("Computing attack vectors...")
        avg_unsuccessful = np.mean(unsuccessful_embeddings, axis=0)
        attack_vectors = successful_embeddings - avg_unsuccessful
        
        # Compute new attack vector metrics
        print("Computing enhanced attack vector metrics...")
        attack_vectors_nu, attack_vectors_mu, attack_vectors_sp= self._compute_enhanced_attack_vectors(
            successful_embeddings, unsuccessful_embeddings
        )
        
        # Find optimal k if not specified
        if n_clusters is None:
            print("Finding optimal number of clusters (silhouette analysis)...")
            n_clusters = self.find_optimal_k(attack_vectors)
            print(f"Optimal clusters (silhouette): k = {n_clusters}")
        else:
            print(f"Using user-specified number of clusters: k = {n_clusters}")
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(attack_vectors)
        
        # Calculate silhouette score
        try:
            if len(np.unique(cluster_labels)) >= 2 and len(attack_vectors) > len(np.unique(cluster_labels)):
                silhouette = silhouette_score(attack_vectors, cluster_labels)
            else:
                silhouette = 0.0
                print(f"Warning: Cannot compute silhouette score with {len(np.unique(cluster_labels))} clusters and {len(attack_vectors)} samples")
        except ValueError as e:
            print(f"Warning: Silhouette score calculation failed: {e}")
            silhouette = 0.0
        
        # Reduce dimensionality for visualization
        print("Creating visualizations...")
        if len(attack_vectors) > 1:
            # TSNE requires perplexity < n_samples
            perplexity = min(30, len(attack_vectors) - 1)
            if perplexity < 1:
                perplexity = 1
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                reduced_vectors = tsne.fit_transform(attack_vectors)
            except ValueError as e:
                print(f"Warning: TSNE failed, using PCA instead: {e}")
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(2, len(attack_vectors), attack_vectors.shape[1]))
                reduced_vectors = pca.fit_transform(attack_vectors)
        else:
            # If only one vector, create a dummy 2D representation
            reduced_vectors = np.array([[0, 0]])
        
        # Calculate magnitudes for all attack vector types
        magnitudes = np.linalg.norm(attack_vectors, axis=1)
        magnitudes_nu = np.linalg.norm(attack_vectors_nu, axis=1)
        magnitudes_mu = np.linalg.norm(attack_vectors_mu, axis=1)
        magnitudes_sp = np.linalg.norm(attack_vectors_sp, axis=1)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame({
            'prompt': successful_prompts,
            'cluster': cluster_labels,
            'magnitude': magnitudes,
            'magnitude_nu': magnitudes_nu,
            'magnitude_mu': magnitudes_mu,
            'magnitude_sp': magnitudes_sp,
            'cosine_sim_nu': self.nu_similarities,
            'cosine_sim_mu': self.mu_similarities,
            'cosine_sim_sp': self.sp_similarities,
            'seed_prompt': self.seed_prompt_texts if hasattr(self, 'seed_prompt_texts') else ['Unknown'] * len(successful_prompts),
            'x': reduced_vectors[:, 0],
            'y': reduced_vectors[:, 1]
        })
        
        # Save results
        df.to_csv(output_path / "attack_vectors.csv", index=False)
        
        # Create comprehensive visualizations including SP metrics
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # Row 1: Traditional plots
        # Cluster plot
        scatter = axes[0, 0].scatter(df['x'], df['y'], c=df['cluster'], cmap='viridis', alpha=0.7)
        axes[0, 0].set_title(f'Attack Vectors by Cluster (n={n_clusters})')
        axes[0, 0].set_xlabel('t-SNE 1')
        axes[0, 0].set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Magnitude distribution comparison
        axes[0, 1].hist([df['magnitude'], df['magnitude_nu'], df['magnitude_mu'], df['magnitude_sp']], 
                       bins=15, alpha=0.6, label=['Original (vs avg)', 'NU (vs nearest)', 'MU (vs cluster)', 'SP (vs seed)'],
                       edgecolor='black')
        axes[0, 1].set_title('Distribution of Attack Vector Magnitudes')
        axes[0, 1].set_xlabel('Magnitude')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Row 2: Enhanced magnitude vs similarity analysis
        # NU and MU magnitude vs similarity
        axes[1, 0].scatter(df['magnitude_nu'], df['cosine_sim_nu'], 
                          alpha=0.7, c=df['cluster'], cmap='viridis', label='NU', s=60)
        axes[1, 0].scatter(df['magnitude_mu'], df['cosine_sim_mu'], 
                          alpha=0.7, c=df['cluster'], cmap='viridis', marker='^', label='MU', s=60)
        axes[1, 0].set_xlabel('Attack Vector Magnitude')
        axes[1, 0].set_ylabel('Cosine Similarity')
        axes[1, 0].set_title('Magnitude vs Similarity (NU & MU)')
        axes[1, 0].legend()
        
        # SP magnitude vs similarity (evolutionary trajectory analysis)
        scatter = axes[1, 1].scatter(df['magnitude_sp'], df['cosine_sim_sp'], 
                                   alpha=0.7, c=df['cluster'], cmap='viridis', s=60)
        axes[1, 1].set_xlabel('SP Magnitude (Seed → Success Evolution)')
        axes[1, 1].set_ylabel('SP Similarity (to Seed)')
        axes[1, 1].set_title('Evolutionary Trajectory: Seed to Success')
        axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Mid Similarity')
        axes[1, 1].legend()
        plt.colorbar(scatter, ax=axes[1, 1], label='Cluster')
        
        # Row 3: Comparative magnitude analysis
        # NU vs MU magnitude (colored by SP magnitude)
        scatter = axes[2, 0].scatter(df['magnitude_nu'], df['magnitude_mu'], 
                                   alpha=0.7, c=df['magnitude_sp'], cmap='plasma', s=60)
        axes[2, 0].set_xlabel('Magnitude NU (vs. nearest unsuccessful)')
        axes[2, 0].set_ylabel('Magnitude MU (vs. cluster mean)')
        axes[2, 0].set_title('NU vs MU Magnitude (colored by SP evolution)')
        plt.colorbar(scatter, ax=axes[2, 0], label='SP Magnitude')
        
        # Add diagonal line for reference
        min_val = min(df['magnitude_nu'].min(), df['magnitude_mu'].min())
        max_val = max(df['magnitude_nu'].max(), df['magnitude_mu'].max())
        axes[2, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='NU = MU')
        axes[2, 0].legend()
        
        # SP vs Original magnitude comparison
        scatter = axes[2, 1].scatter(df['magnitude'], df['magnitude_sp'], 
                                   alpha=0.7, c=df['cluster'], cmap='viridis', s=60)
        axes[2, 1].set_xlabel('Original Magnitude (vs. avg unsuccessful)')
        axes[2, 1].set_ylabel('SP Magnitude (vs. seed)')
        axes[2, 1].set_title('Original vs Evolutionary Magnitude')
        plt.colorbar(scatter, ax=axes[2, 1], label='Cluster')
        
        # Add diagonal line for reference
        min_val = min(df['magnitude'].min(), df['magnitude_sp'].min())
        max_val = max(df['magnitude'].max(), df['magnitude_sp'].max())
        axes[2, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Original = SP')
        axes[2, 1].legend()
        min_val = min(df['magnitude_nu'].min(), df['magnitude_mu'].min())
        max_val = max(df['magnitude_nu'].max(), df['magnitude_mu'].max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / "attack_vectors_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print first 5 prompts per cluster
        print("\nSample prompts by cluster (first 5 per cluster):")
        for c in sorted(df['cluster'].unique()):
            print(f"\n===== Cluster {c} =====")
            sample = df[df['cluster'] == c].head(5)
            for p in sample['prompt']:
                print("-", p[:200].replace("\n", " "), "…")

        # Compute pairwise cosine similarity matrix between all attack_vec_nu vectors
        nu_cosine_pairwise_similarities = cosine_similarity(attack_vectors_nu)
        sp_cosine_pairwise_similarities = cosine_similarity(attack_vectors_sp)

        # Compute the mean of the upper triangle of the cosine similarity matrix
        nu_cosine_pairwise_similarities_mean = np.mean(nu_cosine_pairwise_similarities)
        sp_cosine_pairwise_similarities_mean = np.mean(sp_cosine_pairwise_similarities)
        
        # Compute the std of the upper triangle of the cosine similarity matrix
        nu_cosine_pairwise_similarities_std = np.std(nu_cosine_pairwise_similarities)
        sp_cosine_pairwise_similarities_std = np.std(sp_cosine_pairwise_similarities)

        # Compute pairwise L2 distance matrix between all attack_vec_nu vectors
        
        nu_l2_pairwise_distances = euclidean_distances(attack_vectors_nu)
        sp_l2_pairwise_distances = euclidean_distances(attack_vectors_sp)

        # Compute the mean of the upper triangle of the L2 distance matrix
        nu_l2_pairwise_distances_mean = np.mean(nu_l2_pairwise_distances)
        sp_l2_pairwise_distances_mean = np.mean(sp_l2_pairwise_distances)
        
        # Compute the std of the upper triangle of the L2 distance matrix
        nu_l2_pairwise_distances_std = np.std(nu_l2_pairwise_distances)
        sp_l2_pairwise_distances_std = np.std(sp_l2_pairwise_distances)

        # Save summary statistics
        summary = {
            'log_file': str(self.log_path),
            'total_successful': len(successful_prompts),
            'total_unsuccessful': len(unsuccessful_prompts),
            'success_rate': len(successful_prompts) / (len(successful_prompts) + len(unsuccessful_prompts)),
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'avg_magnitude': float(np.mean(magnitudes)),
            'std_magnitude': float(np.std(magnitudes)),
            'avg_magnitude_nu': float(np.mean(magnitudes_nu)),
            'std_magnitude_nu': float(np.std(magnitudes_nu)),
            'avg_magnitude_mu': float(np.mean(magnitudes_mu)),
            'std_magnitude_mu': float(np.std(magnitudes_mu)),
            'avg_magnitude_sp': float(np.mean(magnitudes_sp)),
            'std_magnitude_sp': float(np.std(magnitudes_sp)),
            'avg_cosine_sim_nu': float(np.mean(self.nu_similarities)),
            'std_cosine_sim_nu': float(np.std(self.nu_similarities)),
            'avg_cosine_sim_mu': float(np.mean(self.mu_similarities)),
            'std_cosine_sim_mu': float(np.std(self.mu_similarities)),
            'avg_cosine_sim_sp': float(np.mean(self.sp_similarities)),
            'std_cosine_sim_sp': float(np.std(self.sp_similarities)),
            'avg_nu_pairwise_similarities': float(nu_cosine_pairwise_similarities_mean),
            'std_nu_pairwise_similarities': float(nu_cosine_pairwise_similarities_std),
            'avg_sp_pairwise_similarities': float(sp_cosine_pairwise_similarities_mean),
            'std_sp_pairwise_similarities': float(sp_cosine_pairwise_similarities_std),
            'avg_nu_pairwise_L2_distances': float(nu_l2_pairwise_distances_mean),
            'std_nu_pairwise_L2_distances': float(nu_l2_pairwise_distances_std),
            'avg_sp_pairwise_L2_distances': float(sp_l2_pairwise_distances_mean),
            'std_sp_pairwise_L2_distances': float(sp_l2_pairwise_distances_std),
            'cluster_distribution': df['cluster'].value_counts().to_dict(),
            'diversity': self._compute_diversity_metrics(magnitudes_nu, magnitudes_sp, self.nu_similarities, self.sp_similarities)
        }
        
        with open(output_path / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\nAnalysis complete! Results saved to {output_path}/")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Silhouette score: {silhouette:.4f}")
        print(f"Attack Vector Magnitudes:")
        print(f"  Original (vs. avg unsuccessful): {summary['avg_magnitude']:.4f} ± {summary['std_magnitude']:.4f}")
        print(f"  NU (vs. nearest unsuccessful): {summary['avg_magnitude_nu']:.4f} ± {summary['std_magnitude_nu']:.4f}")
        print(f"  MU (vs. cluster mean): {summary['avg_magnitude_mu']:.4f} ± {summary['std_magnitude_mu']:.4f}")
        print(f"  SP (vs. seed prompt): {summary['avg_magnitude_sp']:.4f} ± {summary['std_magnitude_sp']:.4f}")
        print(f"Cosine Similarities:")
        print(f"  NU (to nearest unsuccessful): {summary['avg_cosine_sim_nu']:.4f} ± {summary['std_cosine_sim_nu']:.4f}")
        print(f"  MU (to cluster centroid): {summary['avg_cosine_sim_mu']:.4f} ± {summary['std_cosine_sim_mu']:.4f}")
        print(f"  SP (to seed prompt): {summary['avg_cosine_sim_sp']:.4f} ± {summary['std_cosine_sim_sp']:.4f}")
        print(f"Cluster distribution: {summary['cluster_distribution']}")
        
        # Compute correlations between different magnitude measures
        corr_orig_nu = np.corrcoef(magnitudes, magnitudes_nu)[0, 1]
        corr_orig_mu = np.corrcoef(magnitudes, magnitudes_mu)[0, 1]
        corr_orig_sp = np.corrcoef(magnitudes, magnitudes_sp)[0, 1]
        corr_nu_mu = np.corrcoef(magnitudes_nu, magnitudes_mu)[0, 1]
        corr_nu_sp = np.corrcoef(magnitudes_nu, magnitudes_sp)[0, 1]
        corr_mu_sp = np.corrcoef(magnitudes_mu, magnitudes_sp)[0, 1]
        
        # Correlations between magnitudes and similarities
        corr_mag_nu_sim_nu = np.corrcoef(magnitudes_nu, self.nu_similarities)[0, 1]
        corr_mag_mu_sim_mu = np.corrcoef(magnitudes_mu, self.mu_similarities)[0, 1]
        corr_mag_sp_sim_sp = np.corrcoef(magnitudes_sp, self.sp_similarities)[0, 1]
        corr_sim_nu_mu = np.corrcoef(self.nu_similarities, self.mu_similarities)[0, 1]
        corr_sim_nu_sp = np.corrcoef(self.nu_similarities, self.sp_similarities)[0, 1]
        corr_sim_mu_sp = np.corrcoef(self.mu_similarities, self.sp_similarities)[0, 1]
        
        print(f"\nMagnitude Correlations:")
        print(f"  Original vs NU: {corr_orig_nu:.3f}")
        print(f"  Original vs MU: {corr_orig_mu:.3f}")
        print(f"  Original vs SP: {corr_orig_sp:.3f}")
        print(f"  NU vs MU: {corr_nu_mu:.3f}")
        print(f"  NU vs SP: {corr_nu_sp:.3f}")
        print(f"  MU vs SP: {corr_mu_sp:.3f}")
        print(f"\nMagnitude-Similarity Correlations:")
        print(f"  NU Magnitude vs NU Similarity: {corr_mag_nu_sim_nu:.3f}")
        print(f"  MU Magnitude vs MU Similarity: {corr_mag_mu_sim_mu:.3f}")
        print(f"  SP Magnitude vs SP Similarity: {corr_mag_sp_sim_sp:.3f}")
        print(f"\nSimilarity Cross-Correlations:")
        print(f"  NU vs MU Similarity: {corr_sim_nu_mu:.3f}")
        print(f"  NU vs SP Similarity: {corr_sim_nu_sp:.3f}")
        print(f"  MU vs SP Similarity: {corr_sim_mu_sp:.3f}")
        
        # Print comprehensive metrics explanation
        self.print_metrics_explanation()
        
        # Save DataFrame for further analysis
        self.df = df
        self.attack_vectors = attack_vectors
        self.attack_vectors_nu = attack_vectors_nu
        self.attack_vectors_mu = attack_vectors_mu
        self.avg_unsuccessful = avg_unsuccessful
        
        # TF-IDF analysis (always run)
        self.tfidf_analysis(df, output_dir, top_n=10)
        # Version 3: Surface feature analysis (always run)
        self.surface_feature_analysis(df, attack_vectors, avg_unsuccessful, output_dir)
        # Attack vector comparison analysis (new)
        self.compare_attack_vector_types(output_dir)
        # Comprehensive similarity analysis (new)
        self.similarity_analysis(output_dir)
        # Rainbow teaming specific analysis (new)
        self.create_rainbow_teaming_visualizations(output_dir)
        # Success vs unsuccessful comparison (always run)
        self.tfidf_success_vs_unsuccess(successful_prompts, unsuccessful_prompts, output_dir, top_n=10)
        self.surface_feature_success_vs_unsuccess(successful_prompts, unsuccessful_prompts, output_dir)
        # Load lineage fields if present
        has_lineage = self.load_lineage_fields()
        
        # Version 2: Seed-grouped and lineage analysis (only if flag is set)
        if prompt_tree_analysis:
            print("\nVersion 2: Seed-grouped and lineage analysis:")
            lineage_df = self.lineage_dataframe()
            
            if not lineage_df.empty:
                # Per-seed attack vector stats
                try:
                    per_seed = lineage_df.groupby('seed_id').agg({
                        'prompt': 'count',
                        'magnitude': ['mean', 'std', 'max'],
                        'cluster': pd.Series.nunique
                    })
                    print("\nPer-seed attack vector stats:")
                    print(per_seed.head())
                except Exception as e:
                    print(f"Error computing per-seed stats: {e}")
                    per_seed = pd.DataFrame()
                
                # Tree metrics: depth, branch factor
                try:
                    # Build parent-child mapping
                    parent_map = {}
                    for idx, row in lineage_df.iterrows():
                        pid = row['prompt_id']
                        parid = row['parent_id']
                        if parid is not None:  # Only add if parent exists
                            if parid not in parent_map:
                                parent_map[parid] = []
                            parent_map[parid].append(pid)
                    
                    # Compute depth for each prompt
                    def compute_depth(pid, parent_map, depth=0, visited=None):
                        if visited is None:
                            visited = set()
                        if pid in visited:  # Avoid infinite recursion
                            return depth
                        visited.add(pid)
                        
                        if pid not in parent_map or not parent_map[pid]:
                            return depth
                        return max(compute_depth(child, parent_map, depth+1, visited.copy()) 
                                 for child in parent_map[pid])
                    
                    seed_depths = {}
                    for seed in lineage_df['seed_id'].unique():
                        # Find root prompt_ids for this seed (no parent)
                        root_ids = lineage_df[
                            (lineage_df['seed_id']==seed) & 
                            (lineage_df['parent_id'].isnull())
                        ]['prompt_id'].tolist()
                        
                        if root_ids:
                            seed_depths[seed] = max(compute_depth(root, parent_map) for root in root_ids)
                        else:
                            # If no clear roots, use depth 0
                            seed_depths[seed] = 0
                    
                    print("\nMax tree depth per seed:")
                    print(seed_depths if seed_depths else "No tree depth data")
                    
                    # Branch factor
                    branch_factors = {pid: len(children) for pid, children in parent_map.items() if children}
                    print("\nSample branch factors:")
                    print(dict(list(branch_factors.items())[:5]) if branch_factors else "No branch factor data")
                    
                except Exception as e:
                    print(f"Error computing tree metrics: {e}")
                    seed_depths = {}
                    branch_factors = {}
                
                # Save lineage DataFrame
                try:
                    tree_dir = Path(output_dir) / "tree_analysis"
                    tree_dir.mkdir(parents=True, exist_ok=True)
                    lineage_df.to_csv(tree_dir / "lineage_analysis.csv", index=False)
                    print(f"Lineage data saved to {tree_dir / 'lineage_analysis.csv'}")
                except Exception as e:
                    print(f"Error saving lineage data: {e}")
                
                # Analyze mutations from replaced parents (archive dynamics)
                try:
                    mutations_from_replaced = self.analyze_mutations_from_replaced_parents(lineage_df)
                    # Save the analysis results
                    if mutations_from_replaced:
                        mutations_df = pd.DataFrame(list(mutations_from_replaced.values()))
                        mutations_df.to_csv(tree_dir / "mutations_from_replaced_parents.csv", index=False)
                        print(f"Mutations from replaced parents analysis saved to {tree_dir / 'mutations_from_replaced_parents.csv'}")
                except Exception as e:
                    print(f"Error analyzing mutations from replaced parents: {e}")
                
                # Visualizations
                try:
                    self.visualize_prompt_tree(lineage_df, output_dir)
                except Exception as e:
                    print(f"Error creating prompt tree visualizations: {e}")
                
                try:
                    self.visualize_tree_stats(lineage_df, seed_depths, branch_factors, output_dir)
                except Exception as e:
                    print(f"Error creating tree statistics: {e}")
            else:
                print("No lineage data available for tree analysis.")
        else:
            print("Prompt tree analysis disabled (use --prompt_tree_analysis to enable)")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Run attack vector analysis on comprehensive logs")
    parser.add_argument("log_path", help="Path to comprehensive log JSON file")
    parser.add_argument("--clusters", type=int, default=None, help="Number of clusters to use (if not set, auto-select)")
    parser.add_argument("--output", default="attack_results", help="Output directory")
    parser.add_argument("--prompt_tree_analysis", action="store_true", help="Enable prompt tree/lineage/seed-grouped analysis and visualizations (Version 2 features)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_path):
        print(f"Error: Log file {args.log_path} not found!")
        return
    
    analyzer = AttackVectorAnalyzer(args.log_path)
    analyzer.analyze(n_clusters=args.clusters, output_dir=args.output, prompt_tree_analysis=args.prompt_tree_analysis)

if __name__ == "__main__":
    main() 