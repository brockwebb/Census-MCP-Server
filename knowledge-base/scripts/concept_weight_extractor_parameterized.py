#!/usr/bin/env python3
"""
concept_weight_extractor_parameterized.py
Flexible configuration for different category sets and thresholds.
Modified to support raw similarity output for geography scalars.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import time
import sys
from datetime import datetime

# ==================== CONFIGURATION ====================
# Easy to modify parameters at top of file

# No Geography Configuration (7 categories, geography excluded)
NO_GEOGRAPHY_CATEGORIES = [
    'core_demographics',
    'economics',
    'education',
    # 'geography',  # REMOVED for Step 0
    'health_social',
    'housing',
    'specialized_populations',
    'transportation'
]

# Geography Only Configuration (1 category, just geography)
GEOGRAPHY_ONLY_CATEGORIES = [
    'geography'
]

# Alternative configurations (for testing)
ORIGINAL_8_CATEGORIES = [
    'core_demographics',
    'economics',
    'education',
    'geography',
    'health_social',
    'housing',
    'specialized_populations',
    'transportation'
]

DEMOGRAPHICS_INDEPENDENT = [
    'core_demographics',
    'demographics_extended',  # if we want to split this out
    'economics',
    'education',
    'health_social',
    'housing',
    'specialized_populations',
    'transportation'
]

# Model and processing parameters
DEFAULT_MODEL = 'BAAI/bge-large-en-v1.5'
STEP0_THRESHOLD = 0.09  # For 12 categories (1/12 = 8.3% + buffer)
ORIGINAL_THRESHOLD = 0.05  # Original threshold
DEFAULT_BATCH_SIZE = 100
DEFAULT_EMBED_BATCH_SIZE = 50

# Output format options
KEEP_LOG_WEIGHTS = True  # Always keep log weights (they cost nothing)
SIMPLIFIED_OUTPUT = False  # If True, output only category_weights field

# ==================== END CONFIGURATION ====================

# Set up logging with minimal noise
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConceptWeightExtractor:
    def __init__(self,
                 categories_config: Dict[str, List[str]],
                 threshold: float = STEP0_THRESHOLD,
                 model_name: str = DEFAULT_MODEL,
                 embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
                 checkpoint_batch_size: int = DEFAULT_BATCH_SIZE,
                 simplified_output: bool = SIMPLIFIED_OUTPUT):
        
        self.categories = categories_config['categories']
        self.threshold = threshold
        self.model_name = model_name
        self.embed_batch_size = embed_batch_size
        self.checkpoint_batch_size = checkpoint_batch_size
        self.simplified_output = simplified_output
        self.raw_scalars = False  # Will be set by main()
        
        # Generate cache-friendly identifiers
        categories_str = '_'.join(sorted(self.categories))
        threshold_str = f"thresh{str(threshold).replace('.', 'p')}"
        model_str = model_name.replace('/', '_').replace('-', '_')
        
        self.cache_id = f"{categories_str}_{threshold_str}_{model_str}"
        
        # Load model
        logger.info(f"Loading sentence transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("✅ Model loaded successfully")
        
        # Load and prepare category concepts
        self.category_concepts = self._load_and_group_concepts()
        self.category_embeddings = self._get_category_embeddings()
        
        logger.info(f"Loaded {len(self.categories)} categories")
        for cat in self.categories:
            count = len(self.category_concepts.get(cat, []))
            logger.info(f"  {cat}: {count} concepts")
    
    def _load_and_group_concepts(self) -> Dict[str, List[Dict]]:
        """Load concepts and group by category"""
        with open("COOS_Complete_Ontology.json", 'r') as f:
            ontology = json.load(f)
        
        # Group concepts by category
        category_concepts = {cat: [] for cat in self.categories}
        
        # Get approved concepts
        for concept in ontology.get('approved_concepts', []):
            category = concept.get('category', '').lower()
            if category in category_concepts:
                category_concepts[category].append(concept)
        
        # Get modified concepts
        for concept in ontology.get('modified_concepts', []):
            category = concept.get('category', '').lower()
            if category in category_concepts:
                category_concepts[category].append(concept)
        
        return category_concepts
    
    def _get_category_embeddings(self) -> Dict[str, np.ndarray]:
        """Generate embeddings for each category by combining concept texts"""
        category_embeddings = {}
        
        for category, concepts in self.category_concepts.items():
            if not concepts:
                logger.warning(f"No concepts found for category: {category}")
                continue
            
            # Combine all concept texts for this category
            category_texts = []
            for concept in concepts:
                text_parts = [
                    concept.get('label', ''),
                    concept.get('definition', ''),
                    concept.get('universe', '')
                ]
                concept_text = ' | '.join([part for part in text_parts if part])
                category_texts.append(concept_text)
            
            # Create comprehensive category text by combining all concepts
            combined_category_text = ' | '.join(category_texts)
            
            # Generate embedding for this category
            category_embedding = self.model.encode([combined_category_text], show_progress_bar=False)[0]
            category_embeddings[category] = category_embedding
            
        logger.info(f"Generated embeddings for {len(category_embeddings)} categories")
        return category_embeddings
    
    def _load_universe_data(self) -> List[Dict]:
        """Load enriched universe data"""
        with open("2023_ACS_Enriched_Universe.json", 'r') as f:
            data = json.load(f)
        
        variables = data.get('variables', {})
        
        # Convert to list format for processing
        universe_list = []
        for var_id, var_data in variables.items():
            var_entry = {
                'variable_id': var_id,
                **var_data
            }
            universe_list.append(var_entry)
        
        logger.info(f"Loaded {len(universe_list)} variables from universe")
        return universe_list
    
    def _get_analysis_text(self, variable: Dict[str, Any]) -> str:
        """Extract analysis text from variable data - same as working extractor"""
        # Check for enrichment_text field first (from consolidated universe)
        if variable.get('enrichment_text'):
            return variable.get('enrichment_text')
        
        # Check for concept field
        if variable.get('concept'):
            return variable.get('concept')
        
        # Check for label field
        if variable.get('label'):
            return variable.get('label')
        
        # Check nested enrichment structure (fallback)
        enrichment = variable.get('enrichment', {})
        if enrichment.get('analysis'):
            return enrichment.get('analysis')
        
        if enrichment.get('enhanced_description'):
            return enrichment.get('enhanced_description')
        
        # Legacy nested analysis structures
        for analysis_key in ['coos_analysis', 'bulk_analysis', 'early_analysis']:
            if analysis_key in variable:
                analysis = variable[analysis_key]
                if isinstance(analysis, dict) and 'enhanced_description' in analysis:
                    return analysis['enhanced_description']
        
        # Fallback to original description
        return variable.get('description', '')
    
    def _compute_category_weights_batch(self, analysis_texts: List[str]) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        """Compute category weights for batch of analysis texts - returns both linear and log weights"""
        # Filter out empty texts and track indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(analysis_texts):
            if text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            return [{} for _ in analysis_texts], [{} for _ in analysis_texts]
        
        # Generate embeddings for valid texts
        text_embeddings = self.model.encode(valid_texts, show_progress_bar=False, batch_size=self.embed_batch_size)
        
        # Create matrix of category embeddings
        category_embedding_matrix = np.array([self.category_embeddings[cat] for cat in self.categories])
        
        # Compute cosine similarities (batch operation)
        similarities_batch = cosine_similarity(text_embeddings, category_embedding_matrix)
        
        # Process results for both linear and log weights
        all_linear_weights = []
        all_log_weights = []
        valid_idx = 0
        
        for i in range(len(analysis_texts)):
            if i in valid_indices:
                similarities = similarities_batch[valid_idx]
                
                # Create raw weights dictionary with categories
                raw_weights = {}
                for j, category in enumerate(self.categories):
                    similarity = float(similarities[j])
                    raw_weights[category] = similarity
                
                # Process linear weights (standard normalization)
                linear_weights = self._process_weights_linear(raw_weights)
                all_linear_weights.append(linear_weights)
                
                # Process log weights (log normalization for winner-takes-most)
                log_weights = self._process_weights_log(raw_weights)
                all_log_weights.append(log_weights)
                
                valid_idx += 1
            else:
                all_linear_weights.append({})
                all_log_weights.append({})
        
        return all_linear_weights, all_log_weights
    
    def _process_weights_linear(self, raw_weights: Dict[str, float]) -> Dict[str, float]:
        """Process weights with linear normalization"""
        if not raw_weights:
            return {}
        
        # NEW: If raw scalars requested, skip normalization
        if hasattr(self, 'raw_scalars') and self.raw_scalars:
            return raw_weights
            
        # Normalize weights to sum to 1.0
        total_weight = sum(raw_weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in raw_weights.items()}
            
            # Apply threshold - remove weights below threshold
            filtered_weights = {k: v for k, v in weights.items() if v >= self.threshold}
            
            # Renormalize after filtering
            if filtered_weights:
                total_filtered = sum(filtered_weights.values())
                return {k: v/total_filtered for k, v in filtered_weights.items()}
            else:
                # If all weights below threshold, keep the highest one
                max_category = max(weights.keys(), key=lambda k: weights[k])
                return {max_category: 1.0}
        else:
            return {}
    
    def _process_weights_log(self, raw_weights: Dict[str, float]) -> Dict[str, float]:
        """Process weights with log normalization for winner-takes-most effect"""
        if not raw_weights:
            return {}
        
        # NEW: If raw scalars requested, skip normalization
        if hasattr(self, 'raw_scalars') and self.raw_scalars:
            return raw_weights
        
        # Apply log transformation to emphasize stronger categories
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        log_weights = {}
        for k, v in raw_weights.items():
            # Apply log(1 + weight) to preserve ordering while emphasizing differences
            log_weights[k] = np.log(1 + max(v, epsilon))
        
        # Normalize log weights to sum to 1.0
        total_log_weight = sum(log_weights.values())
        if total_log_weight > 0:
            normalized_log_weights = {k: v/total_log_weight for k, v in log_weights.items()}
            
            # Apply threshold - remove weights below threshold
            filtered_log_weights = {k: v for k, v in normalized_log_weights.items() if v >= self.threshold}
            
            # Renormalize after filtering
            if filtered_log_weights:
                total_filtered = sum(filtered_log_weights.values())
                return {k: v/total_filtered for k, v in filtered_log_weights.items()}
            else:
                # If all weights below threshold, keep the highest one
                max_category = max(normalized_log_weights.keys(), key=lambda k: normalized_log_weights[k])
                return {max_category: 1.0}
        else:
            return {}
    
    def extract_weights(self, universe_data: List[Dict]) -> List[Dict]:
        """Extract category weights from universe data with checkpointing"""
        total_variables = len(universe_data)
        logger.info(f"🚀 Starting category weight extraction...")
        logger.info(f"📊 Processing {total_variables:,} variables in batches of {self.embed_batch_size}")
        logger.info(f"🎯 Weight threshold: {self.threshold}")
        logger.info(f"🏷️  Categories ({len(self.categories)}): {', '.join(self.categories)}")
        
        results = []
        start_time = time.time()
        
        for i in range(0, total_variables, self.embed_batch_size):
            batch_end = min(i + self.embed_batch_size, total_variables)
            batch = universe_data[i:batch_end]
            
            # Extract analysis texts for batch
            analysis_texts = [self._get_analysis_text(var) for var in batch]
            
            # Compute weights for batch
            linear_weights_batch, log_weights_batch = self._compute_category_weights_batch(analysis_texts)
            
            # Process batch results
            for j, variable in enumerate(batch):
                linear_weights = linear_weights_batch[j]
                log_weights = log_weights_batch[j]
                
                if self.simplified_output:
                    # Simple output format - just category weights
                    results.append({
                        'variable_id': variable['variable_id'],
                        'category_weights': linear_weights
                    })
                else:
                    # Full output format with both weight types
                    results.append({
                        'variable_id': variable['variable_id'],
                        'category_weights_linear': linear_weights,
                        'category_weights_log': log_weights if KEEP_LOG_WEIGHTS else None
                    })
            
            # Progress update
            self._progress_update(batch_end, total_variables)
        
        elapsed_time = time.time() - start_time
        logger.info(f"\n🎉 Extraction complete! Processed {total_variables:,} variables")
        logger.info(f"⏱️  Total processing time: {elapsed_time:.2f} seconds")
        
        return results
    
    def _progress_update(self, current: int, total: int):
        """Update progress on same line"""
        if current == total:
            # Final update
            print(f"\r🔄 [{current:,}/{total:,}] (100.0%) | ✅ COMPLETE")
        else:
            percent = (current / total) * 100
            bar_length = 20
            filled_length = int(bar_length * current // total)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\r🔄 [{bar}] {current:,}/{total:,} ({percent:.1f}%)", end='', flush=True)

def get_categories_config(categories_name: str) -> Dict[str, Any]:
    """Get categories configuration by name"""
    configs = {
        'no_geography': {
            'categories': NO_GEOGRAPHY_CATEGORIES,
            'description': '7 categories (geography excluded)'
        },
        'geography_only': {
            'categories': GEOGRAPHY_ONLY_CATEGORIES,
            'description': '1 category (geography only)'
        },
        'original': {
            'categories': ORIGINAL_8_CATEGORIES,
            'description': '8 categories (original set)'
        },
        'demographics_independent': {
            'categories': DEMOGRAPHICS_INDEPENDENT,
            'description': '8 categories (demographics split)'
        }
    }
    
    if categories_name not in configs:
        raise ValueError(f"Unknown categories config: {categories_name}. Available: {list(configs.keys())}")
    
    return configs[categories_name]

def generate_output_filename(categories_config: Dict, threshold: float, model_name: str) -> str:
    """Generate descriptive output filename"""
    # Clean up category name
    category_count = len(categories_config['categories'])
    
    # Determine category set name
    if set(categories_config['categories']) == set(NO_GEOGRAPHY_CATEGORIES):
        cat_name = f"{category_count}cat_nogeo"
    elif set(categories_config['categories']) == set(GEOGRAPHY_ONLY_CATEGORIES):
        cat_name = "1cat_geo_only"
    elif set(categories_config['categories']) == set(ORIGINAL_8_CATEGORIES):
        cat_name = f"{category_count}cat_original"
    else:
        cat_name = f"{category_count}cat_custom"
    
    # Clean up threshold and model name
    threshold_str = f"thresh{str(threshold).replace('.', 'p')}"
    model_str = model_name.replace('/', '_').replace('-', '_')
    
    return f"2023_ACS_Enriched_Universe_{cat_name}_{threshold_str}_{model_str}.json"

def main():
    parser = argparse.ArgumentParser(description='Extract concept weights from ACS variables')
    parser.add_argument('--categories', default='no_geography',
                       choices=['no_geography', 'geography_only', 'original', 'demographics_independent'],
                       help='Category configuration to use')
    parser.add_argument('--threshold', type=float, default=STEP0_THRESHOLD,
                       help='Minimum weight threshold (default: 0.09)')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                       help='Sentence transformer model to use')
    parser.add_argument('--embed-batch-size', type=int, default=DEFAULT_EMBED_BATCH_SIZE,
                       help='Batch size for embedding generation')
    parser.add_argument('--checkpoint-batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help='Batch size for checkpointing')
    parser.add_argument('--output', help='Output filename (auto-generated if not specified)')
    parser.add_argument('--simplified-output', action='store_true',
                       help='Output only category_weights field (no log weights)')
    parser.add_argument('--raw-scalars', action='store_true',
                       help='Output raw cosine similarities without normalization')
    
    args = parser.parse_args()
    
    # Get categories configuration
    categories_config = get_categories_config(args.categories)
    
    # Generate output filename if not provided
    output_file = args.output
    if not output_file:
        output_file = generate_output_filename(categories_config, args.threshold, args.model)
    
    # Create extractor
    extractor = ConceptWeightExtractor(
        categories_config=categories_config,
        threshold=args.threshold,
        model_name=args.model,
        embed_batch_size=args.embed_batch_size,
        checkpoint_batch_size=args.checkpoint_batch_size,
        simplified_output=args.simplified_output
    )
    
    # Set raw scalars flag
    extractor.raw_scalars = args.raw_scalars
    
    # Load universe data
    universe_data = extractor._load_universe_data()
    
    # Extract weights
    results = extractor.extract_weights(universe_data)
    
    # Save results
    logger.info(f"💾 Saving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate and display statistics
    if results:
        # Calculate category distribution
        category_counts = {}
        total_variables = len(results)
        
        for result in results:
            weights_key = 'category_weights' if args.simplified_output else 'category_weights_linear'
            weights = result.get(weights_key, {})
            
            for category, weight in weights.items():
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1
        
        logger.info(f"\n📊 EXTRACTION SUMMARY ({len(categories_config['categories'])} CATEGORIES):")
        logger.info(f"   Total variables processed: {total_variables:,}")
        logger.info(f"   Categories used: {', '.join(categories_config['categories'])}")
        logger.info(f"   Weight threshold applied: {args.threshold}")
        
        logger.info(f"\n🏷️  CATEGORY DISTRIBUTION:")
        for category in sorted(category_counts.keys()):
            count = category_counts[category]
            percentage = (count / total_variables) * 100
            logger.info(f"   {category}: {count:,} variables ({percentage:.1f}%)")
    
    logger.info(f"\n✅ Processing complete! Output saved to: {output_file}")

if __name__ == "__main__":
    main()
