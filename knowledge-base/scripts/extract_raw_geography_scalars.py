#!/usr/bin/env python3
"""
extract_geography_scalars.py
Custom version of concept_weight_extractor_parameterized.py for raw geography similarity scores.

Based on the working extractor but simplified for single-category raw similarities without normalization.
Uses the same proven data loading and embedding logic.

Usage:
    python extract_geography_scalars.py
"""

import json
import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeographyScalarExtractor:
    """Extract raw geographic similarity scores using proven extractor logic"""
    
    def __init__(self, model_name='BAAI/bge-large-en-v1.5'):
        self.model_name = model_name
        self.model = None
        self.geography_embeddings = None
        
    def load_model(self):
        """Load sentence transformer model"""
        logger.info(f"Loading sentence transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("✅ Model loaded successfully")
        
    def load_geography_concepts(self):
        """Load geography concepts from COOS ontology using working extractor logic"""
        ontology_path = Path("COOS_Complete_Ontology.json")
        
        with open(ontology_path, 'r') as f:
            ontology = json.load(f)
        
        # Use EXACT same logic as working extractor
        geography_concepts = []
        
        # Get approved concepts for geography category
        for concept in ontology.get('approved_concepts', []):
            category = concept.get('category', '').lower()
            if category == 'geography':
                text_parts = [
                    concept.get('label', ''),
                    concept.get('definition', ''),
                    concept.get('universe', '')
                ]
                concept_text = ' | '.join([part for part in text_parts if part])
                if concept_text:
                    geography_concepts.append(concept_text)
        
        # Get modified concepts for geography category
        for concept in ontology.get('modified_concepts', []):
            category = concept.get('category', '').lower()
            if category == 'geography':
                text_parts = [
                    concept.get('label', ''),
                    concept.get('definition', ''),
                    concept.get('universe', '')
                ]
                concept_text = ' | '.join([part for part in text_parts if part])
                if concept_text:
                    geography_concepts.append(concept_text)
        
        logger.info(f"Loaded {len(geography_concepts)} geography concepts")
        
        if len(geography_concepts) == 0:
            logger.error("No geography concepts found! Checking ontology structure...")
            # Debug: show available categories
            approved_cats = set()
            modified_cats = set()
            for concept in ontology.get('approved_concepts', []):
                approved_cats.add(concept.get('category', '').lower())
            for concept in ontology.get('modified_concepts', []):
                modified_cats.add(concept.get('category', '').lower())
            logger.error(f"Available approved categories: {sorted(approved_cats)}")
            logger.error(f"Available modified categories: {sorted(modified_cats)}")
            return []
            
        return geography_concepts
    
    def generate_geography_embeddings(self, geography_concepts):
        """Generate embeddings for each geography concept (max-over-concepts approach)"""
        logger.info("Generating geography concept embeddings...")
        
        # Embed each concept separately (Spock's max-over-concepts approach)
        self.geography_embeddings = self.model.encode(
            geography_concepts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        logger.info(f"Generated {len(self.geography_embeddings)} geography embeddings")
        
    def load_enriched_variables(self):
        """Load enriched variables using working extractor logic"""
        universe_path = Path("2023_ACS_Enriched_Universe.json")
        
        with open(universe_path, 'r') as f:
            data = json.load(f)
        
        # Check if variables is a dict or list and handle accordingly
        variables_data = data.get('variables', {})
        logger.info(f"Loaded {len(variables_data)} variables from universe")
        
        variables = {}
        
        # Handle both dict and list formats
        if isinstance(variables_data, dict):
            # Variables is a dict with variable_id as keys
            for var_id, var_data in variables_data.items():
                # Build text combining multiple fields (same as working extractor)
                text_parts = []
                if var_data.get('label'):
                    text_parts.append(var_data.get('label'))
                if var_data.get('concept'):
                    text_parts.append(var_data.get('concept'))
                if var_data.get('enrichment_text'):
                    text_parts.append(var_data.get('enrichment_text'))
                    
                # Combine all available text
                combined_text = ' '.join(text_parts).strip()
                
                if combined_text:
                    variables[var_id] = combined_text
                    
        elif isinstance(variables_data, list):
            # Variables is a list of objects
            for row in variables_data:
                if isinstance(row, dict):
                    variable_id = row.get('variable_id')
                    if not variable_id:
                        continue
                        
                    # Build text combining multiple fields
                    text_parts = []
                    if row.get('label'):
                        text_parts.append(row.get('label'))
                    if row.get('concept'):
                        text_parts.append(row.get('concept'))
                    if row.get('enrichment_text'):
                        text_parts.append(row.get('enrichment_text'))
                        
                    # Combine all available text
                    combined_text = ' '.join(text_parts).strip()
                    
                    if combined_text:
                        variables[variable_id] = combined_text
        
        logger.info(f"Processed {len(variables)} variables with valid text")
        return variables
    
    def compute_geography_scalars(self, variables):
        """Compute raw geography similarity scores using max-over-concepts"""
        logger.info("Computing raw geography similarity scores...")
        
        # Extract variable data for batch processing
        variable_ids = list(variables.keys())
        variable_texts = list(variables.values())
        
        # Process in batches
        batch_size = 50
        geography_scalars = {}
        
        for i in tqdm(range(0, len(variable_texts), batch_size), desc="Computing geography scalars"):
            batch_texts = variable_texts[i:i+batch_size]
            batch_ids = variable_ids[i:i+batch_size]
            
            # Generate embeddings for batch
            text_embeddings = self.model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            # Compute similarities with each geography concept
            similarities_matrix = cosine_similarity(text_embeddings, self.geography_embeddings)
            
            # Take max similarity over all geography concepts (Spock's approach)
            for j, variable_id in enumerate(batch_ids):
                max_similarity = float(np.max(similarities_matrix[j]))
                geography_scalars[variable_id] = max_similarity
        
        return geography_scalars
    
    def save_results(self, geography_scalars):
        """Save raw geography scalars"""
        # Use same filename as original for consistency
        output_path = Path("geo_similarity_scalars.json")
        
        # Create output format similar to working extractor
        output_data = {}
        for var_id, score in geography_scalars.items():
            output_data[var_id] = {"geography": round(score, 4)}
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Calculate and log statistics
        values = list(geography_scalars.values())
        min_val = min(values)
        max_val = max(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        logger.info(f"✅ Raw geography scalars saved to: {output_path}")
        logger.info(f"📊 Statistics:")
        logger.info(f"   Variables processed: {len(geography_scalars)}")
        logger.info(f"   Min similarity: {min_val:.4f}")
        logger.info(f"   Max similarity: {max_val:.4f}")
        logger.info(f"   Mean similarity: {mean_val:.4f}")
        logger.info(f"   Std deviation: {std_val:.4f}")
        
        # Show first three lines for Spock's sanity check
        logger.info(f"\n📋 First three statistics for Spock's sanity check:")
        logger.info(f"   min≈{min_val:.2f}, mean≈{mean_val:.1f}, max≈{max_val:.1f}")
        
        return output_path

def main():
    """Main execution function"""
    extractor = GeographyScalarExtractor()
    
    # Load model
    extractor.load_model()
    
    # Load geography concepts
    geography_concepts = extractor.load_geography_concepts()
    if not geography_concepts:
        logger.error("Failed to load geography concepts")
        return
    
    # Generate embeddings for geography concepts
    extractor.generate_geography_embeddings(geography_concepts)
    
    # Load enriched variables
    variables = extractor.load_enriched_variables()
    if not variables:
        logger.error("Failed to load variables")
        return
    
    # Compute raw geography similarity scores
    geography_scalars = extractor.compute_geography_scalars(variables)
    
    # Save results
    output_path = extractor.save_results(geography_scalars)
    
    logger.info(f"\n🎯 Success! Geography scalars ready for GeoAdvisor at: {output_path}")

if __name__ == "__main__":
    main()
