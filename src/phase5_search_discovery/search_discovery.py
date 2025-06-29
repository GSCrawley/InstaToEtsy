"""
Instagram to Etsy Automation - Phase 5: Search Discovery

This module implements the Multi-agent Retrieval Protocol for intelligent content discovery,
search capabilities, and optimization of content acquisition from Instagram.

Components:
1. QueryAgent: Refines and optimizes search queries
2. RetrievalAgent: Executes searches against Instagram
3. RerankerAgent: Reranks and filters results for relevance
4. SummarizationAgent: Generates metadata for Etsy listings
"""

import os
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import random

from .. import config
from ..phase1_acquisition.instagram_scraper import process_instagram_posts

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchDiscovery:
    """
    Implements the Multi-agent Retrieval Protocol for intelligent content discovery
    from Instagram and optimization for Etsy listing generation.
    """
    
    def __init__(self, base_dir: str = 'data'):
        """
        Initialize the search discovery system.
        
        Args:
            base_dir: Base directory for storing search results and metadata
        """
        self.base_dir = base_dir
        self.query_agent = QueryAgent()
        self.retrieval_agent = RetrievalAgent(base_dir=base_dir)
        self.reranker_agent = RerankerAgent()
        self.summarization_agent = SummarizationAgent()
        
        # Create directories for search results
        os.makedirs(os.path.join(base_dir, 'search'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'search', 'results'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'search', 'metadata'), exist_ok=True)
        
        logger.info("Search Discovery system initialized")
        
    def discover_content(self, 
                        search_query: str, 
                        max_results: int = 10,
                        min_quality_score: float = 0.7) -> Dict[str, Any]:
        """
        Execute the full discovery workflow to find relevant content on Instagram.
        
        Args:
            search_query: The user's search query or content requirements
            max_results: Maximum number of results to return
            min_quality_score: Minimum quality score for results
            
        Returns:
            Dictionary containing search results and metadata
        """
        logger.info(f"Starting content discovery for query: {search_query}")
        start_time = time.time()
        
        # Step 1: Query refinement
        refined_queries = self.query_agent.refine_query(search_query)
        logger.info(f"Generated {len(refined_queries)} refined queries")
        
        # Step 2: Content retrieval
        all_results = []
        for query in refined_queries:
            query_text = query['query']
            query_score = query['score']
            
            logger.info(f"Executing retrieval for query: {query_text} (score: {query_score:.2f})")
            results = self.retrieval_agent.retrieve_content(query_text)
            
            # Add query metadata to results
            for result in results:
                result['query_text'] = query_text
                result['query_score'] = query_score
                
            all_results.extend(results)
            
        logger.info(f"Retrieved {len(all_results)} total results across all queries")
        
        # Step 3: Reranking and filtering
        ranked_results = self.reranker_agent.rerank_results(all_results, search_query)
        
        # Filter by quality score
        filtered_results = [
            r for r in ranked_results 
            if r.get('quality_score', 0) >= min_quality_score
        ]
        
        # Limit to max_results
        top_results = filtered_results[:max_results]
        
        logger.info(f"After reranking and filtering: {len(top_results)} results meet quality threshold")
        
        # Step 4: Generate metadata
        for result in top_results:
            result['etsy_metadata'] = self.summarization_agent.generate_metadata(
                result, search_query
            )
            
        # Save search results
        timestamp = int(time.time())
        results_path = os.path.join(self.base_dir, 'search', 'results', f"search_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump({
                'query': search_query,
                'timestamp': timestamp,
                'execution_time': time.time() - start_time,
                'total_results': len(all_results),
                'filtered_results': len(filtered_results),
                'returned_results': len(top_results),
                'results': top_results
            }, f, indent=2)
            
        logger.info(f"Search discovery complete. Results saved to {results_path}")
        
        return {
            'query': search_query,
            'total_results': len(all_results),
            'returned_results': len(top_results),
            'results': top_results,
            'results_path': results_path
        }

class QueryAgent:
    """
    Agent for refining and expanding user queries to optimize search results.
    Implements the Query Agent role from the Multi-agent Retrieval Protocol.
    """
    
    def __init__(self):
        """Initialize the query agent."""
        pass
        
    def refine_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Refine and expand a user query into multiple search variations.
        
        Args:
            query: The original user query
            
        Returns:
            List of refined queries with relevance scores
        """
        # Simplified implementation - in production, this would use more sophisticated
        # NLP techniques or potentially call out to an LLM for query refinement
        
        # Extract main keywords
        keywords = [k.strip() for k in query.lower().split() if len(k.strip()) > 3]
        
        # Generate variations
        variations = [
            {'query': query, 'score': 1.0},  # Original query with highest score
        ]
        
        # Add Instagram-specific variations
        if 'landscape' in query.lower():
            variations.append({
                'query': f"beautiful landscape photography {' '.join(keywords[:2] if len(keywords) > 2 else keywords)}",
                'score': 0.9
            })
            variations.append({
                'query': f"scenic landscape views {' '.join(keywords[:2] if len(keywords) > 2 else keywords)}",
                'score': 0.85
            })
            
        if 'mountain' in query.lower():
            variations.append({
                'query': f"mountain peaks photography {' '.join(keywords[:2] if len(keywords) > 2 else keywords)}",
                'score': 0.9
            })
            
        if 'water' in query.lower() or 'lake' in query.lower() or 'ocean' in query.lower():
            variations.append({
                'query': f"water reflection photography {' '.join(keywords[:2] if len(keywords) > 2 else keywords)}",
                'score': 0.9
            })
            
        # Add some general high-performing searches
        variations.append({
            'query': f"fine art landscape photography {' '.join(keywords[:1] if keywords else '')}",
            'score': 0.8
        })
        variations.append({
            'query': f"professional nature photography {' '.join(keywords[:1] if keywords else '')}",
            'score': 0.75
        })
        
        # Ensure we have at least 3 query variations
        if len(variations) < 3:
            variations.append({
                'query': f"beautiful photography {' '.join(keywords)}",
                'score': 0.7
            })
            
        return variations

class RetrievalAgent:
    """
    Agent for retrieving content from Instagram based on refined queries.
    Implements the Retrieval Agent role from the Multi-agent Retrieval Protocol.
    """
    
    def __init__(self, base_dir: str = 'data'):
        """
        Initialize the retrieval agent.
        
        Args:
            base_dir: Base directory for storing retrieved content
        """
        self.base_dir = base_dir
        
    def retrieve_content(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve content from Instagram based on the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of retrieved content items
        """
        # Convert query to Instagram profile search
        # This is a simplified implementation - in production, this would
        # use more sophisticated techniques to find relevant Instagram profiles
        
        # For now, we'll use the configured profiles and pretend we're searching
        logger.info(f"Searching Instagram for content matching: {query}")
        
        # In a real implementation, this would search for profiles based on the query
        # For now, we'll use the configured profiles from config
        if not config.INSTAGRAM_TARGET_PROFILES:
            logger.warning("No Instagram profiles configured for retrieval")
            return []
            
        # Use a subset of configured profiles (simulating search results)
        profiles_to_search = config.INSTAGRAM_TARGET_PROFILES
        
        # Use the Instagram scraper to get posts
        try:
            posts = process_instagram_posts(
                profile_urls=profiles_to_search,
                max_posts=max_results * 2,  # Get more than we need for filtering
                landscape_only=True,
                base_dir=self.base_dir
            )
            
            # Extract relevant information and add retrieval metadata
            results = []
            for post in posts:
                # Add retrieval metadata
                post['retrieval_query'] = query
                post['retrieval_timestamp'] = time.time()
                post['initial_score'] = random.uniform(0.7, 1.0)  # Simplified scoring
                
                results.append(post)
                
            logger.info(f"Retrieved {len(results)} results from Instagram")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving content from Instagram: {e}")
            return []

class RerankerAgent:
    """
    Agent for reranking and filtering retrieved content based on relevance and quality.
    Implements the Reranker Agent role from the Multi-agent Retrieval Protocol.
    """
    
    def __init__(self):
        """Initialize the reranker agent."""
        pass
        
    def rerank_results(self, results: List[Dict[str, Any]], original_query: str) -> List[Dict[str, Any]]:
        """
        Rerank and filter results based on relevance to the original query.
        
        Args:
            results: List of retrieved content items
            original_query: The original user query
            
        Returns:
            Reranked list of content items
        """
        if not results:
            return []
            
        # Analyze query for keywords
        query_keywords = set(original_query.lower().split())
        
        # Score each result
        for result in results:
            # Start with the initial score from retrieval
            score = result.get('initial_score', 0.5)
            
            # Adjust based on query relevance
            if 'caption' in result:
                caption = result['caption'].lower() if result.get('caption') else ""
                caption_words = set(caption.split())
                
                # Calculate overlap between caption and query keywords
                overlap = query_keywords.intersection(caption_words)
                score += len(overlap) * 0.05
                
            # Adjust based on hashtags
            if 'hashtags' in result:
                hashtags = result.get('hashtags', [])
                for hashtag in hashtags:
                    if any(keyword in hashtag.lower() for keyword in query_keywords):
                        score += 0.03
                        
            # Adjust based on engagement metrics
            likes = result.get('likes', 0)
            comments = result.get('comments', 0)
            
            # Simple engagement score - more sophisticated in production
            engagement_score = min((likes + comments * 3) / 1000, 0.2)
            score += engagement_score
            
            # Prefer landscape images
            if result.get('is_landscape', False):
                score += 0.1
                
            # Adjust based on image quality if available
            if 'image_metadata' in result:
                metadata = result.get('image_metadata', {})
                
                # Higher resolution images get a boost
                width = metadata.get('width', 0)
                height = metadata.get('height', 0)
                resolution_score = min((width * height) / (1920 * 1080 * 4), 0.15)
                score += resolution_score
                
            # Cap the score at 1.0
            score = min(score, 1.0)
            
            # Add the quality score to the result
            result['quality_score'] = score
            
        # Sort by quality score
        ranked_results = sorted(results, key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # Remove duplicates based on image similarity (simplified)
        # In production, this would use more sophisticated image similarity checks
        deduplicated_results = []
        seen_shortcodes = set()
        
        for result in ranked_results:
            shortcode = result.get('shortcode', '')
            
            if shortcode and shortcode not in seen_shortcodes:
                seen_shortcodes.add(shortcode)
                deduplicated_results.append(result)
                
        logger.info(f"Reranked {len(results)} results to {len(deduplicated_results)} deduplicated results")
        return deduplicated_results

class SummarizationAgent:
    """
    Agent for generating metadata and descriptions for Etsy listings.
    Implements the Summarization Agent role from the Multi-agent Retrieval Protocol.
    """
    
    def __init__(self):
        """Initialize the summarization agent."""
        pass
        
    def generate_metadata(self, content_item: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Generate Etsy-optimized metadata for a content item.
        
        Args:
            content_item: The content item to generate metadata for
            query: The original search query
            
        Returns:
            Dictionary of Etsy-optimized metadata
        """
        # Extract existing metadata
        caption = content_item.get('caption', '')
        hashtags = content_item.get('hashtags', [])
        location = content_item.get('location', 'Beautiful Location')
        
        # Generate title
        title_keywords = [
            'Fine Art Print',
            'Landscape Photography',
            'Wall Art',
            'Home Decor',
            'Nature Print'
        ]
        
        # Use location if available
        if location and location != 'Beautiful Location':
            title = f"{location} - Fine Art Landscape Photography Print - Wall Art"
        else:
            # Extract potential title from caption
            caption_words = caption.split()[:10] if caption else []
            caption_excerpt = ' '.join(caption_words)
            
            # Fallback title
            title = f"Landscape Photography Wall Art Print - Fine Art Nature Print"
            
            # Use caption excerpt if it's substantial
            if len(caption_excerpt) > 20:
                title = f"{caption_excerpt} - Fine Art Landscape Print"
                
        # Generate description
        description = f"Beautiful landscape photography fine art print"
        
        if location and location != 'Beautiful Location':
            description += f" of {location}"
            
        description += ". Perfect for home decor, office spaces, or as a thoughtful gift. "
        description += "This premium quality print captures the beauty of nature with vibrant colors and exceptional detail.\n\n"
        
        # Add more details if we have a caption
        if caption and len(caption) > 30:
            description += f"About this image:\n{caption}\n\n"
            
        description += "Available in multiple sizes and materials to fit your space.\n\n"
        description += "• Printed on premium fine art paper with archival inks for vibrant colors and detail\n"
        description += "• Available as canvas prints and framed prints\n"
        description += "• Each print is made to order\n"
        description += "• Ships within 2-5 business days\n\n"
        description += "Note: Frame not included unless selected as an option."
        
        # Generate SEO tags
        tags = []
        
        # Add hashtags from Instagram (remove # symbol)
        hashtag_tags = [tag.replace('#', '').lower() for tag in hashtags]
        tags.extend(hashtag_tags[:5])  # Use up to 5 hashtags
        
        # Add standard tags
        standard_tags = [
            'landscape photography',
            'wall art',
            'fine art print',
            'home decor',
            'nature print',
            'photography print',
            'wall decor'
        ]
        
        # Add location-based tag if available
        if location and location != 'Beautiful Location':
            tags.append(location.lower())
            
        # Combine and deduplicate tags
        all_tags = list(set(tags + standard_tags))
        
        # Limit to 13 tags (Etsy maximum)
        final_tags = all_tags[:13]
        
        return {
            'title': title,
            'description': description,
            'tags': final_tags
        }
