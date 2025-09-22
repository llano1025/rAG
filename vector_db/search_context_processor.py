# /vector_db/context_processor.py

from typing import Optional, List, Dict, Any
import re
from dataclasses import dataclass

@dataclass
class ProcessedContext:
    text: str
    metadata: Dict
    weights: Dict[str, float]

@dataclass
class ContextAnalysis:
    """Enhanced context analysis result."""
    enhanced_query: str
    temporal_context: Dict[str, Any]
    technical_context: Dict[str, Any]
    intent_context: Dict[str, Any]
    specificity_context: Dict[str, Any]
    confidence: float
    recommended_strategy: str
    dynamic_weights: Dict[str, float]

class ContextProcessor:
    """Enhanced query context processor with intelligent pattern recognition."""

    def __init__(self):
        # Expanded pattern recognition for better query understanding
        self.context_patterns = {
            # Temporal context - enhanced
            'time': r'\b(today|yesterday|last\s+\w+|Q[1-4]\s+\d{4}|\d{4}|recent|latest|current|this\s+\w+|past\s+\w+)\b',

            # Technical context
            'technical': r'\b(API|function|method|class|algorithm|implementation|code|programming|config|configuration|settings|key|token|auth|authentication)\b',

            # Document types
            'doc_type': r'\b(pdf|document|report|manual|specification|contract|agreement|policy|procedure|guide)\b',

            # Intent patterns - enhanced
            'intent': r'\b(find|search|locate|show|list|compare|analyze|summarize|explain|describe|understand|what|how|why|when|where|who)\b',

            # Specificity indicators
            'specific': r'\b([A-Z_]{2,}|[a-z]+_[a-z]+|\d+\.\d+|version\s+\d+|v\d+|\w+@\w+|https?://)\b',

            # Relationship context
            'relation': r'\b(related\s+to|similar\s+to|different\s+from|compared\s+to|versus|vs|like|unlike)\b',

            # Scope context
            'scope': r'\b(all|any|every|most|some|few|specific|particular|certain|exact|precisely)\b',

            # Action context - enhanced
            'action': r'\b(how\s+to|what\s+is|define|explain|describe|configure|setup|install|create|build|deploy|troubleshoot)\b'
        }

    def process_context(
        self,
        query: str,
        context: Optional[str] = None
    ) -> str:
        """Process and enhance query context (backward compatibility)."""
        if not context:
            # Extract implicit context from query
            context = self._extract_implicit_context(query)

        # Combine explicit and implicit context
        combined_context = self._combine_context(query, context)

        # Clean and normalize
        cleaned_context = self._clean_context(combined_context)

        return cleaned_context

    def analyze_query_context(self, query: str) -> ContextAnalysis:
        """
        Enhanced context analysis for intelligent search optimization.
        Provides deep understanding of query intent and characteristics.
        """
        try:
            # Extract different types of context
            temporal_context = self._extract_temporal_context(query)
            technical_context = self._extract_technical_context(query)
            intent_context = self._extract_intent_context(query)
            specificity_context = self._extract_specificity_context(query)

            # Calculate dynamic weights based on context
            dynamic_weights = self._calculate_dynamic_weights(
                temporal_context, technical_context, intent_context, specificity_context
            )

            # Generate enhanced query
            enhanced_query = self._enhance_query_text(
                query, temporal_context, technical_context, intent_context
            )

            # Determine recommended search strategy
            recommended_strategy = self._recommend_search_strategy(
                temporal_context, technical_context, intent_context, specificity_context
            )

            # Calculate confidence in the analysis
            confidence = self._calculate_confidence_score(
                temporal_context, technical_context, intent_context, specificity_context
            )

            return ContextAnalysis(
                enhanced_query=enhanced_query,
                temporal_context=temporal_context,
                technical_context=technical_context,
                intent_context=intent_context,
                specificity_context=specificity_context,
                confidence=confidence,
                recommended_strategy=recommended_strategy,
                dynamic_weights=dynamic_weights
            )

        except Exception as e:
            # Fallback to basic processing
            return ContextAnalysis(
                enhanced_query=query,
                temporal_context={},
                technical_context={},
                intent_context={},
                specificity_context={},
                confidence=0.5,
                recommended_strategy='semantic',
                dynamic_weights={
                    'semantic_weight': 0.7,
                    'lexical_weight': 0.3,
                    'content_weight': 0.7,
                    'context_weight': 0.3
                }
            )

    def _extract_implicit_context(self, query: str) -> str:
        """Extract context implicitly from query."""
        contexts = []
        
        for context_type, pattern in self.context_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                contexts.extend(matches)
        
        return " ".join(str(c) for c in contexts)

    def _combine_context(self, query: str, context: str) -> str:
        """Combine query and context information."""
        # Remove any duplicate information
        context_words = set(context.lower().split())
        query_words = set(query.lower().split())
        
        unique_context_words = context_words - query_words
        
        # Reconstruct context with unique information
        combined = f"{query} {' '.join(unique_context_words)}"
        return combined.strip()

    def _clean_context(self, context: str) -> str:
        """Clean and normalize context text."""
        # Remove extra whitespace
        context = re.sub(r'\s+', ' ', context)
        
        # Remove special characters
        context = re.sub(r'[^\w\s]', ' ', context)
        
        # Normalize to lowercase
        context = context.lower().strip()
        
        return context

    def _extract_temporal_context(self, query: str) -> Dict[str, Any]:
        """Extract temporal context from query."""
        temporal_matches = re.findall(self.context_patterns['time'], query, re.IGNORECASE)
        return {
            'has_temporal': bool(temporal_matches),
            'temporal_terms': temporal_matches,
            'is_recent_focused': any(term in query.lower() for term in ['recent', 'latest', 'current', 'today'])
        }

    def _extract_technical_context(self, query: str) -> Dict[str, Any]:
        """Extract technical context from query."""
        technical_matches = re.findall(self.context_patterns['technical'], query, re.IGNORECASE)
        specific_matches = re.findall(self.context_patterns['specific'], query, re.IGNORECASE)

        return {
            'has_technical': bool(technical_matches),
            'technical_terms': technical_matches,
            'has_specific_terms': bool(specific_matches),
            'specific_terms': specific_matches,
            'is_code_related': any(term in query.lower() for term in ['function', 'method', 'class', 'code']),
            'is_config_related': any(term in query.lower() for term in ['config', 'settings', 'key', 'token'])
        }

    def _extract_intent_context(self, query: str) -> Dict[str, Any]:
        """Extract intent context from query."""
        intent_matches = re.findall(self.context_patterns['intent'], query, re.IGNORECASE)
        action_matches = re.findall(self.context_patterns['action'], query, re.IGNORECASE)

        intent_type = 'search'
        if any(term in query.lower() for term in ['how', 'explain', 'describe']):
            intent_type = 'explanation'
        elif any(term in query.lower() for term in ['find', 'search', 'locate']):
            intent_type = 'search'
        elif any(term in query.lower() for term in ['compare', 'versus', 'different']):
            intent_type = 'comparison'

        return {
            'has_intent': bool(intent_matches),
            'intent_terms': intent_matches,
            'action_terms': action_matches,
            'intent_type': intent_type,
            'is_question': query.strip().endswith('?') or query.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who'))
        }

    def _extract_specificity_context(self, query: str) -> Dict[str, Any]:
        """Extract specificity indicators from query."""
        scope_matches = re.findall(self.context_patterns['scope'], query, re.IGNORECASE)
        specific_matches = re.findall(self.context_patterns['specific'], query, re.IGNORECASE)

        return {
            'has_scope': bool(scope_matches),
            'scope_terms': scope_matches,
            'specific_terms': specific_matches,
            'is_broad': any(term in query.lower() for term in ['all', 'any', 'every']),
            'is_narrow': any(term in query.lower() for term in ['specific', 'particular', 'exact'])
        }

    def _calculate_dynamic_weights(
        self,
        temporal_context: Dict[str, Any],
        technical_context: Dict[str, Any],
        intent_context: Dict[str, Any],
        specificity_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate dynamic search weights based on context analysis."""
        # Start with default weights
        weights = {
            'semantic_weight': 0.7,
            'lexical_weight': 0.3,
            'content_weight': 0.7,
            'context_weight': 0.3
        }

        # Adjust based on technical content
        if technical_context.get('has_technical') or technical_context.get('has_specific_terms'):
            weights['lexical_weight'] += 0.2  # Boost exact term matching
            weights['semantic_weight'] -= 0.2

        # Adjust based on specificity
        if specificity_context.get('is_narrow') or specificity_context.get('specific_terms'):
            weights['lexical_weight'] += 0.15
            weights['semantic_weight'] -= 0.15

        # Adjust based on intent (explanation queries need more context)
        if intent_context.get('intent_type') == 'explanation':
            weights['context_weight'] += 0.2
            weights['content_weight'] -= 0.2

        # Ensure weights don't go below minimum thresholds
        weights['semantic_weight'] = max(0.3, weights['semantic_weight'])
        weights['lexical_weight'] = max(0.1, weights['lexical_weight'])
        weights['content_weight'] = max(0.3, weights['content_weight'])
        weights['context_weight'] = max(0.1, weights['context_weight'])

        # Normalize to ensure they sum to 1.0 for each pair
        total_search = weights['semantic_weight'] + weights['lexical_weight']
        weights['semantic_weight'] /= total_search
        weights['lexical_weight'] /= total_search

        total_content = weights['content_weight'] + weights['context_weight']
        weights['content_weight'] /= total_content
        weights['context_weight'] /= total_content

        return weights

    def _enhance_query_text(
        self,
        query: str,
        temporal_context: Dict[str, Any],
        technical_context: Dict[str, Any],
        intent_context: Dict[str, Any]
    ) -> str:
        """Generate enhanced query text for better search results."""
        enhanced_parts = [query]

        # Add technical context terms for better matching
        if technical_context.get('technical_terms'):
            # Don't duplicate terms already in query
            query_lower = query.lower()
            new_terms = [term for term in technical_context['technical_terms']
                        if term.lower() not in query_lower]
            enhanced_parts.extend(new_terms[:2])  # Limit to 2 additional terms

        # Clean and return enhanced query
        enhanced_query = ' '.join(enhanced_parts)
        return self._clean_context(enhanced_query)

    def _recommend_search_strategy(
        self,
        temporal_context: Dict[str, Any],
        technical_context: Dict[str, Any],
        intent_context: Dict[str, Any],
        specificity_context: Dict[str, Any]
    ) -> str:
        """Recommend optimal search strategy based on context analysis."""

        # Technical + specific terms suggest hybrid
        if (technical_context.get('has_technical') and
            (specificity_context.get('specific_terms') or technical_context.get('has_specific_terms'))):
            return 'hybrid'

        # Explanation queries with technical content
        if intent_context.get('intent_type') == 'explanation' and technical_context.get('has_technical'):
            return 'hybrid'

        # Broad semantic queries
        if (intent_context.get('is_question') and
            not technical_context.get('has_technical') and
            specificity_context.get('is_broad')):
            return 'semantic'

        # Specific search queries
        if (intent_context.get('intent_type') == 'search' and
            (specificity_context.get('is_narrow') or specificity_context.get('specific_terms'))):
            return 'keyword'

        # Default to contextual (enhanced)
        return 'contextual'

    def _calculate_confidence_score(
        self,
        temporal_context: Dict[str, Any],
        technical_context: Dict[str, Any],
        intent_context: Dict[str, Any],
        specificity_context: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the context analysis."""
        confidence = 0.5  # Base confidence

        # Boost confidence for clear patterns
        if intent_context.get('has_intent'):
            confidence += 0.2

        if technical_context.get('has_technical'):
            confidence += 0.15

        if specificity_context.get('specific_terms'):
            confidence += 0.1

        if temporal_context.get('has_temporal'):
            confidence += 0.05

        # Ensure confidence stays within bounds
        return min(1.0, max(0.1, confidence))