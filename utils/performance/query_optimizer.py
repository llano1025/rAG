"""
Query optimization system for vector operations and database queries.
Provides query caching, result caching, query planning, and performance monitoring.
"""

import asyncio
import logging
import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import numpy as np

from utils.caching.redis_manager import get_redis_manager
from utils.caching.cache_strategy import CacheStrategy, CacheConfig

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries for optimization."""
    VECTOR_SIMILARITY = "vector_similarity"
    DOCUMENT_SEARCH = "document_search"
    METADATA_FILTER = "metadata_filter"
    HYBRID_SEARCH = "hybrid_search"
    BATCH_SEARCH = "batch_search"
    DATABASE_QUERY = "database_query"

@dataclass
class QueryPlan:
    """Query execution plan."""
    query_id: str
    query_type: QueryType
    estimated_cost: float
    cache_strategy: str
    use_index: bool
    parallel_execution: bool
    batch_size: Optional[int] = None
    timeout_seconds: int = 30
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class QueryStats:
    """Query execution statistics."""
    query_id: str
    query_type: QueryType
    execution_time: float
    cache_hit: bool
    result_count: int
    memory_usage_mb: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class QueryOptimizer:
    """
    Advanced query optimizer for vector and database operations.
    
    Features:
    - Query result caching with TTL
    - Query plan optimization
    - Automatic index usage
    - Batch query optimization
    - Performance monitoring
    - Adaptive query strategies
    """
    
    def __init__(
        self,
        cache_ttl_seconds: int = 3600,
        max_cache_size_mb: int = 500,
        enable_query_planning: bool = True,
        enable_statistics: bool = True
    ):
        """
        Initialize query optimizer.
        
        Args:
            cache_ttl_seconds: Default cache TTL
            max_cache_size_mb: Maximum cache size in MB
            enable_query_planning: Enable query plan optimization
            enable_statistics: Enable query statistics collection
        """
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_cache_size_mb = max_cache_size_mb
        self.enable_query_planning = enable_query_planning
        self.enable_statistics = enable_statistics
        
        # Cache configuration
        self.cache_config = CacheConfig(
            strategy=CacheStrategy.SLIDING,
            ttl_seconds=cache_ttl_seconds,
            max_size=max_cache_size_mb * 1024 * 1024
        )
        
        # Redis manager for caching
        self.redis_manager = get_redis_manager()
        
        # Query plans and statistics
        self.query_plans: Dict[str, QueryPlan] = {}
        self.query_stats: List[QueryStats] = []
        
        # Performance metrics
        self.metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_execution_time': 0.0,
            'total_execution_time': 0.0,
            'query_types': {}
        }
        
        # Index usage tracking
        self.index_usage: Dict[str, int] = {}
        
        # Adaptive thresholds
        self.similarity_threshold = 0.7
        self.batch_size_threshold = 100
    
    async def optimize_vector_search(
        self,
        query_vector: List[float],
        index_name: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        user_id: Optional[int] = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Optimize vector similarity search with caching and planning.
        
        Args:
            query_vector: Query vector
            index_name: Vector index name
            top_k: Number of results to return
            filters: Optional metadata filters
            user_id: User ID for access control
            force_refresh: Force cache refresh
            
        Returns:
            Optimized search results
        """
        start_time = time.time()
        
        # Generate query ID for caching
        query_id = self._generate_query_id(
            QueryType.VECTOR_SIMILARITY,
            query_vector=query_vector,
            index_name=index_name,
            top_k=top_k,
            filters=filters,
            user_id=user_id
        )
        
        # Check cache first
        if not force_refresh:
            cached_result = await self._get_cached_result(query_id)
            if cached_result is not None:
                await self._record_query_stats(
                    query_id, QueryType.VECTOR_SIMILARITY,
                    time.time() - start_time, True, len(cached_result.get('results', []))
                )
                return cached_result
        
        # Create query plan
        query_plan = await self._create_query_plan(
            QueryType.VECTOR_SIMILARITY,
            query_id=query_id,
            parameters={
                'index_name': index_name,
                'top_k': top_k,
                'filters': filters,
                'vector_dim': len(query_vector)
            }
        )
        
        try:
            # Execute optimized search
            if query_plan.parallel_execution and filters:
                # Parallel execution for filtered searches
                result = await self._execute_parallel_vector_search(
                    query_vector, index_name, top_k, filters, user_id, query_plan
                )
            else:
                # Standard execution
                result = await self._execute_vector_search(
                    query_vector, index_name, top_k, filters, user_id
                )
            
            # Cache the result
            await self._cache_result(query_id, result, self.cache_ttl_seconds)
            
            # Record statistics
            execution_time = time.time() - start_time
            await self._record_query_stats(
                query_id, QueryType.VECTOR_SIMILARITY,
                execution_time, False, len(result.get('results', []))
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Vector search optimization failed: {e}")
            # Fallback to basic search
            return await self._execute_vector_search(
                query_vector, index_name, top_k, filters, user_id
            )
    
    async def optimize_batch_search(
        self,
        queries: List[Dict[str, Any]],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Optimize batch search operations.
        
        Args:
            queries: List of search queries
            batch_size: Batch processing size
            
        Returns:
            List of search results
        """
        start_time = time.time()
        
        if not queries:
            return []
        
        # Determine optimal batch size
        optimal_batch_size = batch_size or self._calculate_optimal_batch_size(len(queries))
        
        # Group queries by similarity to optimize caching
        query_groups = await self._group_similar_queries(queries)
        
        # Process batches
        all_results = []
        
        for group in query_groups:
            # Process each group in batches
            for i in range(0, len(group), optimal_batch_size):
                batch = group[i:i + optimal_batch_size]
                
                # Check cache for batch
                cached_results, uncached_queries = await self._check_batch_cache(batch)
                all_results.extend(cached_results)
                
                # Process uncached queries
                if uncached_queries:
                    batch_results = await self._process_query_batch(uncached_queries)
                    all_results.extend(batch_results)
                    
                    # Cache batch results
                    await self._cache_batch_results(uncached_queries, batch_results)
        
        # Record batch statistics
        execution_time = time.time() - start_time
        await self._record_query_stats(
            f"batch_{int(time.time())}", QueryType.BATCH_SEARCH,
            execution_time, False, len(all_results)
        )
        
        return all_results
    
    async def optimize_database_query(
        self,
        query_func: Callable,
        query_params: Dict[str, Any],
        cache_key: Optional[str] = None,
        ttl_seconds: Optional[int] = None
    ) -> Any:
        """
        Optimize database queries with caching.
        
        Args:
            query_func: Database query function
            query_params: Query parameters
            cache_key: Custom cache key
            ttl_seconds: Cache TTL override
            
        Returns:
            Query result
        """
        start_time = time.time()
        
        # Generate cache key
        if cache_key is None:
            cache_key = self._generate_query_id(
                QueryType.DATABASE_QUERY,
                **query_params
            )
        
        # Check cache
        cached_result = await self._get_cached_result(cache_key)
        if cached_result is not None:
            await self._record_query_stats(
                cache_key, QueryType.DATABASE_QUERY,
                time.time() - start_time, True, 1
            )
            return cached_result
        
        try:
            # Execute query
            if asyncio.iscoroutinefunction(query_func):
                result = await query_func(**query_params)
            else:
                result = query_func(**query_params)
            
            # Cache result
            cache_ttl = ttl_seconds or self.cache_ttl_seconds
            await self._cache_result(cache_key, result, cache_ttl)
            
            # Record statistics
            execution_time = time.time() - start_time
            await self._record_query_stats(
                cache_key, QueryType.DATABASE_QUERY,
                execution_time, False, 1
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Database query optimization failed: {e}")
            raise
    
    async def get_query_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get query performance statistics for the specified time period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Filter recent statistics
        recent_stats = [
            stat for stat in self.query_stats
            if stat.created_at >= cutoff_time
        ]
        
        if not recent_stats:
            return {
                'total_queries': 0,
                'cache_hit_rate': 0.0,
                'average_execution_time': 0.0,
                'query_types': {},
                'performance_trends': []
            }
        
        # Calculate metrics
        total_queries = len(recent_stats)
        cache_hits = sum(1 for stat in recent_stats if stat.cache_hit)
        cache_hit_rate = cache_hits / total_queries * 100
        
        avg_execution_time = sum(stat.execution_time for stat in recent_stats) / total_queries
        
        # Query type breakdown
        query_types = {}
        for stat in recent_stats:
            query_type = stat.query_type.value
            if query_type not in query_types:
                query_types[query_type] = {
                    'count': 0,
                    'avg_time': 0.0,
                    'cache_hit_rate': 0.0
                }
            
            query_types[query_type]['count'] += 1
        
        # Calculate per-type metrics
        for query_type in query_types:
            type_stats = [stat for stat in recent_stats if stat.query_type.value == query_type]
            query_types[query_type]['avg_time'] = sum(stat.execution_time for stat in type_stats) / len(type_stats)
            query_types[query_type]['cache_hit_rate'] = sum(1 for stat in type_stats if stat.cache_hit) / len(type_stats) * 100
        
        # Performance trends (hourly breakdown)
        performance_trends = self._calculate_performance_trends(recent_stats, hours)
        
        return {
            'total_queries': total_queries,
            'cache_hit_rate': cache_hit_rate,
            'average_execution_time': avg_execution_time,
            'query_types': query_types,
            'performance_trends': performance_trends,
            'index_usage': dict(self.index_usage),
            'optimization_suggestions': await self._generate_optimization_suggestions()
        }
    
    async def clear_cache(self, pattern: Optional[str] = None):
        """Clear query cache with optional pattern matching."""
        if pattern:
            # Clear specific cache entries
            await self.redis_manager.delete_pattern(f"query_cache:{pattern}*")
        else:
            # Clear all query cache
            await self.redis_manager.delete_pattern("query_cache:*")
        
        logger.info(f"Cleared query cache with pattern: {pattern or 'all'}")
    
    async def optimize_cache_usage(self):
        """Optimize cache usage based on query patterns."""
        stats = await self.get_query_statistics(hours=24)
        
        # Adjust cache TTL based on query patterns
        if stats['cache_hit_rate'] < 30:
            # Low hit rate, increase TTL
            self.cache_ttl_seconds = min(self.cache_ttl_seconds * 1.5, 7200)  # Max 2 hours
        elif stats['cache_hit_rate'] > 80:
            # High hit rate, could reduce TTL for fresher data
            self.cache_ttl_seconds = max(self.cache_ttl_seconds * 0.8, 300)  # Min 5 minutes
        
        # Adjust batch size based on performance
        avg_batch_time = stats['query_types'].get('batch_search', {}).get('avg_time', 0)
        if avg_batch_time > 5.0:  # Too slow
            self.batch_size_threshold = max(self.batch_size_threshold // 2, 10)
        elif avg_batch_time < 1.0:  # Could handle more
            self.batch_size_threshold = min(self.batch_size_threshold * 1.5, 500)
        
        logger.info(f"Optimized cache: TTL={self.cache_ttl_seconds}s, batch_size={self.batch_size_threshold}")
    
    def _generate_query_id(self, query_type: QueryType, **kwargs) -> str:
        """Generate unique query ID for caching."""
        # Create deterministic hash from query parameters
        query_data = {
            'type': query_type.value,
            **kwargs
        }
        
        # Convert to JSON with sorted keys for consistency
        query_json = json.dumps(query_data, sort_keys=True, default=str)
        
        # Generate hash
        query_hash = hashlib.sha256(query_json.encode()).hexdigest()[:16]
        
        return f"{query_type.value}_{query_hash}"
    
    async def _get_cached_result(self, query_id: str) -> Optional[Any]:
        """Get cached query result."""
        try:
            cache_key = f"query_cache:{query_id}"
            cached_data = await self.redis_manager.get_json(cache_key)
            
            if cached_data:
                self.metrics['cache_hits'] += 1
                return cached_data
            else:
                self.metrics['cache_misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            self.metrics['cache_misses'] += 1
            return None
    
    async def _cache_result(self, query_id: str, result: Any, ttl_seconds: int):
        """Cache query result."""
        try:
            cache_key = f"query_cache:{query_id}"
            await self.redis_manager.set_json(cache_key, result, ttl_seconds)
        except Exception as e:
            logger.error(f"Cache storage failed: {e}")
    
    async def _create_query_plan(
        self,
        query_type: QueryType,
        query_id: str,
        parameters: Dict[str, Any]
    ) -> QueryPlan:
        """Create optimized query execution plan."""
        if not self.enable_query_planning:
            return QueryPlan(
                query_id=query_id,
                query_type=query_type,
                estimated_cost=1.0,
                cache_strategy="simple",
                use_index=True,
                parallel_execution=False
            )
        
        # Estimate query cost
        estimated_cost = self._estimate_query_cost(query_type, parameters)
        
        # Determine optimization strategies
        use_parallel = (
            query_type in [QueryType.VECTOR_SIMILARITY, QueryType.HYBRID_SEARCH] and
            estimated_cost > 2.0
        )
        
        batch_size = None
        if query_type == QueryType.BATCH_SEARCH:
            batch_size = self._calculate_optimal_batch_size(
                parameters.get('query_count', 10)
            )
        
        plan = QueryPlan(
            query_id=query_id,
            query_type=query_type,
            estimated_cost=estimated_cost,
            cache_strategy="sliding" if estimated_cost > 1.5 else "simple",
            use_index=True,
            parallel_execution=use_parallel,
            batch_size=batch_size
        )
        
        self.query_plans[query_id] = plan
        return plan
    
    def _estimate_query_cost(self, query_type: QueryType, parameters: Dict[str, Any]) -> float:
        """Estimate query execution cost."""
        base_costs = {
            QueryType.VECTOR_SIMILARITY: 1.0,
            QueryType.DOCUMENT_SEARCH: 0.8,
            QueryType.METADATA_FILTER: 0.5,
            QueryType.HYBRID_SEARCH: 2.0,
            QueryType.BATCH_SEARCH: 3.0,
            QueryType.DATABASE_QUERY: 0.6
        }
        
        base_cost = base_costs.get(query_type, 1.0)
        
        # Adjust based on parameters
        if query_type == QueryType.VECTOR_SIMILARITY:
            top_k = parameters.get('top_k', 10)
            vector_dim = parameters.get('vector_dim', 768)
            filters = parameters.get('filters')
            
            # Higher dimension = higher cost
            cost_multiplier = 1.0 + (vector_dim / 1000)
            
            # More results = higher cost
            cost_multiplier *= 1.0 + (top_k / 100)
            
            # Filters add cost
            if filters:
                cost_multiplier *= 1.5
            
            return base_cost * cost_multiplier
        
        return base_cost
    
    def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """Calculate optimal batch size based on total items and system capacity."""
        if total_items <= 10:
            return total_items
        
        # Base batch size on system threshold and total items
        if total_items <= 50:
            return min(10, self.batch_size_threshold)
        elif total_items <= 200:
            return min(20, self.batch_size_threshold)
        else:
            return min(50, self.batch_size_threshold)
    
    async def _execute_vector_search(
        self,
        query_vector: List[float],
        index_name: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        user_id: Optional[int]
    ) -> Dict[str, Any]:
        """Execute basic vector search (placeholder - integrate with actual vector search)."""
        # This would integrate with your actual vector search implementation
        # For now, return a placeholder structure
        
        await asyncio.sleep(0.1)  # Simulate search time
        
        return {
            'results': [
                {
                    'id': f'doc_{i}',
                    'score': 0.9 - (i * 0.1),
                    'content': f'Sample document {i}',
                    'metadata': {'source': f'file_{i}.txt'}
                }
                for i in range(min(top_k, 5))
            ],
            'total_count': min(top_k, 5),
            'query_time': 0.1
        }
    
    async def _execute_parallel_vector_search(
        self,
        query_vector: List[float],
        index_name: str,
        top_k: int,
        filters: Dict[str, Any],
        user_id: Optional[int],
        query_plan: QueryPlan
    ) -> Dict[str, Any]:
        """Execute parallel vector search with filters."""
        # Split filters for parallel execution
        filter_groups = self._split_filters_for_parallel(filters)
        
        # Execute searches in parallel
        tasks = []
        for filter_group in filter_groups:
            task = self._execute_vector_search(
                query_vector, index_name, top_k, filter_group, user_id
            )
            tasks.append(task)
        
        # Wait for all searches to complete
        results = await asyncio.gather(*tasks)
        
        # Merge and rank results
        merged_results = self._merge_search_results(results, top_k)
        
        return merged_results
    
    def _split_filters_for_parallel(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split filters for parallel execution."""
        # Simple implementation - could be more sophisticated
        if len(filters) <= 2:
            return [filters]
        
        # Split filters into groups
        filter_items = list(filters.items())
        mid = len(filter_items) // 2
        
        return [
            dict(filter_items[:mid]),
            dict(filter_items[mid:])
        ]
    
    def _merge_search_results(self, results: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
        """Merge and rank parallel search results."""
        all_results = []
        total_time = 0.0
        
        for result in results:
            all_results.extend(result.get('results', []))
            total_time += result.get('query_time', 0.0)
        
        # Sort by score and take top_k
        all_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        final_results = all_results[:top_k]
        
        return {
            'results': final_results,
            'total_count': len(final_results),
            'query_time': max(result.get('query_time', 0.0) for result in results)
        }
    
    async def _group_similar_queries(self, queries: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group similar queries for batch optimization."""
        # Simple grouping by query type
        groups = {}
        
        for query in queries:
            query_type = query.get('type', 'unknown')
            if query_type not in groups:
                groups[query_type] = []
            groups[query_type].append(query)
        
        return list(groups.values())
    
    async def _check_batch_cache(
        self,
        queries: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Check cache for batch queries."""
        cached_results = []
        uncached_queries = []
        
        for query in queries:
            query_id = self._generate_query_id(
                QueryType.VECTOR_SIMILARITY,  # Default type
                **query
            )
            
            cached_result = await self._get_cached_result(query_id)
            if cached_result:
                cached_results.append(cached_result)
            else:
                uncached_queries.append(query)
        
        return cached_results, uncached_queries
    
    async def _process_query_batch(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of uncached queries."""
        tasks = []
        
        for query in queries:
            # Process each query (placeholder implementation)
            task = self._execute_vector_search(
                query.get('vector', []),
                query.get('index_name', 'default'),
                query.get('top_k', 10),
                query.get('filters'),
                query.get('user_id')
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def _cache_batch_results(
        self,
        queries: List[Dict[str, Any]],
        results: List[Dict[str, Any]]
    ):
        """Cache batch query results."""
        for query, result in zip(queries, results):
            query_id = self._generate_query_id(
                QueryType.VECTOR_SIMILARITY,
                **query
            )
            await self._cache_result(query_id, result, self.cache_ttl_seconds)
    
    async def _record_query_stats(
        self,
        query_id: str,
        query_type: QueryType,
        execution_time: float,
        cache_hit: bool,
        result_count: int
    ):
        """Record query execution statistics."""
        if not self.enable_statistics:
            return
        
        # Create query stats
        stats = QueryStats(
            query_id=query_id,
            query_type=query_type,
            execution_time=execution_time,
            cache_hit=cache_hit,
            result_count=result_count,
            memory_usage_mb=0.0  # Could add memory tracking
        )
        
        self.query_stats.append(stats)
        
        # Update global metrics
        self.metrics['total_queries'] += 1
        self.metrics['total_execution_time'] += execution_time
        self.metrics['average_execution_time'] = (
            self.metrics['total_execution_time'] / self.metrics['total_queries']
        )
        
        query_type_key = query_type.value
        if query_type_key not in self.metrics['query_types']:
            self.metrics['query_types'][query_type_key] = 0
        self.metrics['query_types'][query_type_key] += 1
        
        # Limit stats history
        if len(self.query_stats) > 10000:
            self.query_stats = self.query_stats[-5000:]  # Keep recent 5000
    
    def _calculate_performance_trends(
        self,
        stats: List[QueryStats],
        hours: int
    ) -> List[Dict[str, Any]]:
        """Calculate hourly performance trends."""
        if not stats:
            return []
        
        # Group stats by hour
        hourly_stats = {}
        
        for stat in stats:
            hour_key = stat.created_at.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_stats:
                hourly_stats[hour_key] = []
            hourly_stats[hour_key].append(stat)
        
        # Calculate trends
        trends = []
        for hour, hour_stats in sorted(hourly_stats.items()):
            avg_time = sum(s.execution_time for s in hour_stats) / len(hour_stats)
            cache_hit_rate = sum(1 for s in hour_stats if s.cache_hit) / len(hour_stats) * 100
            
            trends.append({
                'hour': hour.isoformat(),
                'query_count': len(hour_stats),
                'avg_execution_time': avg_time,
                'cache_hit_rate': cache_hit_rate
            })
        
        return trends
    
    async def _generate_optimization_suggestions(self) -> List[str]:
        """Generate optimization suggestions based on statistics."""
        suggestions = []
        
        if self.metrics['cache_hits'] + self.metrics['cache_misses'] > 0:
            hit_rate = self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses']) * 100
            
            if hit_rate < 50:
                suggestions.append("Consider increasing cache TTL or optimizing query patterns")
            
            if self.metrics['average_execution_time'] > 2.0:
                suggestions.append("Consider enabling parallel execution for complex queries")
            
            if self.metrics['query_types'].get('batch_search', 0) > 100:
                suggestions.append("High batch query volume - consider optimizing batch sizes")
        
        return suggestions


# Global query optimizer instance
_query_optimizer: Optional[QueryOptimizer] = None

def get_query_optimizer() -> QueryOptimizer:
    """Get global query optimizer instance."""
    global _query_optimizer
    
    if _query_optimizer is None:
        _query_optimizer = QueryOptimizer()
    
    return _query_optimizer