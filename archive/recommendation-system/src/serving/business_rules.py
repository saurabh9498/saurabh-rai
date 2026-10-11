"""
Business Rules Engine

Post-ranking filters and adjustments for recommendations:
- Eligibility filtering (inventory, geo, age restrictions)
- Score boosting/penalizing
- Diversity enforcement
- Freshness boosting
"""

from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class RuleResult:
    """Result of applying a business rule."""
    item_id: str
    passed: bool
    score_modifier: float = 1.0
    reason: Optional[str] = None


@dataclass 
class FilteredItem:
    """Item after filtering with metadata."""
    item_id: str
    original_score: float
    final_score: float
    rank: int
    filters_passed: List[str]
    filters_failed: List[str]
    boost_reasons: List[str]


class BusinessRule:
    """Base class for business rules."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
    
    def apply(
        self,
        item_id: str,
        item_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> RuleResult:
        """Apply rule to an item."""
        raise NotImplementedError


class InStockFilter(BusinessRule):
    """Filter out of stock items."""
    
    def __init__(self, enabled: bool = True):
        super().__init__("in_stock_filter", enabled)
    
    def apply(
        self,
        item_id: str,
        item_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> RuleResult:
        in_stock = item_data.get('in_stock', True)
        return RuleResult(
            item_id=item_id,
            passed=in_stock,
            reason="Out of stock" if not in_stock else None,
        )


class GeoRestrictionFilter(BusinessRule):
    """Filter items not available in user's region."""
    
    def __init__(self, enabled: bool = True):
        super().__init__("geo_restriction_filter", enabled)
    
    def apply(
        self,
        item_id: str,
        item_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> RuleResult:
        user_country = context.get('country')
        restricted_countries = item_data.get('restricted_countries', [])
        available_countries = item_data.get('available_countries', [])
        
        if user_country in restricted_countries:
            return RuleResult(
                item_id=item_id,
                passed=False,
                reason=f"Not available in {user_country}",
            )
        
        if available_countries and user_country not in available_countries:
            return RuleResult(
                item_id=item_id,
                passed=False,
                reason=f"Not available in {user_country}",
            )
        
        return RuleResult(item_id=item_id, passed=True)


class AgeRestrictionFilter(BusinessRule):
    """Filter age-restricted items."""
    
    def __init__(self, enabled: bool = True):
        super().__init__("age_restriction_filter", enabled)
    
    def apply(
        self,
        item_id: str,
        item_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> RuleResult:
        min_age = item_data.get('min_age', 0)
        user_age = context.get('user_age', 99)
        
        if user_age < min_age:
            return RuleResult(
                item_id=item_id,
                passed=False,
                reason=f"Age restricted (min: {min_age})",
            )
        
        return RuleResult(item_id=item_id, passed=True)


class RecentPurchaseFilter(BusinessRule):
    """Filter items recently purchased by user."""
    
    def __init__(self, days: int = 30, enabled: bool = True):
        super().__init__("recent_purchase_filter", enabled)
        self.days = days
    
    def apply(
        self,
        item_id: str,
        item_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> RuleResult:
        recent_purchases = context.get('recent_purchases', [])
        
        if item_id in recent_purchases:
            return RuleResult(
                item_id=item_id,
                passed=False,
                reason=f"Purchased in last {self.days} days",
            )
        
        return RuleResult(item_id=item_id, passed=True)


class NewItemBoost(BusinessRule):
    """Boost score for new items."""
    
    def __init__(
        self,
        days_threshold: int = 7,
        boost_factor: float = 1.2,
        enabled: bool = True,
    ):
        super().__init__("new_item_boost", enabled)
        self.days_threshold = days_threshold
        self.boost_factor = boost_factor
    
    def apply(
        self,
        item_id: str,
        item_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> RuleResult:
        days_since_listed = item_data.get('days_since_listed', 999)
        
        if days_since_listed <= self.days_threshold:
            return RuleResult(
                item_id=item_id,
                passed=True,
                score_modifier=self.boost_factor,
                reason="New item boost",
            )
        
        return RuleResult(item_id=item_id, passed=True)


class DiscountBoost(BusinessRule):
    """Boost score for discounted items."""
    
    def __init__(
        self,
        min_discount: float = 0.2,
        boost_factor: float = 1.1,
        enabled: bool = True,
    ):
        super().__init__("discount_boost", enabled)
        self.min_discount = min_discount
        self.boost_factor = boost_factor
    
    def apply(
        self,
        item_id: str,
        item_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> RuleResult:
        discount_pct = item_data.get('discount_pct', 0)
        
        if discount_pct >= self.min_discount:
            return RuleResult(
                item_id=item_id,
                passed=True,
                score_modifier=self.boost_factor,
                reason=f"Discount boost ({discount_pct:.0%} off)",
            )
        
        return RuleResult(item_id=item_id, passed=True)


class LowStockPenalty(BusinessRule):
    """Penalize items with low stock."""
    
    def __init__(
        self,
        threshold: int = 5,
        penalty_factor: float = 0.8,
        enabled: bool = True,
    ):
        super().__init__("low_stock_penalty", enabled)
        self.threshold = threshold
        self.penalty_factor = penalty_factor
    
    def apply(
        self,
        item_id: str,
        item_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> RuleResult:
        inventory = item_data.get('inventory_level', 999)
        
        if 0 < inventory <= self.threshold:
            return RuleResult(
                item_id=item_id,
                passed=True,
                score_modifier=self.penalty_factor,
                reason="Low stock penalty",
            )
        
        return RuleResult(item_id=item_id, passed=True)


class CategoryDiversityEnforcer(BusinessRule):
    """Enforce category diversity in results."""
    
    def __init__(
        self,
        max_per_category: int = 3,
        enabled: bool = True,
    ):
        super().__init__("category_diversity", enabled)
        self.max_per_category = max_per_category
        self._category_counts: Dict[int, int] = {}
    
    def reset(self):
        """Reset category counts for new request."""
        self._category_counts = {}
    
    def apply(
        self,
        item_id: str,
        item_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> RuleResult:
        category = item_data.get('category_l2', 0)
        
        current_count = self._category_counts.get(category, 0)
        
        if current_count >= self.max_per_category:
            return RuleResult(
                item_id=item_id,
                passed=False,
                reason=f"Category limit reached ({self.max_per_category})",
            )
        
        # Update count
        self._category_counts[category] = current_count + 1
        
        return RuleResult(item_id=item_id, passed=True)


class BusinessRulesEngine:
    """Orchestrates business rules for recommendations."""
    
    def __init__(self):
        self.filters: List[BusinessRule] = []
        self.boosters: List[BusinessRule] = []
        self.diversity_rules: List[BusinessRule] = []
    
    def add_filter(self, rule: BusinessRule):
        """Add a filtering rule."""
        self.filters.append(rule)
        return self
    
    def add_booster(self, rule: BusinessRule):
        """Add a score boosting rule."""
        self.boosters.append(rule)
        return self
    
    def add_diversity_rule(self, rule: BusinessRule):
        """Add a diversity enforcement rule."""
        self.diversity_rules.append(rule)
        return self
    
    def apply(
        self,
        items: List[Dict[str, Any]],
        item_data: Dict[str, Dict[str, Any]],
        context: Dict[str, Any],
        max_items: int = 50,
    ) -> List[FilteredItem]:
        """
        Apply all business rules to candidate items.
        
        Args:
            items: List of {item_id, score} dicts
            item_data: Dict mapping item_id to item attributes
            context: Request context (user info, etc.)
            max_items: Maximum items to return
            
        Returns:
            List of FilteredItem with applied rules
        """
        # Reset stateful rules
        for rule in self.diversity_rules:
            if hasattr(rule, 'reset'):
                rule.reset()
        
        results = []
        
        for item in items:
            item_id = item['item_id']
            original_score = item['score']
            data = item_data.get(item_id, {})
            
            # Apply filters
            filters_passed = []
            filters_failed = []
            passed_all_filters = True
            
            for rule in self.filters:
                if not rule.enabled:
                    continue
                    
                result = rule.apply(item_id, data, context)
                
                if result.passed:
                    filters_passed.append(rule.name)
                else:
                    filters_failed.append(f"{rule.name}: {result.reason}")
                    passed_all_filters = False
                    break  # Stop on first failed filter
            
            if not passed_all_filters:
                continue
            
            # Apply diversity rules
            for rule in self.diversity_rules:
                if not rule.enabled:
                    continue
                    
                result = rule.apply(item_id, data, context)
                
                if not result.passed:
                    filters_failed.append(f"{rule.name}: {result.reason}")
                    passed_all_filters = False
                    break
            
            if not passed_all_filters:
                continue
            
            # Apply boosters
            score_modifier = 1.0
            boost_reasons = []
            
            for rule in self.boosters:
                if not rule.enabled:
                    continue
                    
                result = rule.apply(item_id, data, context)
                score_modifier *= result.score_modifier
                
                if result.reason:
                    boost_reasons.append(result.reason)
            
            final_score = original_score * score_modifier
            
            results.append(FilteredItem(
                item_id=item_id,
                original_score=original_score,
                final_score=final_score,
                rank=0,  # Will be set after sorting
                filters_passed=filters_passed,
                filters_failed=filters_failed,
                boost_reasons=boost_reasons,
            ))
            
            if len(results) >= max_items:
                break
        
        # Sort by final score and assign ranks
        results.sort(key=lambda x: x.final_score, reverse=True)
        for i, item in enumerate(results):
            item.rank = i + 1
        
        return results


def create_default_engine() -> BusinessRulesEngine:
    """Create engine with default rules."""
    engine = BusinessRulesEngine()
    
    # Filters
    engine.add_filter(InStockFilter())
    engine.add_filter(GeoRestrictionFilter())
    engine.add_filter(AgeRestrictionFilter())
    engine.add_filter(RecentPurchaseFilter(days=30))
    
    # Diversity
    engine.add_diversity_rule(CategoryDiversityEnforcer(max_per_category=5))
    
    # Boosters
    engine.add_booster(NewItemBoost(days_threshold=7, boost_factor=1.2))
    engine.add_booster(DiscountBoost(min_discount=0.2, boost_factor=1.1))
    engine.add_booster(LowStockPenalty(threshold=5, penalty_factor=0.8))
    
    return engine
