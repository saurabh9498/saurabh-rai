"""
Analyst Agent.

Specialized agent for data analysis, synthesis,
visualization recommendations, and structured reasoning.
"""

from typing import List, Dict, Any, Optional, Union
import json
import logging
from dataclasses import dataclass

from .base_agent import BaseAgent, Tool
from ..core.context import ConversationContext


logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Structured analysis result."""
    summary: str
    insights: List[str]
    recommendations: List[str]
    data_points: Dict[str, Any]
    confidence: float
    methodology: str


class AnalystAgent(BaseAgent):
    """
    Agent specialized in analysis tasks including:
    - Data analysis and interpretation
    - Trend identification
    - Comparative analysis
    - Structured reasoning
    - Visualization recommendations
    """
    
    def __init__(
        self,
        name: str = "analyst_agent",
        database_connector=None,
        **kwargs
    ):
        self.database_connector = database_connector
        
        super().__init__(
            name=name,
            description="Analyst agent for data analysis and synthesis",
            **kwargs
        )
        
        self._register_default_tools()
    
    def _default_system_prompt(self) -> str:
        return """You are an Analyst Agent specialized in data analysis and structured reasoning.

Your capabilities:
1. Data Analysis: Analyze datasets and extract insights
2. Trend Analysis: Identify patterns and trends over time
3. Comparative Analysis: Compare entities, metrics, or approaches
4. Synthesis: Combine multiple data sources into coherent insights
5. Visualization: Recommend appropriate visualizations

Guidelines:
- Be precise with numbers and statistics
- Distinguish correlation from causation
- Acknowledge limitations in data or analysis
- Provide actionable recommendations
- Structure analysis with clear methodology

When given an analysis task:
1. Clarify the analysis objective
2. Identify required data sources
3. Choose appropriate methodology
4. Execute analysis step by step
5. Synthesize findings with confidence levels
6. Provide recommendations

Respond with structured analysis including methodology, findings, and confidence levels."""
    
    def _register_default_tools(self) -> None:
        """Register the default analyst tools."""
        
        # Data Query Tool
        self.register_tool(Tool(
            name="query_data",
            description="Query structured data from databases",
            function=self._query_data,
            parameters={
                "query": {"type": "string", "description": "SQL or natural language query"},
                "database": {"type": "string", "description": "Target database", "default": "default"}
            },
            required_params=["query"]
        ))
        
        # Statistical Analysis Tool
        self.register_tool(Tool(
            name="statistical_analysis",
            description="Perform statistical analysis on data",
            function=self._statistical_analysis,
            parameters={
                "data": {"type": "object", "description": "Data to analyze"},
                "analysis_type": {"type": "string", "description": "Type of analysis (descriptive, correlation, regression)"}
            },
            required_params=["data", "analysis_type"]
        ))
        
        # Trend Analysis Tool
        self.register_tool(Tool(
            name="trend_analysis",
            description="Analyze trends in time series data",
            function=self._trend_analysis,
            parameters={
                "data": {"type": "object", "description": "Time series data"},
                "period": {"type": "string", "description": "Analysis period"}
            },
            required_params=["data"]
        ))
        
        # Comparison Tool
        self.register_tool(Tool(
            name="compare",
            description="Compare multiple entities or metrics",
            function=self._compare,
            parameters={
                "items": {"type": "array", "description": "Items to compare"},
                "criteria": {"type": "array", "description": "Comparison criteria"}
            },
            required_params=["items"]
        ))
    
    async def _execute_task(
        self,
        task: str,
        context: ConversationContext,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute an analysis task.
        
        Args:
            task: The analysis question or task
            context: Conversation context
            data: Optional data to analyze
            
        Returns:
            Analysis results
        """
        # Step 1: Understand the analysis request
        analysis_plan = await self._plan_analysis(task, data)
        self._log_action("analysis_plan", {"plan": analysis_plan})
        
        # Step 2: Execute analysis based on type
        analysis_type = analysis_plan.get("type", "general")
        
        if analysis_type == "comparative":
            result = await self._execute_comparative_analysis(task, data, analysis_plan)
        elif analysis_type == "trend":
            result = await self._execute_trend_analysis(task, data, analysis_plan)
        elif analysis_type == "statistical":
            result = await self._execute_statistical_analysis(task, data, analysis_plan)
        else:
            result = await self._execute_general_analysis(task, data, analysis_plan)
        
        # Step 3: Generate recommendations
        recommendations = await self._generate_recommendations(task, result)
        result["recommendations"] = recommendations
        
        # Step 4: Suggest visualizations
        visualizations = await self._suggest_visualizations(task, result)
        result["suggested_visualizations"] = visualizations
        
        return result
    
    async def _plan_analysis(
        self,
        task: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Plan the analysis approach."""
        data_description = json.dumps(list(data.keys())) if data else "No data provided"
        
        prompt = f"""Analysis Task: {task}

Available Data: {data_description}

Plan this analysis by determining:
1. Analysis type (comparative, trend, statistical, general)
2. Key metrics to analyze
3. Methodology to use
4. Expected outputs

Respond in JSON format:
{{
    "type": "comparative|trend|statistical|general",
    "metrics": ["metric1", "metric2"],
    "methodology": "Description of approach",
    "expected_outputs": ["output1", "output2"]
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "type": "general",
                "metrics": [],
                "methodology": "General analysis",
                "expected_outputs": ["summary", "insights"]
            }
    
    async def _execute_comparative_analysis(
        self,
        task: str,
        data: Optional[Dict[str, Any]],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a comparative analysis."""
        prompt = f"""Perform a comparative analysis for: {task}

Data: {json.dumps(data) if data else 'None provided'}

Analysis Plan: {json.dumps(plan)}

Provide a structured comparison including:
1. Items being compared
2. Comparison criteria
3. Analysis for each criterion
4. Overall comparison summary
5. Winner/recommendation (if applicable)

Respond in JSON format:
{{
    "items_compared": ["item1", "item2"],
    "criteria": ["criterion1", "criterion2"],
    "analysis": {{
        "criterion1": {{"item1": "analysis", "item2": "analysis"}},
        "criterion2": {{"item1": "analysis", "item2": "analysis"}}
    }},
    "summary": "Overall comparison summary",
    "recommendation": "Recommended choice with reasoning",
    "confidence": 0.85
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"summary": response, "confidence": 0.5}
    
    async def _execute_trend_analysis(
        self,
        task: str,
        data: Optional[Dict[str, Any]],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a trend analysis."""
        prompt = f"""Perform a trend analysis for: {task}

Data: {json.dumps(data) if data else 'None provided'}

Analysis Plan: {json.dumps(plan)}

Identify and analyze:
1. Current trends
2. Trend direction (increasing, decreasing, stable)
3. Rate of change
4. Seasonality or patterns
5. Projected future trends

Respond in JSON format:
{{
    "trends": [
        {{"name": "trend1", "direction": "increasing", "strength": 0.8}}
    ],
    "patterns": ["pattern1", "pattern2"],
    "projections": {{"short_term": "projection", "long_term": "projection"}},
    "key_findings": ["finding1", "finding2"],
    "confidence": 0.75
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"summary": response, "confidence": 0.5}
    
    async def _execute_statistical_analysis(
        self,
        task: str,
        data: Optional[Dict[str, Any]],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a statistical analysis."""
        prompt = f"""Perform a statistical analysis for: {task}

Data: {json.dumps(data) if data else 'None provided'}

Analysis Plan: {json.dumps(plan)}

Calculate and report:
1. Descriptive statistics (mean, median, std, etc.)
2. Relevant statistical tests
3. Correlations (if applicable)
4. Significance levels
5. Key statistical insights

Respond in JSON format:
{{
    "descriptive_stats": {{"mean": 0, "median": 0, "std": 0}},
    "correlations": [{{"variables": ["x", "y"], "coefficient": 0.8}}],
    "tests_performed": [{{"test": "t-test", "result": "significant", "p_value": 0.01}}],
    "insights": ["insight1", "insight2"],
    "confidence": 0.9
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"summary": response, "confidence": 0.5}
    
    async def _execute_general_analysis(
        self,
        task: str,
        data: Optional[Dict[str, Any]],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a general analysis."""
        prompt = f"""Perform an analysis for: {task}

Data: {json.dumps(data) if data else 'None provided'}

Analysis Plan: {json.dumps(plan)}

Provide a comprehensive analysis including:
1. Key findings
2. Supporting evidence
3. Limitations
4. Confidence level

Respond in JSON format:
{{
    "summary": "Executive summary",
    "key_findings": ["finding1", "finding2"],
    "evidence": ["evidence1", "evidence2"],
    "limitations": ["limitation1"],
    "confidence": 0.8
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"summary": response, "confidence": 0.5}
    
    async def _generate_recommendations(
        self,
        task: str,
        analysis_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis."""
        prompt = f"""Based on this analysis:

Task: {task}
Results: {json.dumps(analysis_result)}

Generate 3-5 actionable recommendations that:
1. Address the original question
2. Are specific and measurable
3. Have clear priorities

Respond in JSON format:
{{
    "recommendations": [
        {{"priority": 1, "action": "action", "rationale": "why", "expected_impact": "impact"}}
    ]
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages)
        
        try:
            result = json.loads(response)
            return result.get("recommendations", [])
        except json.JSONDecodeError:
            return []
    
    async def _suggest_visualizations(
        self,
        task: str,
        analysis_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Suggest appropriate visualizations for the analysis."""
        prompt = f"""Based on this analysis:

Task: {task}
Results: {json.dumps(analysis_result)}

Suggest 2-3 appropriate visualizations that would best communicate these findings.

Respond in JSON format:
{{
    "visualizations": [
        {{
            "type": "bar_chart|line_chart|scatter_plot|pie_chart|heatmap|table",
            "title": "Chart title",
            "description": "What it shows",
            "data_requirements": ["required_field1", "required_field2"]
        }}
    ]
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages)
        
        try:
            result = json.loads(response)
            return result.get("visualizations", [])
        except json.JSONDecodeError:
            return []
    
    # Tool implementations
    async def _query_data(self, query: str, database: str = "default") -> Dict[str, Any]:
        """Query data from a database."""
        if self.database_connector:
            return await self.database_connector.query(query, database)
        return {"data": [], "message": "No database connector configured"}
    
    async def _statistical_analysis(
        self,
        data: Dict[str, Any],
        analysis_type: str
    ) -> Dict[str, Any]:
        """Perform statistical analysis on data."""
        # In production, use numpy/scipy for actual calculations
        return {
            "analysis_type": analysis_type,
            "results": "Statistical analysis placeholder"
        }
    
    async def _trend_analysis(
        self,
        data: Dict[str, Any],
        period: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze trends in time series data."""
        return {
            "period": period,
            "trends": "Trend analysis placeholder"
        }
    
    async def _compare(
        self,
        items: List[Any],
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple items."""
        return {
            "items": items,
            "criteria": criteria,
            "comparison": "Comparison placeholder"
        }
