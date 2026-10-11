"""
Chain Prompts Example

Demonstrates prompt chaining and sequential agent workflows.
Shows how to build complex pipelines by connecting multiple LLM calls.
"""

import asyncio
import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import system components
from src.core.llm import LLMClient
from src.utils import (
    setup_logging,
    get_logger,
    token_tracker,
    log_execution_time,
)

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


@dataclass
class ChainStep:
    """Represents a step in a prompt chain."""
    name: str
    prompt_template: str
    model: str = "gpt-4o-mini"
    max_tokens: int = 500
    temperature: float = 0.7
    parser: Optional[Callable[[str], Any]] = None
    
    def format_prompt(self, **kwargs) -> str:
        """Format the prompt template with provided variables."""
        return self.prompt_template.format(**kwargs)


class PromptChain:
    """
    Executes a sequence of prompts, passing outputs between steps.
    
    Supports:
    - Sequential execution
    - Variable passing between steps
    - Custom output parsers
    - Error handling and retries
    """
    
    def __init__(self, steps: List[ChainStep]):
        self.steps = steps
        self.client = LLMClient(provider="openai", model="gpt-4o-mini")
        self.logger = get_logger("PromptChain")
        self.results: Dict[str, Any] = {}
    
    async def run(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the chain with initial input.
        
        Returns dict with all intermediate and final results.
        """
        self.results = {"input": initial_input}
        current_vars = initial_input.copy()
        
        for i, step in enumerate(self.steps):
            self.logger.info(f"Executing step {i+1}/{len(self.steps)}: {step.name}")
            
            try:
                # Format prompt with current variables
                prompt = step.format_prompt(**current_vars)
                
                # Get completion
                response = await self.client.complete(
                    prompt=prompt,
                    model=step.model,
                    max_tokens=step.max_tokens,
                    temperature=step.temperature,
                )
                
                # Parse output if parser provided
                if step.parser:
                    parsed = step.parser(response)
                else:
                    parsed = response.strip()
                
                # Store result and update variables
                self.results[step.name] = {
                    "prompt": prompt,
                    "raw_output": response,
                    "parsed_output": parsed,
                }
                current_vars[step.name] = parsed
                current_vars["previous_output"] = parsed
                
            except Exception as e:
                self.logger.error(f"Step {step.name} failed: {e}")
                self.results[step.name] = {"error": str(e)}
                raise
        
        self.results["final_output"] = current_vars.get("previous_output")
        return self.results
    
    def get_result(self, step_name: str) -> Any:
        """Get the result of a specific step."""
        return self.results.get(step_name, {}).get("parsed_output")


async def research_and_summarize_chain():
    """
    Chain that researches a topic and creates a summary.
    
    Steps:
    1. Generate research questions
    2. Create detailed notes
    3. Synthesize into summary
    4. Create bullet points
    """
    print("\n" + "="*60)
    print("RESEARCH AND SUMMARIZE CHAIN")
    print("="*60)
    
    steps = [
        ChainStep(
            name="questions",
            prompt_template="""Generate 3 focused research questions about: {topic}

Output only the questions, one per line.""",
            max_tokens=200,
        ),
        ChainStep(
            name="research_notes",
            prompt_template="""Based on these research questions:
{questions}

Write detailed notes answering each question about {topic}.
Include facts, explanations, and examples.""",
            max_tokens=800,
        ),
        ChainStep(
            name="summary",
            prompt_template="""Based on these research notes:
{research_notes}

Write a comprehensive 2-paragraph summary about {topic}.""",
            max_tokens=400,
        ),
        ChainStep(
            name="bullet_points",
            prompt_template="""Convert this summary into 5 key bullet points:

{summary}

Format: Start each bullet with "•" """,
            max_tokens=300,
        ),
    ]
    
    chain = PromptChain(steps)
    results = await chain.run({"topic": "the impact of artificial intelligence on healthcare"})
    
    # Display results
    print(f"\n📚 Topic: {results['input']['topic']}")
    print(f"\n📝 Research Questions:\n{chain.get_result('questions')}")
    print(f"\n📋 Summary:\n{chain.get_result('summary')}")
    print(f"\n🎯 Key Points:\n{chain.get_result('bullet_points')}")
    
    return results


async def content_creation_chain():
    """
    Chain for creating marketing content.
    
    Steps:
    1. Analyze product
    2. Identify target audience
    3. Generate headlines
    4. Write copy
    """
    print("\n" + "="*60)
    print("CONTENT CREATION CHAIN")
    print("="*60)
    
    steps = [
        ChainStep(
            name="product_analysis",
            prompt_template="""Analyze this product for marketing purposes:
Product: {product_name}
Description: {product_description}

Identify:
1. Key features
2. Unique value proposition
3. Main benefits""",
            max_tokens=300,
        ),
        ChainStep(
            name="target_audience",
            prompt_template="""Based on this product analysis:
{product_analysis}

Define the ideal target audience:
1. Demographics
2. Pain points
3. Goals and motivations""",
            max_tokens=300,
        ),
        ChainStep(
            name="headlines",
            prompt_template="""Create 5 compelling headlines for this product.

Product Analysis:
{product_analysis}

Target Audience:
{target_audience}

Generate headlines that:
- Grab attention
- Highlight benefits
- Speak to the target audience""",
            max_tokens=200,
            temperature=0.9,
        ),
        ChainStep(
            name="marketing_copy",
            prompt_template="""Write persuasive marketing copy using the best elements from these headlines:
{headlines}

Target Audience:
{target_audience}

Write a 100-word marketing paragraph that converts.""",
            max_tokens=200,
        ),
    ]
    
    chain = PromptChain(steps)
    results = await chain.run({
        "product_name": "SmartFocus Pro",
        "product_description": "An AI-powered productivity app that blocks distractions, schedules deep work sessions, and provides insights on your focus patterns.",
    })
    
    # Display results
    print(f"\n🎯 Target Audience:\n{chain.get_result('target_audience')}")
    print(f"\n📰 Headlines:\n{chain.get_result('headlines')}")
    print(f"\n✍️ Marketing Copy:\n{chain.get_result('marketing_copy')}")
    
    return results


async def code_review_chain():
    """
    Chain for reviewing and improving code.
    
    Steps:
    1. Analyze code
    2. Identify issues
    3. Suggest improvements
    4. Generate improved version
    """
    print("\n" + "="*60)
    print("CODE REVIEW CHAIN")
    print("="*60)
    
    sample_code = '''
def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] != None:
            result.append(data[i] * 2)
    return result
'''
    
    steps = [
        ChainStep(
            name="analysis",
            prompt_template="""Analyze this Python code:
```python
{code}
```

Describe:
1. What the code does
2. Code style observations
3. Potential performance considerations""",
            max_tokens=300,
        ),
        ChainStep(
            name="issues",
            prompt_template="""Based on this analysis:
{analysis}

List specific issues with the code:
- Style issues
- Potential bugs
- Performance problems
- Best practice violations""",
            max_tokens=300,
        ),
        ChainStep(
            name="improvements",
            prompt_template="""For these identified issues:
{issues}

Suggest specific improvements with explanations.""",
            max_tokens=300,
        ),
        ChainStep(
            name="improved_code",
            prompt_template="""Apply these improvements:
{improvements}

To the original code and provide the improved version.

Original:
```python
{code}
```

Output only the improved code in a Python code block.""",
            max_tokens=300,
        ),
    ]
    
    chain = PromptChain(steps)
    results = await chain.run({"code": sample_code})
    
    # Display results
    print(f"\n📝 Original Code:{sample_code}")
    print(f"\n🔍 Issues Found:\n{chain.get_result('issues')}")
    print(f"\n✨ Improved Code:\n{chain.get_result('improved_code')}")
    
    return results


async def parallel_chain_demo():
    """
    Demonstrate running chains in parallel.
    """
    print("\n" + "="*60)
    print("PARALLEL CHAINS DEMO")
    print("="*60)
    
    # Simple parallel chains for different aspects
    async def analyze_sentiment(text: str) -> str:
        client = LLMClient(provider="openai", model="gpt-4o-mini")
        return await client.complete(
            prompt=f"Analyze the sentiment of this text in one word (positive/negative/neutral): {text}",
            max_tokens=10,
        )
    
    async def extract_entities(text: str) -> str:
        client = LLMClient(provider="openai", model="gpt-4o-mini")
        return await client.complete(
            prompt=f"List the key entities (people, places, organizations) in this text: {text}",
            max_tokens=100,
        )
    
    async def summarize(text: str) -> str:
        client = LLMClient(provider="openai", model="gpt-4o-mini")
        return await client.complete(
            prompt=f"Summarize this in one sentence: {text}",
            max_tokens=100,
        )
    
    text = """Apple announced today that CEO Tim Cook will present the company's 
    new Vision Pro headset at their Cupertino headquarters. The revolutionary 
    device has generated excitement among tech enthusiasts and investors alike, 
    with analysts predicting strong sales in the coming quarter."""
    
    print(f"\n📄 Input Text:\n{text}")
    
    # Run analyses in parallel
    results = await asyncio.gather(
        analyze_sentiment(text),
        extract_entities(text),
        summarize(text),
    )
    
    print(f"\n😊 Sentiment: {results[0].strip()}")
    print(f"\n🏷️ Entities: {results[1].strip()}")
    print(f"\n📝 Summary: {results[2].strip()}")
    
    return results


async def main():
    """Run all chain examples."""
    print("\n" + "#"*60)
    print("# MULTI-AGENT SYSTEM - PROMPT CHAIN EXAMPLES")
    print("#"*60)
    
    try:
        # Run chain demos
        await research_and_summarize_chain()
        await content_creation_chain()
        await code_review_chain()
        await parallel_chain_demo()
        
        # Print final stats
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        
        stats = token_tracker.get_stats()
        print(f"\nChain Execution Summary:")
        print(f"  Total API calls: {stats.total_requests}")
        print(f"  Total tokens: {stats.total_tokens:,}")
        print(f"  Total cost: ${stats.total_cost:.6f}")
        
        if stats.by_model:
            print(f"\nBy Model:")
            for model, model_stats in stats.by_model.items():
                print(f"  {model}: {model_stats['requests']} calls, "
                      f"{model_stats['total_tokens']:,} tokens")
        
        print("="*60 + "\n")
        
    except Exception as e:
        logger.exception(f"Chain demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
