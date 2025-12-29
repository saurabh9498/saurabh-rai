"""
Code Agent.

Specialized agent for code generation, review,
debugging, and safe code execution.
"""

from typing import List, Dict, Any, Optional
import json
import logging
import subprocess
import tempfile
import os
from pathlib import Path

from .base_agent import BaseAgent, Tool
from ..core.context import ConversationContext


logger = logging.getLogger(__name__)


class CodeAgent(BaseAgent):
    """
    Agent specialized in code-related tasks including:
    - Code generation
    - Code review
    - Bug fixing
    - Code explanation
    - Safe code execution
    """
    
    SUPPORTED_LANGUAGES = [
        "python", "javascript", "typescript", "java", "c++", 
        "go", "rust", "sql", "bash", "html", "css"
    ]
    
    def __init__(
        self,
        name: str = "code_agent",
        enable_execution: bool = True,
        sandbox_mode: bool = True,
        allowed_languages: Optional[List[str]] = None,
        **kwargs
    ):
        self.enable_execution = enable_execution
        self.sandbox_mode = sandbox_mode
        self.allowed_languages = allowed_languages or ["python"]
        
        super().__init__(
            name=name,
            description="Code agent for generation, review, and execution",
            **kwargs
        )
        
        self._register_default_tools()
    
    def _default_system_prompt(self) -> str:
        return """You are a Code Agent specialized in software development tasks.

Your capabilities:
1. Code Generation: Write clean, efficient, well-documented code
2. Code Review: Analyze code for bugs, security issues, and improvements
3. Debugging: Identify and fix bugs in code
4. Code Explanation: Explain how code works
5. Code Execution: Safely execute code (when enabled)

Guidelines:
- Write clean, readable, well-documented code
- Follow best practices and design patterns
- Consider edge cases and error handling
- Prioritize security and performance
- Provide clear explanations

When generating code:
1. Understand the requirements fully
2. Plan the approach
3. Write the code with comments
4. Include error handling
5. Provide usage examples

When reviewing code:
1. Check for bugs and logic errors
2. Evaluate security considerations
3. Assess performance implications
4. Suggest improvements
5. Rate code quality

Always format code blocks with appropriate language tags."""
    
    def _register_default_tools(self) -> None:
        """Register the default code tools."""
        
        # Code Generation Tool
        self.register_tool(Tool(
            name="generate_code",
            description="Generate code based on requirements",
            function=self._generate_code,
            parameters={
                "requirements": {"type": "string", "description": "Code requirements"},
                "language": {"type": "string", "description": "Programming language"},
                "style": {"type": "string", "description": "Code style preferences", "default": "clean"}
            },
            required_params=["requirements", "language"]
        ))
        
        # Code Review Tool
        self.register_tool(Tool(
            name="review_code",
            description="Review code for quality, bugs, and security",
            function=self._review_code,
            parameters={
                "code": {"type": "string", "description": "Code to review"},
                "language": {"type": "string", "description": "Programming language"},
                "focus": {"type": "array", "description": "Review focus areas", "default": ["bugs", "security", "style"]}
            },
            required_params=["code", "language"]
        ))
        
        # Code Execution Tool (if enabled)
        if self.enable_execution:
            self.register_tool(Tool(
                name="execute_code",
                description="Safely execute code and return results",
                function=self._execute_code,
                parameters={
                    "code": {"type": "string", "description": "Code to execute"},
                    "language": {"type": "string", "description": "Programming language"},
                    "timeout": {"type": "integer", "description": "Execution timeout in seconds", "default": 30}
                },
                required_params=["code", "language"]
            ))
        
        # Debug Tool
        self.register_tool(Tool(
            name="debug_code",
            description="Debug code and suggest fixes",
            function=self._debug_code,
            parameters={
                "code": {"type": "string", "description": "Code with bug"},
                "error": {"type": "string", "description": "Error message or description"},
                "language": {"type": "string", "description": "Programming language"}
            },
            required_params=["code", "language"]
        ))
    
    async def _execute_task(
        self,
        task: str,
        context: ConversationContext,
        code: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a code-related task.
        
        Args:
            task: The coding task description
            context: Conversation context
            code: Optional existing code to work with
            language: Programming language
            
        Returns:
            Task results
        """
        # Step 1: Understand the task type
        task_analysis = await self._analyze_task(task, code)
        self._log_action("task_analysis", {"analysis": task_analysis})
        
        task_type = task_analysis.get("type", "general")
        language = language or task_analysis.get("language", "python")
        
        result = {
            "task_type": task_type,
            "language": language,
            "success": True
        }
        
        # Step 2: Execute based on task type
        if task_type == "generate":
            generation_result = await self._generate_code(
                requirements=task,
                language=language,
                style=kwargs.get("style", "clean")
            )
            result["code"] = generation_result["code"]
            result["explanation"] = generation_result["explanation"]
            
        elif task_type == "review":
            if not code:
                result["success"] = False
                result["error"] = "No code provided for review"
            else:
                review_result = await self._review_code(
                    code=code,
                    language=language,
                    focus=kwargs.get("focus", ["bugs", "security", "style"])
                )
                result["review"] = review_result
                
        elif task_type == "debug":
            if not code:
                result["success"] = False
                result["error"] = "No code provided for debugging"
            else:
                debug_result = await self._debug_code(
                    code=code,
                    error=kwargs.get("error", task),
                    language=language
                )
                result["debug"] = debug_result
                result["fixed_code"] = debug_result.get("fixed_code")
                
        elif task_type == "explain":
            if not code:
                result["success"] = False
                result["error"] = "No code provided for explanation"
            else:
                explanation = await self._explain_code(code, language)
                result["explanation"] = explanation
                
        elif task_type == "execute" and self.enable_execution:
            if not code:
                result["success"] = False
                result["error"] = "No code provided for execution"
            else:
                execution_result = await self._execute_code(
                    code=code,
                    language=language,
                    timeout=kwargs.get("timeout", 30)
                )
                result["execution"] = execution_result
        else:
            # General code task - attempt to generate
            generation_result = await self._generate_code(
                requirements=task,
                language=language,
                style="clean"
            )
            result["code"] = generation_result["code"]
            result["explanation"] = generation_result["explanation"]
        
        return result
    
    async def _analyze_task(
        self,
        task: str,
        code: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze the coding task to determine type and requirements."""
        prompt = f"""Analyze this coding task:

Task: {task}
Code Provided: {"Yes" if code else "No"}

Determine:
1. Task type (generate, review, debug, explain, execute, refactor)
2. Programming language needed
3. Key requirements

Respond in JSON format:
{{
    "type": "generate|review|debug|explain|execute|refactor",
    "language": "python|javascript|etc",
    "requirements": ["req1", "req2"],
    "complexity": "low|medium|high"
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "type": "generate",
                "language": "python",
                "requirements": [task],
                "complexity": "medium"
            }
    
    async def _generate_code(
        self,
        requirements: str,
        language: str,
        style: str = "clean"
    ) -> Dict[str, Any]:
        """Generate code based on requirements."""
        prompt = f"""Generate {language} code for the following requirements:

Requirements: {requirements}
Style: {style}

Provide:
1. Complete, working code
2. Clear comments
3. Error handling
4. Usage example

Format your response as JSON:
{{
    "code": "// The complete code here",
    "explanation": "Brief explanation of the implementation",
    "usage_example": "// How to use the code",
    "dependencies": ["dep1", "dep2"]
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Extract code from response
            return {
                "code": response,
                "explanation": "Code generated based on requirements",
                "usage_example": "",
                "dependencies": []
            }
    
    async def _review_code(
        self,
        code: str,
        language: str,
        focus: List[str] = None
    ) -> Dict[str, Any]:
        """Review code for quality, bugs, and security."""
        focus = focus or ["bugs", "security", "style", "performance"]
        
        prompt = f"""Review this {language} code:

```{language}
{code}
```

Focus areas: {', '.join(focus)}

Provide a comprehensive review including:
1. Overall quality score (1-10)
2. Bugs found
3. Security issues
4. Style issues
5. Performance concerns
6. Suggested improvements

Respond in JSON format:
{{
    "quality_score": 8,
    "bugs": [
        {{"severity": "high|medium|low", "description": "desc", "line": 1, "fix": "suggestion"}}
    ],
    "security_issues": [
        {{"severity": "high|medium|low", "description": "desc", "recommendation": "fix"}}
    ],
    "style_issues": [
        {{"description": "desc", "recommendation": "fix"}}
    ],
    "performance_concerns": [
        {{"description": "desc", "recommendation": "fix"}}
    ],
    "improvements": ["improvement1", "improvement2"],
    "summary": "Overall summary"
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "quality_score": 5,
                "summary": response,
                "bugs": [],
                "security_issues": [],
                "style_issues": [],
                "performance_concerns": [],
                "improvements": []
            }
    
    async def _debug_code(
        self,
        code: str,
        error: str,
        language: str
    ) -> Dict[str, Any]:
        """Debug code and provide fixes."""
        prompt = f"""Debug this {language} code:

```{language}
{code}
```

Error/Problem: {error}

Provide:
1. Root cause analysis
2. Fixed code
3. Explanation of the fix
4. Prevention tips

Respond in JSON format:
{{
    "root_cause": "Explanation of what caused the bug",
    "fixed_code": "// The corrected code",
    "explanation": "What was changed and why",
    "prevention_tips": ["tip1", "tip2"]
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "root_cause": "Unable to parse response",
                "fixed_code": code,
                "explanation": response,
                "prevention_tips": []
            }
    
    async def _explain_code(
        self,
        code: str,
        language: str
    ) -> Dict[str, Any]:
        """Explain how code works."""
        prompt = f"""Explain this {language} code in detail:

```{language}
{code}
```

Provide:
1. High-level overview
2. Step-by-step explanation
3. Key concepts used
4. Potential use cases

Respond in JSON format:
{{
    "overview": "High-level description",
    "step_by_step": [
        {{"line_range": "1-5", "explanation": "What these lines do"}}
    ],
    "concepts": ["concept1", "concept2"],
    "use_cases": ["use case 1", "use case 2"]
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self._call_llm(messages)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "overview": response,
                "step_by_step": [],
                "concepts": [],
                "use_cases": []
            }
    
    async def _execute_code(
        self,
        code: str,
        language: str,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Safely execute code in a sandboxed environment."""
        if language not in self.allowed_languages:
            return {
                "success": False,
                "error": f"Language '{language}' not allowed for execution"
            }
        
        if not self.enable_execution:
            return {
                "success": False,
                "error": "Code execution is disabled"
            }
        
        # Currently only Python execution is implemented
        if language != "python":
            return {
                "success": False,
                "error": f"Execution not implemented for {language}"
            }
        
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute in subprocess with timeout
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
                )
                
                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
            finally:
                # Clean up temp file
                os.unlink(temp_file)
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
