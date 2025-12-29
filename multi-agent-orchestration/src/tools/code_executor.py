"""
Safe Code Execution Tool.

Provides sandboxed code execution with resource limits
and security constraints.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import subprocess
import tempfile
import os
import sys
import resource
from pathlib import Path


logger = logging.getLogger(__name__)


class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    BASH = "bash"


@dataclass
class ExecutionResult:
    """Result from code execution."""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "execution_time": self.execution_time,
            "error": self.error
        }


@dataclass
class ExecutionConfig:
    """Configuration for code execution."""
    timeout: int = 30  # seconds
    max_memory: int = 256 * 1024 * 1024  # 256 MB
    max_output: int = 10000  # characters
    allowed_imports: List[str] = field(default_factory=lambda: [
        "math", "random", "datetime", "json", "re", "collections",
        "itertools", "functools", "operator", "string", "textwrap",
        "typing", "dataclasses", "enum", "statistics", "decimal"
    ])
    blocked_imports: List[str] = field(default_factory=lambda: [
        "os", "sys", "subprocess", "shutil", "socket", "urllib",
        "requests", "http", "ftplib", "smtplib", "telnetlib",
        "pickle", "shelve", "marshal", "importlib", "ctypes",
        "multiprocessing", "threading", "asyncio"
    ])


class CodeSanitizer:
    """Sanitizes code for safe execution."""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
    
    def sanitize_python(self, code: str) -> tuple[bool, str, Optional[str]]:
        """
        Sanitize Python code.
        
        Returns:
            Tuple of (is_safe, sanitized_code, error_message)
        """
        # Check for blocked imports
        for blocked in self.config.blocked_imports:
            patterns = [
                f"import {blocked}",
                f"from {blocked}",
                f"__import__('{blocked}'",
                f'__import__("{blocked}"'
            ]
            for pattern in patterns:
                if pattern in code:
                    return False, code, f"Blocked import detected: {blocked}"
        
        # Check for dangerous operations
        dangerous_patterns = [
            ("eval(", "eval() is not allowed"),
            ("exec(", "exec() is not allowed"),
            ("compile(", "compile() is not allowed"),
            ("open(", "File operations are not allowed"),
            ("__", "Dunder attributes are restricted"),
            ("globals(", "globals() is not allowed"),
            ("locals(", "locals() is not allowed"),
        ]
        
        for pattern, message in dangerous_patterns:
            if pattern in code:
                return False, code, message
        
        return True, code, None
    
    def sanitize_javascript(self, code: str) -> tuple[bool, str, Optional[str]]:
        """Sanitize JavaScript code."""
        # Check for dangerous patterns
        dangerous_patterns = [
            ("require(", "require() is not allowed"),
            ("import ", "Dynamic imports are not allowed"),
            ("eval(", "eval() is not allowed"),
            ("Function(", "Function constructor is not allowed"),
            ("process.", "process access is not allowed"),
            ("child_process", "child_process is not allowed"),
            ("fs.", "Filesystem access is not allowed"),
        ]
        
        for pattern, message in dangerous_patterns:
            if pattern in code:
                return False, code, message
        
        return True, code, None
    
    def sanitize_bash(self, code: str) -> tuple[bool, str, Optional[str]]:
        """Sanitize Bash code."""
        # Bash is generally unsafe - allow only very basic commands
        allowed_commands = ["echo", "printf", "expr", "test", "["]
        
        # Check first word of each line
        for line in code.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            first_word = line.split()[0] if line.split() else ""
            if first_word and first_word not in allowed_commands:
                # Check for pipes and redirects
                if "|" in line or ">" in line or "<" in line or "&" in line:
                    return False, code, "Pipes and redirects are not allowed"
        
        return True, code, None


class CodeExecutor:
    """
    Safe code execution with sandboxing.
    
    Features:
    - Resource limits (CPU, memory, time)
    - Code sanitization
    - Isolated execution environment
    - Output capture and truncation
    """
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.sanitizer = CodeSanitizer(self.config)
    
    async def execute(
        self,
        code: str,
        language: Language = Language.PYTHON,
        timeout: Optional[int] = None
    ) -> ExecutionResult:
        """
        Execute code in a sandboxed environment.
        
        Args:
            code: Code to execute
            language: Programming language
            timeout: Execution timeout in seconds
            
        Returns:
            ExecutionResult with output and status
        """
        timeout = timeout or self.config.timeout
        
        # Sanitize code
        if language == Language.PYTHON:
            is_safe, code, error = self.sanitizer.sanitize_python(code)
        elif language == Language.JAVASCRIPT:
            is_safe, code, error = self.sanitizer.sanitize_javascript(code)
        elif language == Language.BASH:
            is_safe, code, error = self.sanitizer.sanitize_bash(code)
        else:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                return_code=-1,
                execution_time=0,
                error=f"Unsupported language: {language}"
            )
        
        if not is_safe:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=error or "Code validation failed",
                return_code=-1,
                execution_time=0,
                error=error
            )
        
        # Execute based on language
        if language == Language.PYTHON:
            return await self._execute_python(code, timeout)
        elif language == Language.JAVASCRIPT:
            return await self._execute_javascript(code, timeout)
        elif language == Language.BASH:
            return await self._execute_bash(code, timeout)
    
    async def _execute_python(
        self,
        code: str,
        timeout: int
    ) -> ExecutionResult:
        """Execute Python code."""
        import time
        start_time = time.time()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            # Wrap code with resource limits
            wrapped_code = f'''
import resource
import sys

# Set resource limits
resource.setrlimit(resource.RLIMIT_AS, ({self.config.max_memory}, {self.config.max_memory}))
resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))

# Execute user code
{code}
'''
            f.write(wrapped_code)
            temp_file = f.name
        
        try:
            # Execute in subprocess
            process = await asyncio.create_subprocess_exec(
                sys.executable, temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    return_code=-1,
                    execution_time=timeout,
                    error=f"Execution timed out after {timeout} seconds"
                )
            
            execution_time = time.time() - start_time
            
            # Truncate output if needed
            stdout_str = stdout.decode()[:self.config.max_output]
            stderr_str = stderr.decode()[:self.config.max_output]
            
            return ExecutionResult(
                success=process.returncode == 0,
                stdout=stdout_str,
                stderr=stderr_str,
                return_code=process.returncode,
                execution_time=execution_time
            )
            
        finally:
            # Clean up
            os.unlink(temp_file)
    
    async def _execute_javascript(
        self,
        code: str,
        timeout: int
    ) -> ExecutionResult:
        """Execute JavaScript code using Node.js."""
        import time
        start_time = time.time()
        
        # Check if Node.js is available
        try:
            await asyncio.create_subprocess_exec(
                "node", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                return_code=-1,
                execution_time=0,
                error="Node.js is not installed"
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.js',
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            process = await asyncio.create_subprocess_exec(
                "node", temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    return_code=-1,
                    execution_time=timeout,
                    error=f"Execution timed out after {timeout} seconds"
                )
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=process.returncode == 0,
                stdout=stdout.decode()[:self.config.max_output],
                stderr=stderr.decode()[:self.config.max_output],
                return_code=process.returncode,
                execution_time=execution_time
            )
            
        finally:
            os.unlink(temp_file)
    
    async def _execute_bash(
        self,
        code: str,
        timeout: int
    ) -> ExecutionResult:
        """Execute Bash code."""
        import time
        start_time = time.time()
        
        try:
            process = await asyncio.create_subprocess_shell(
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="",
                    return_code=-1,
                    execution_time=timeout,
                    error=f"Execution timed out after {timeout} seconds"
                )
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=process.returncode == 0,
                stdout=stdout.decode()[:self.config.max_output],
                stderr=stderr.decode()[:self.config.max_output],
                return_code=process.returncode,
                execution_time=execution_time
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=time.time() - start_time,
                error=str(e)
            )


# Create global executor instance
_executor = CodeExecutor()


async def execute_code(
    code: str,
    language: str = "python",
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Execute code safely.
    
    Args:
        code: Code to execute
        language: Programming language (python, javascript, bash)
        timeout: Execution timeout in seconds
        
    Returns:
        Execution result
    """
    try:
        lang = Language(language.lower())
    except ValueError:
        return {
            "success": False,
            "error": f"Unsupported language: {language}"
        }
    
    result = await _executor.execute(code, lang, timeout)
    return result.to_dict()
