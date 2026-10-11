"""
Database Query Tool.

Provides database querying capabilities with support for
SQL and natural language queries.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

from .registry import tool, ToolCategory


logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from a database query."""
    success: bool
    data: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    query: str
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "columns": self.columns,
            "row_count": self.row_count,
            "query": self.query,
            "error": self.error
        }


class DatabaseConnector(ABC):
    """Abstract base class for database connectors."""
    
    @abstractmethod
    async def execute(self, query: str) -> QueryResult:
        """Execute a SQL query."""
        pass
    
    @abstractmethod
    async def get_schema(self) -> Dict[str, Any]:
        """Get database schema information."""
        pass


class SQLiteConnector(DatabaseConnector):
    """SQLite database connector."""
    
    def __init__(self, database_path: str):
        self.database_path = database_path
    
    async def execute(self, query: str) -> QueryResult:
        """Execute a SQL query against SQLite."""
        import aiosqlite
        
        try:
            async with aiosqlite.connect(self.database_path) as db:
                db.row_factory = aiosqlite.Row
                
                async with db.execute(query) as cursor:
                    rows = await cursor.fetchall()
                    columns = [description[0] for description in cursor.description] if cursor.description else []
                    
                    data = [dict(row) for row in rows]
                    
                    return QueryResult(
                        success=True,
                        data=data,
                        columns=columns,
                        row_count=len(data),
                        query=query
                    )
                    
        except Exception as e:
            logger.error(f"SQLite query failed: {e}")
            return QueryResult(
                success=False,
                data=[],
                columns=[],
                row_count=0,
                query=query,
                error=str(e)
            )
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get SQLite database schema."""
        import aiosqlite
        
        schema = {"tables": {}}
        
        try:
            async with aiosqlite.connect(self.database_path) as db:
                # Get tables
                async with db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ) as cursor:
                    tables = await cursor.fetchall()
                
                for (table_name,) in tables:
                    # Get columns for each table
                    async with db.execute(f"PRAGMA table_info({table_name})") as cursor:
                        columns = await cursor.fetchall()
                    
                    schema["tables"][table_name] = {
                        "columns": [
                            {
                                "name": col[1],
                                "type": col[2],
                                "nullable": not col[3],
                                "primary_key": bool(col[5])
                            }
                            for col in columns
                        ]
                    }
            
            return schema
            
        except Exception as e:
            logger.error(f"Failed to get schema: {e}")
            return {"error": str(e)}


class PostgresConnector(DatabaseConnector):
    """PostgreSQL database connector."""
    
    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str
    ):
        self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    async def execute(self, query: str) -> QueryResult:
        """Execute a SQL query against PostgreSQL."""
        try:
            import asyncpg
            
            conn = await asyncpg.connect(self.connection_string)
            try:
                rows = await conn.fetch(query)
                
                if rows:
                    columns = list(rows[0].keys())
                    data = [dict(row) for row in rows]
                else:
                    columns = []
                    data = []
                
                return QueryResult(
                    success=True,
                    data=data,
                    columns=columns,
                    row_count=len(data),
                    query=query
                )
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"PostgreSQL query failed: {e}")
            return QueryResult(
                success=False,
                data=[],
                columns=[],
                row_count=0,
                query=query,
                error=str(e)
            )
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get PostgreSQL database schema."""
        query = """
        SELECT 
            table_name,
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
        """
        
        result = await self.execute(query)
        
        if not result.success:
            return {"error": result.error}
        
        schema = {"tables": {}}
        for row in result.data:
            table = row["table_name"]
            if table not in schema["tables"]:
                schema["tables"][table] = {"columns": []}
            
            schema["tables"][table]["columns"].append({
                "name": row["column_name"],
                "type": row["data_type"],
                "nullable": row["is_nullable"] == "YES",
                "default": row["column_default"]
            })
        
        return schema


class NaturalLanguageQueryProcessor:
    """
    Converts natural language queries to SQL using LLM.
    """
    
    def __init__(self, llm, connector: DatabaseConnector):
        self.llm = llm
        self.connector = connector
        self._schema_cache: Optional[Dict[str, Any]] = None
    
    async def process(self, natural_query: str) -> QueryResult:
        """
        Process a natural language query.
        
        Args:
            natural_query: Query in natural language
            
        Returns:
            Query result
        """
        # Get schema for context
        if self._schema_cache is None:
            self._schema_cache = await self.connector.get_schema()
        
        # Generate SQL from natural language
        sql = await self._generate_sql(natural_query, self._schema_cache)
        
        if not sql:
            return QueryResult(
                success=False,
                data=[],
                columns=[],
                row_count=0,
                query=natural_query,
                error="Could not generate SQL from query"
            )
        
        # Execute the generated SQL
        result = await self.connector.execute(sql)
        result.query = f"NL: {natural_query}\nSQL: {sql}"
        
        return result
    
    async def _generate_sql(
        self,
        natural_query: str,
        schema: Dict[str, Any]
    ) -> Optional[str]:
        """Generate SQL from natural language query."""
        schema_text = self._format_schema(schema)
        
        prompt = f"""Given this database schema:

{schema_text}

Convert this natural language query to SQL:
"{natural_query}"

Return only the SQL query, nothing else. Use proper SQL syntax."""
        
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm.generate(messages, temperature=0)
        
        sql = response.content.strip()
        
        # Basic cleanup
        if sql.startswith("```sql"):
            sql = sql[6:]
        if sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        
        return sql.strip()
    
    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format schema for LLM context."""
        lines = []
        for table, info in schema.get("tables", {}).items():
            columns = ", ".join([
                f"{col['name']} ({col['type']})"
                for col in info.get("columns", [])
            ])
            lines.append(f"Table: {table}\n  Columns: {columns}")
        
        return "\n\n".join(lines)


class DatabaseTool:
    """
    Database query tool with support for SQL and natural language queries.
    """
    
    def __init__(
        self,
        connectors: Optional[Dict[str, DatabaseConnector]] = None,
        default_connector: str = "default",
        llm = None
    ):
        self.connectors = connectors or {}
        self.default_connector = default_connector
        self.llm = llm
        self._nl_processors: Dict[str, NaturalLanguageQueryProcessor] = {}
    
    def add_connector(self, name: str, connector: DatabaseConnector) -> None:
        """Add a database connector."""
        self.connectors[name] = connector
        
        if self.llm:
            self._nl_processors[name] = NaturalLanguageQueryProcessor(
                self.llm,
                connector
            )
    
    async def query(
        self,
        query: str,
        database: Optional[str] = None,
        natural_language: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a database query.
        
        Args:
            query: SQL or natural language query
            database: Database connector name
            natural_language: Whether query is in natural language
            
        Returns:
            Query results
        """
        db_name = database or self.default_connector
        
        if db_name not in self.connectors:
            return {
                "success": False,
                "error": f"Unknown database: {db_name}"
            }
        
        connector = self.connectors[db_name]
        
        if natural_language:
            if db_name not in self._nl_processors:
                return {
                    "success": False,
                    "error": "Natural language queries require LLM configuration"
                }
            
            result = await self._nl_processors[db_name].process(query)
        else:
            result = await connector.execute(query)
        
        return result.to_dict()
    
    async def get_schema(self, database: Optional[str] = None) -> Dict[str, Any]:
        """Get database schema."""
        db_name = database or self.default_connector
        
        if db_name not in self.connectors:
            return {"error": f"Unknown database: {db_name}"}
        
        return await self.connectors[db_name].get_schema()


# Register as tool
@tool(
    name="database_query",
    description="Execute SQL queries against databases. Can also accept natural language queries if configured.",
    category=ToolCategory.DATA,
    parameters={
        "query": {
            "type": "string",
            "description": "SQL query or natural language query"
        },
        "database": {
            "type": "string",
            "description": "Database name to query",
            "default": "default"
        },
        "natural_language": {
            "type": "boolean",
            "description": "Whether the query is in natural language",
            "default": False
        }
    },
    required_params=["query"]
)
async def database_query(
    query: str,
    database: str = "default",
    natural_language: bool = False
) -> Dict[str, Any]:
    """Execute a database query."""
    # This is a placeholder - in production, would use configured DatabaseTool
    return {
        "success": False,
        "error": "Database not configured",
        "query": query
    }
