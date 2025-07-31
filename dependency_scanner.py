"""
GitHub Repository Dependency Scanner

This script scans GitHub repositories and builds dependency graphs using Neo4j.
It parses import statements across different file types and stores the dependency
graph with nodes representing files/modules and edges representing dependencies.

Required packages:
pip install neo4j langchain langchain-openai langchain-community pydantic python-dotenv gitpython requests rich
"""

import os
import ast
import re
import json
import git
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
from urllib.parse import urlparse
from datetime import datetime
import logging
from dataclasses import dataclass

# Environment and configuration
from dotenv import load_dotenv

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph

# Neo4j imports
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

# Rich for better console output
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv()

# Setup console for rich output
console = Console()


@dataclass
class ImportInfo:
    """Data class for storing import information"""
    module: str
    items: List[str] = None
    import_type: str = "direct"  # direct, from_import, relative, wildcard_import, wildcard_resolved, dynamic_import, exec_import
    line_number: int = 0
    alias: Optional[str] = None
    wildcard_import: bool = False
    dynamic_import: bool = False


@dataclass
class DependencyInfo:
    """Data class for storing dependency information"""
    name: str
    version: Optional[str] = None
    dependency_type: str = "package"  # package, module, function
    source_file: Optional[str] = None


class DependencyScanner:
    """
    Scans GitHub repositories and builds dependency graphs in Neo4j
    """
    
    def __init__(self, 
                 neo4j_uri: Optional[str] = None,
                 neo4j_username: Optional[str] = None,
                 neo4j_password: Optional[str] = None,
                 openai_api_key: Optional[str] = None,
                 model_name: str = "gpt-4o-mini"):
        """
        Initialize the dependency scanner
        
        Args:
            neo4j_uri: Neo4j database URI (defaults to NEO4J_URI env var)
            neo4j_username: Neo4j username (defaults to NEO4J_USERNAME env var)
            neo4j_password: Neo4j password (defaults to NEO4J_PASSWORD env var)
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model_name: OpenAI model name to use
        """
        # Load configuration from environment variables
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_username = neo4j_username or os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not all([self.neo4j_uri, self.neo4j_username, self.neo4j_password]):
            raise ValueError("Missing Neo4j configuration. Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables.")
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_username, self.neo4j_password)
        )
        
        # Initialize LangChain components
        if self.openai_api_key:
            self.llm = ChatOpenAI(temperature=0, model=model_name)
        else:
            self.llm = None
            console.print("⚠️  OpenAI API key not provided. LLM features will be disabled.", style="yellow")
        
        # Initialize Neo4j Graph for LangChain integration
        self.graph_db = Neo4jGraph(enhanced_schema=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # File extensions to scan with their parsers
        self.supported_extensions = {
            '.py': self._parse_python_imports,
            '.js': self._parse_javascript_imports,
            '.ts': self._parse_typescript_imports,
            '.jsx': self._parse_javascript_imports,
            '.tsx': self._parse_typescript_imports,
            '.java': self._parse_java_imports,
            '.go': self._parse_go_imports,
            '.rs': self._parse_rust_imports,
            '.php': self._parse_php_imports,
            '.rb': self._parse_ruby_imports
        }
        
        # Language mapping
        self.language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'JavaScript',
            '.tsx': 'TypeScript',
            '.java': 'Java',
            '.go': 'Go',
            '.rs': 'Rust',
            '.php': 'PHP',
            '.rb': 'Ruby'
        }
        
        # Initialize database schema
        self._setup_database_schema()
        
        console.print("✅ Dependency Scanner initialized successfully", style="green")
    
    def _setup_database_schema(self):
        """Setup Neo4j database schema with constraints and indexes"""
        try:
            with self.driver.session() as session:
                # Create constraints for unique nodes
                session.run("CREATE CONSTRAINT file_path IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE")
                session.run("CREATE CONSTRAINT module_name IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE")
                session.run("CREATE CONSTRAINT package_name IF NOT EXISTS FOR (p:Package) REQUIRE p.name IS UNIQUE")
                session.run("CREATE CONSTRAINT repository_url IF NOT EXISTS FOR (r:Repository) REQUIRE r.url IS UNIQUE")
                session.run("CREATE CONSTRAINT function_name IF NOT EXISTS FOR (f:Function) REQUIRE f.full_name IS UNIQUE")
                
                # Create indexes for better query performance
                session.run("CREATE INDEX file_extension IF NOT EXISTS FOR (f:File) ON (f.extension)")
                session.run("CREATE INDEX file_language IF NOT EXISTS FOR (f:File) ON (f.language)")
                session.run("CREATE INDEX import_type IF NOT EXISTS FOR ()-[i:IMPORTS]-() ON (i.type)")
                session.run("CREATE INDEX import_from_type IF NOT EXISTS FOR ()-[i:IMPORTS_FROM]-() ON (i.type)")
                session.run("CREATE INDEX wildcard_imports IF NOT EXISTS FOR ()-[w:WILDCARD_IMPORTS]-() ON (w.wildcard_import)")
                session.run("CREATE INDEX dynamic_imports IF NOT EXISTS FOR ()-[d:DYNAMIC_IMPORTS]-() ON (d.dynamic_import)")
                session.run("CREATE INDEX transitive_deps IF NOT EXISTS FOR ()-[t:TRANSITIVE_DEPENDS_ON]-() ON (t.added_at)")
                
                console.print("✅ Database schema setup completed", style="green")
                
        except Exception as e:
            console.print(f"❌ Failed to setup database schema: {e}", style="red")
            raise
    
    def scan_repository(self, 
                       repo_path: str, 
                       repo_url: Optional[str] = None,
                       branch: str = "main") -> Dict[str, Any]:
        """
        Scan a repository and build its dependency graph
        
        Args:
            repo_path: Path to the repository (local)
            repo_url: Optional repository URL for metadata
            branch: Branch to scan (default: main)
            
        Returns:
            Dictionary with scan results and statistics
        """
        try:
            repo_path_obj = Path(repo_path)
            if not repo_path_obj.exists():
                raise ValueError(f"Repository path does not exist: {repo_path}")
            
            console.print(f"🔍 Starting scan of repository: {repo_path}", style="blue")
            
            # Create repository node
            repo_id = self._create_repository_node(repo_path, repo_url)
            
            # Scan all files in the repository
            scan_results = self._scan_repository_files(repo_path_obj, repo_id)
            
            # Parse dependency files
            dependency_results = self._parse_dependency_files(repo_path_obj, repo_id)
            
            # Build dependency graph
            graph_results = self._build_dependency_graph(repo_id)
            
            # Generate scan summary
            summary = {
                "repository": repo_url or repo_path,
                "branch": branch,
                "scan_timestamp": datetime.now().isoformat(),
                "files_scanned": scan_results["files_scanned"],
                "dependencies_found": scan_results["dependencies_found"],
                "external_packages": dependency_results["external_packages"],
                "graph_nodes": graph_results["nodes"],
                "graph_relationships": graph_results["relationships"]
            }
            
            console.print(f"✅ Repository scan completed: {summary}", style="green")
            return summary
            
        except Exception as e:
            console.print(f"❌ Failed to scan repository {repo_path}: {e}", style="red")
            raise
    
    def _create_repository_node(self, repo_path: str, repo_url: Optional[str]) -> str:
        """Create a repository node in Neo4j"""
        try:
            with self.driver.session() as session:
                # Use repo_path as URL if none provided
                url = repo_url or f"file://{repo_path}"
                
                result = session.run("""
                    MERGE (r:Repository {url: $url})
                    SET r.path = $path,
                        r.last_scan = datetime(),
                        r.name = $name
                    RETURN r.url as repo_id
                """, url=url, path=repo_path, name=Path(repo_path).name)
                
                repo_id = result.single()["repo_id"]
                console.print(f"✅ Created repository node: {repo_id}", style="green")
                return repo_id
                
        except Exception as e:
            console.print(f"❌ Failed to create repository node: {e}", style="red")
            raise
    
    def _scan_repository_files(self, repo_path: Path, repo_id: str) -> Dict[str, Any]:
        """Scan all files in the repository for dependencies"""
        try:
            files_scanned = 0
            dependencies_found = 0
            
            # Get all files with supported extensions
            supported_files = []
            for ext in self.supported_extensions.keys():
                supported_files.extend(repo_path.rglob(f"*{ext}"))
            
            console.print(f"Found {len(supported_files)} files to scan", style="blue")
            
            # Limit files for large repositories to prevent hanging
            max_files = 1000
            if len(supported_files) > max_files:
                console.print(f"⚠️ Large repository detected. Limiting scan to first {max_files} files", style="yellow")
                supported_files = supported_files[:max_files]
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Scanning files...", total=len(supported_files))
                
                for i, file_path in enumerate(supported_files):
                    try:
                        # Skip files in common directories to ignore
                        if any(part in ['.git', '__pycache__', 'node_modules', 'venv', '.venv', 'dist', 'build'] 
                               for part in file_path.parts):
                            progress.update(task, advance=1)
                            continue
                        
                        # Update progress description with current file
                        if i % 10 == 0:  # Update every 10 files to avoid spam
                            progress.update(task, description=f"Scanning {file_path.name}...")
                        
                        # Create file node
                        file_id = self._create_file_node(file_path, repo_id)
                        files_scanned += 1
                        
                        # Parse dependencies
                        imports = self._parse_file_dependencies(file_path)
                        if imports:
                            self._create_dependency_relationships(file_id, imports, repo_id)
                            dependencies_found += len(imports)
                        
                        progress.update(task, advance=1)
                        
                    except KeyboardInterrupt:
                        console.print(f"\n⚠️ File scanning interrupted by user", style="yellow")
                        break
                    except Exception as e:
                        console.print(f"⚠️ Failed to process file {file_path.name}: {e}", style="yellow")
                        progress.update(task, advance=1)
                        continue
            
            return {
                "files_scanned": files_scanned,
                "dependencies_found": dependencies_found
            }
            
        except Exception as e:
            console.print(f"❌ Failed to scan repository files: {e}", style="red")
            raise
    
    def _create_file_node(self, file_path: Path, repo_id: str) -> str:
        """Create a file node in Neo4j"""
        try:
            with self.driver.session() as session:
                # Get relative path from repository root
                relative_path = str(file_path.relative_to(file_path.parts[0]))
                
                result = session.run("""
                    MERGE (f:File {path: $path})
                    SET f.extension = $extension,
                        f.language = $language,
                        f.size = $size,
                        f.last_modified = datetime()
                    WITH f
                    MATCH (r:Repository {url: $repo_id})
                    MERGE (f)-[:BELONGS_TO]->(r)
                    RETURN f.path as file_id
                """, path=relative_path, 
                     extension=file_path.suffix,
                     language=self.language_map.get(file_path.suffix, 'Unknown'),
                     size=file_path.stat().st_size,
                     repo_id=repo_id)
                
                file_id = result.single()["file_id"]
                return file_id
                
        except Exception as e:
            console.print(f"❌ Failed to create file node: {e}", style="red")
            raise
    
    def _parse_file_dependencies(self, file_path: Path) -> List[ImportInfo]:
        """Parse dependencies from a single file"""
        try:
            extension = file_path.suffix
            if extension not in self.supported_extensions:
                return []
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse dependencies based on file type
            parser_func = self.supported_extensions[extension]
            imports = parser_func(content, str(file_path))
            
            return imports
            
        except Exception as e:
            console.print(f"⚠️ Failed to parse dependencies from {file_path}: {e}", style="yellow")
            return []
    
    def _parse_python_imports(self, content: str, file_path: str) -> List[ImportInfo]:
        """Parse Python import statements with enhanced support for complex imports"""
        imports = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(ImportInfo(
                            module=alias.name,
                            import_type="direct",
                            line_number=node.lineno,
                            alias=alias.asname
                        ))
                        
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    
                    # Check for wildcard imports
                    if any(alias.name == "*" for alias in node.names):
                        imports.append(ImportInfo(
                            module=module,
                            import_type="wildcard_import",
                            line_number=node.lineno,
                            wildcard_import=True
                        ))
                        
                        # Try to resolve wildcard imports to specific symbols
                        wildcard_symbols = self._resolve_wildcard_imports(module)
                        if wildcard_symbols:
                            for symbol in wildcard_symbols:
                                imports.append(ImportInfo(
                                    module=module,
                                    items=[symbol],
                                    import_type="wildcard_resolved",
                                    line_number=node.lineno
                                ))
                    
                    else:
                        # Regular from imports
                        items = []
                        for alias in node.names:
                            items.append(alias.name)
                            if alias.asname:
                                # Handle aliased imports
                                imports.append(ImportInfo(
                                    module=module,
                                    items=[alias.name],
                                    import_type="from_import",
                                    line_number=node.lineno,
                                    alias=alias.asname
                                ))
                        
                        if items:
                            imports.append(ImportInfo(
                                module=module,
                                items=items,
                                import_type="from_import",
                                line_number=node.lineno
                            ))
            
            # Detect dynamic imports
            dynamic_imports = self._detect_dynamic_imports(content)
            imports.extend(dynamic_imports)
                        
        except SyntaxError:
            # Handle syntax errors by using regex fallback
            imports.extend(self._parse_python_imports_regex(content))
        
        return imports
    
    def _parse_python_imports_regex(self, content: str) -> List[ImportInfo]:
        """Fallback regex parser for Python imports"""
        imports = []
        
        # Match import statements
        import_pattern = r'^import\s+([a-zA-Z_][a-zA-Z0-9_.]*)(?:\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*))?'
        from_pattern = r'^from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import\s+(.+)'
        
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            
            # Check import statements
            import_match = re.match(import_pattern, line)
            if import_match:
                module = import_match.group(1)
                alias = import_match.group(2)
                imports.append(ImportInfo(
                    module=module,
                    import_type="direct",
                    line_number=line_num,
                    alias=alias
                ))
            
            # Check from import statements
            from_match = re.match(from_pattern, line)
            if from_match:
                module = from_match.group(1)
                imports_text = from_match.group(2)
                
                # Parse multiple imports
                for import_item in imports_text.split(','):
                    import_item = import_item.strip()
                    if ' as ' in import_item:
                        name, alias = import_item.split(' as ', 1)
                    else:
                        name, alias = import_item, None
                    
                    imports.append(ImportInfo(
                        module=module,
                        items=[name],
                        import_type="from_import",
                        line_number=line_num,
                        alias=alias
                    ))
        
        return imports
    
    def _parse_javascript_imports(self, content: str, file_path: str) -> List[ImportInfo]:
        """Parse JavaScript/JSX import statements"""
        imports = []
        
        # Match ES6 import statements
        import_patterns = [
            r'import\s+(\{[^}]*\})\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'import\s+[\'"]([^\'"]+)[\'"]',
            r'import\s+\*\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+from\s+[\'"]([^\'"]+)[\'"]'
        ]
        
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            
            for pattern in import_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if isinstance(match, tuple):
                        if len(match) == 2:
                            imports_part, module = match
                            if imports_part.startswith('{') and imports_part.endswith('}'):
                                # Named imports
                                named_imports = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*))?', imports_part)
                                for name, alias in named_imports:
                                    imports.append(ImportInfo(
                                        module=module,
                                        items=[name],
                                        import_type="from_import",
                                        line_number=line_num,
                                        alias=alias
                                    ))
                            else:
                                # Default import
                                imports.append(ImportInfo(
                                    module=module,
                                    items=[imports_part],
                                    import_type="from_import",
                                    line_number=line_num
                                ))
                        else:
                            # Namespace import
                            namespace, module = match
                            imports.append(ImportInfo(
                                module=module,
                                items=[namespace],
                                import_type="namespace_import",
                                line_number=line_num
                            ))
                    else:
                        # Simple import
                        imports.append(ImportInfo(
                            module=match,
                            import_type="direct",
                            line_number=line_num
                        ))
        
        return imports
    
    def _parse_typescript_imports(self, content: str, file_path: str) -> List[ImportInfo]:
        """Parse TypeScript imports (same as JavaScript for now)"""
        return self._parse_javascript_imports(content, file_path)
    
    def _parse_java_imports(self, content: str, file_path: str) -> List[ImportInfo]:
        """Parse Java import statements"""
        imports = []
        
        import_pattern = r'^import\s+([a-zA-Z_][a-zA-Z0-9_.]*)(?:\s*;)?'
        
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            
            match = re.match(import_pattern, line)
            if match:
                module = match.group(1)
                imports.append(ImportInfo(
                    module=module,
                    import_type="direct",
                    line_number=line_num
                ))
        
        return imports
    
    def _parse_go_imports(self, content: str, file_path: str) -> List[ImportInfo]:
        """Parse Go import statements"""
        imports = []
        
        # Match Go import blocks
        import_block_pattern = r'import\s*\(\s*([^)]+)\s*\)'
        single_import_pattern = r'^import\s+[\'"]([^\'"]+)[\'"]'
        
        # Check for import blocks
        block_matches = re.findall(import_block_pattern, content, re.DOTALL)
        for block in block_matches:
            for line in block.split('\n'):
                line = line.strip()
                if line and not line.startswith('//'):
                    # Extract package path
                    match = re.search(r'[\'"]([^\'"]+)[\'"]', line)
                    if match:
                        package = match.group(1)
                        imports.append(ImportInfo(
                            module=package,
                            import_type="direct",
                            line_number=0  # Line number not available for blocks
                        ))
        
        # Check for single imports
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            match = re.match(single_import_pattern, line)
            if match:
                package = match.group(1)
                imports.append(ImportInfo(
                    module=package,
                    import_type="direct",
                    line_number=line_num
                ))
        
        return imports
    
    def _parse_rust_imports(self, content: str, file_path: str) -> List[ImportInfo]:
        """Parse Rust use statements"""
        imports = []
        
        use_patterns = [
            r'^use\s+([a-zA-Z_][a-zA-Z0-9_:]*)(?:\s*;)?',
            r'^use\s+([a-zA-Z_][a-zA-Z0-9_:]*)\s*::\s*([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*;)?'
        ]
        
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            
            for pattern in use_patterns:
                match = re.match(pattern, line)
                if match:
                    if len(match.groups()) == 1:
                        module = match.group(1)
                        imports.append(ImportInfo(
                            module=module,
                            import_type="direct",
                            line_number=line_num
                        ))
                    else:
                        module, item = match.groups()
                        imports.append(ImportInfo(
                            module=module,
                            items=[item],
                            import_type="from_import",
                            line_number=line_num
                        ))
        
        return imports
    
    def _parse_php_imports(self, content: str, file_path: str) -> List[ImportInfo]:
        """Parse PHP use statements"""
        imports = []
        
        use_pattern = r'^use\s+([a-zA-Z_][a-zA-Z0-9_\\]*)(?:\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*))?(?:\s*;)?'
        
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            
            match = re.match(use_pattern, line)
            if match:
                namespace = match.group(1)
                alias = match.group(2)
                imports.append(ImportInfo(
                    module=namespace,
                    import_type="direct",
                    line_number=line_num,
                    alias=alias
                ))
        
        return imports
    
    def _parse_ruby_imports(self, content: str, file_path: str) -> List[ImportInfo]:
        """Parse Ruby require/load statements"""
        imports = []
        
        require_pattern = r'^(?:require|load)\s+[\'"]([^\'"]+)[\'"]'
        
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            
            match = re.match(require_pattern, line)
            if match:
                module = match.group(1)
                imports.append(ImportInfo(
                    module=module,
                    import_type="direct",
                    line_number=line_num
                ))
        
        return imports
    
    def _resolve_wildcard_imports(self, module_name: str) -> List[str]:
        """Attempt to resolve wildcard imports to specific symbols"""
        try:
            # Common modules and their public symbols
            common_modules = {
                'os': ['path', 'name', 'environ', 'getcwd', 'chdir', 'mkdir', 'rmdir', 'remove', 'rename', 'stat', 'listdir', 'walk', 'system', 'popen', 'kill', 'getpid', 'getppid', 'getuid', 'getgid', 'setuid', 'setgid', 'umask', 'chmod', 'chown', 'link', 'symlink', 'readlink', 'unlink', 'makedirs', 'removedirs', 'renames', 'stat_float_times', 'curdir', 'pardir', 'sep', 'extsep', 'altsep', 'pathsep', 'defpath', 'devnull', 'linesep'],
                'sys': ['argv', 'path', 'modules', 'version', 'version_info', 'hexversion', 'platform', 'implementation', 'maxsize', 'maxunicode', 'exc_info', 'exc_type', 'exc_value', 'exc_traceback', 'last_type', 'last_value', 'last_traceback', 'stdin', 'stdout', 'stderr', 'stdin_fileno', 'stdout_fileno', 'stderr_fileno', 'getdefaultencoding', 'getfilesystemencoding', 'getrecursionlimit', 'setrecursionlimit', 'getrefcount', 'getsizeof', 'intern', 'exit', 'exitfunc', 'excepthook', 'displayhook', 'getprofile', 'setprofile', 'gettrace', 'settrace', 'getcheckinterval', 'setcheckinterval', 'getswitchinterval', 'setswitchinterval', 'getdlopenflags', 'setdlopenflags', 'getwindowsversion', 'get_asyncgen_hooks', 'set_asyncgen_hooks', 'get_coroutine_origin_tracking_depth', 'set_coroutine_origin_tracking_depth'],
                'math': ['pi', 'e', 'tau', 'inf', 'nan', 'ceil', 'copysign', 'fabs', 'factorial', 'floor', 'fmod', 'frexp', 'fsum', 'gcd', 'isclose', 'isfinite', 'isinf', 'isnan', 'ldexp', 'modf', 'trunc', 'exp', 'expm1', 'log', 'log1p', 'log2', 'log10', 'pow', 'sqrt', 'acos', 'asin', 'atan', 'atan2', 'cos', 'dist', 'hypot', 'sin', 'tan', 'acosh', 'asinh', 'atanh', 'cosh', 'sinh', 'tanh', 'erf', 'erfc', 'gamma', 'lgamma'],
                'json': ['dump', 'dumps', 'load', 'loads', 'JSONEncoder', 'JSONDecoder'],
                're': ['match', 'search', 'findall', 'finditer', 'sub', 'subn', 'split', 'escape', 'purge', 'compile', 'template', 'error', 'A', 'I', 'L', 'M', 'S', 'X', 'U', 'ASCII', 'IGNORECASE', 'LOCALE', 'MULTILINE', 'DOTALL', 'VERBOSE', 'UNICODE', 'DEBUG'],
                'datetime': ['date', 'datetime', 'time', 'timedelta', 'timezone', 'tzinfo', 'MINYEAR', 'MAXYEAR'],
                'collections': ['deque', 'defaultdict', 'namedtuple', 'Counter', 'OrderedDict', 'ChainMap', 'UserDict', 'UserList', 'UserString'],
                'itertools': ['count', 'cycle', 'repeat', 'accumulate', 'chain', 'chain.from_iterable', 'compress', 'dropwhile', 'filterfalse', 'groupby', 'islice', 'starmap', 'takewhile', 'tee', 'zip_longest', 'product', 'permutations', 'combinations', 'combinations_with_replacement'],
                'pathlib': ['Path', 'PurePath', 'PurePosixPath', 'PureWindowsPath', 'PosixPath', 'WindowsPath'],
                'typing': ['Any', 'Callable', 'Dict', 'List', 'Optional', 'Tuple', 'Union', 'TypeVar', 'Generic', 'Protocol', 'runtime_checkable', 'NoReturn', 'Literal', 'TypedDict', 'Final', 'Annotated', 'get_type_hints', 'cast', 'overload', 'Type', 'Text', 'AbstractSet', 'ByteString', 'Collection', 'Container', 'ItemsView', 'Iterable', 'Iterator', 'KeysView', 'Mapping', 'MappingView', 'MutableMapping', 'MutableSequence', 'MutableSet', 'Sequence', 'Set', 'ValuesView', 'Awaitable', 'AsyncIterator', 'AsyncIterable', 'Coroutine', 'Generator', 'AsyncGenerator', 'Reversible', 'SupportsAbs', 'SupportsBytes', 'SupportsComplex', 'SupportsFloat', 'SupportsIndex', 'SupportsInt', 'SupportsRound', 'ChainMap', 'Counter', 'Deque', 'Dict', 'DefaultDict', 'List', 'Set', 'FrozenSet', 'NamedTuple', 'TypedDict', 'Union', 'Optional', 'Literal', 'Final', 'Annotated', 'TypeVar', 'Generic', 'Protocol', 'runtime_checkable', 'NoReturn', 'get_type_hints', 'cast', 'overload', 'Type', 'Text', 'AbstractSet', 'ByteString', 'Collection', 'Container', 'ItemsView', 'Iterable', 'Iterator', 'KeysView', 'Mapping', 'MappingView', 'MutableMapping', 'MutableSequence', 'MutableSet', 'Sequence', 'Set', 'ValuesView', 'Awaitable', 'AsyncIterator', 'AsyncIterable', 'Coroutine', 'Generator', 'AsyncGenerator', 'Reversible', 'SupportsAbs', 'SupportsBytes', 'SupportsComplex', 'SupportsFloat', 'SupportsIndex', 'SupportsInt', 'SupportsRound']
            }
            
            if module_name in common_modules:
                return common_modules[module_name]
            
            # Try to import the module and get its __all__ attribute
            try:
                import importlib
                module = importlib.import_module(module_name)
                
                if hasattr(module, '__all__'):
                    return list(module.__all__)
                else:
                    # Get all public attributes (not starting with _)
                    return [attr for attr in dir(module) if not attr.startswith('_')]
                    
            except (ImportError, AttributeError):
                # If we can't import the module, return empty list
                return []
                
        except Exception as e:
            console.print(f"⚠️ Failed to resolve wildcard imports for {module_name}: {e}", style="yellow")
            return []
    
    def _detect_dynamic_imports(self, content: str) -> List[ImportInfo]:
        """Detect dynamic imports using regex patterns"""
        dynamic_imports = []
        
        # Pattern for importlib.import_module
        importlib_pattern = r'importlib\.import_module\([\'"]([^\'"]+)[\'"]'
        for match in re.finditer(importlib_pattern, content):
            module = match.group(1)
            dynamic_imports.append(ImportInfo(
                module=module,
                import_type="dynamic_import",
                line_number=0,  # Can't determine line number easily
                dynamic_import=True
            ))
        
        # Pattern for __import__
        dunder_import_pattern = r'__import__\([\'"]([^\'"]+)[\'"]'
        for match in re.finditer(dunder_import_pattern, content):
            module = match.group(1)
            dynamic_imports.append(ImportInfo(
                module=module,
                import_type="dynamic_import",
                line_number=0,
                dynamic_import=True
            ))
        
        # Pattern for exec/eval with imports
        exec_import_pattern = r'(?:exec|eval)\s*\([\'"](?:import|from).*?[\'"]'
        if re.search(exec_import_pattern, content):
            dynamic_imports.append(ImportInfo(
                module="unknown",
                import_type="exec_import",
                line_number=0,
                dynamic_import=True
            ))
        
        return dynamic_imports
    
    def _create_dependency_relationships(self, file_id: str, imports: List[ImportInfo], repo_id: str):
        """Create dependency relationships in Neo4j"""
        try:
            # Process imports in smaller batches to avoid session issues
            batch_size = 10
            for i in range(0, len(imports), batch_size):
                batch = imports[i:i + batch_size]
                
                with self.driver.session() as session:
                    for import_info in batch:
                        try:
                            # Create module node
                            session.run("""
                                MERGE (m:Module {name: $module_name})
                                SET m.last_seen = datetime()
                            """, module_name=import_info.module)
                            
                            # Create relationship based on import type
                            if import_info.import_type == "direct":
                                session.run("""
                                    MATCH (f:File {path: $file_id})
                                    MATCH (m:Module {name: $module_name})
                                    MERGE (f)-[r:IMPORTS]->(m)
                                    SET r.line = $line,
                                        r.alias = $alias
                                """, file_id=file_id,
                                     module_name=import_info.module,
                                     line=import_info.line_number,
                                     alias=import_info.alias)
                            
                            elif import_info.import_type == "from_import":
                                session.run("""
                                    MATCH (f:File {path: $file_id})
                                    MATCH (m:Module {name: $module_name})
                                    MERGE (f)-[r:IMPORTS_FROM]->(m)
                                    SET r.line = $line,
                                        r.alias = $alias
                                """, file_id=file_id,
                                     module_name=import_info.module,
                                     line=import_info.line_number,
                                     alias=import_info.alias)
                                
                                # Create specific item relationships if items are specified
                                if import_info.items:
                                    for item in import_info.items:
                                        # Create function/class node
                                        full_name = f"{import_info.module}.{item}"
                                        session.run("""
                                            MERGE (func:Function {full_name: $full_name})
                                            SET func.name = $item_name,
                                                func.module = $module_name,
                                                func.type = 'unknown'
                                        """, full_name=full_name,
                                                 item_name=item,
                                                 module_name=import_info.module)
                                        
                                        # Create CONTAINS relationship
                                        session.run("""
                                            MATCH (m:Module {name: $module_name})
                                            MATCH (func:Function {full_name: $full_name})
                                            MERGE (m)-[r:CONTAINS]->(func)
                                        """, module_name=import_info.module,
                                                 full_name=full_name)
                                        
                                        # Create IMPORTS_SPECIFIC relationship
                                        session.run("""
                                            MATCH (f:File {path: $file_id})
                                            MATCH (func:Function {full_name: $full_name})
                                            MERGE (f)-[r:IMPORTS_SPECIFIC]->(func)
                                            SET r.line = $line,
                                                r.alias = $alias
                                        """, file_id=file_id,
                                                 full_name=full_name,
                                                 line=import_info.line_number,
                                                 alias=import_info.alias)
                            
                            elif import_info.import_type == "wildcard_import":
                                session.run("""
                                    MATCH (f:File {path: $file_id})
                                    MATCH (m:Module {name: $module_name})
                                    MERGE (f)-[r:WILDCARD_IMPORTS]->(m)
                                    SET r.line = $line,
                                        r.wildcard_import = true
                                """, file_id=file_id,
                                     module_name=import_info.module,
                                     line=import_info.line_number)
                            
                            elif import_info.import_type == "dynamic_import":
                                session.run("""
                                    MATCH (f:File {path: $file_id})
                                    MATCH (m:Module {name: $module_name})
                                    MERGE (f)-[r:DYNAMIC_IMPORTS]->(m)
                                    SET r.line = $line,
                                        r.dynamic_import = true
                                """, file_id=file_id,
                                     module_name=import_info.module,
                                     line=import_info.line_number)
                            
                            elif import_info.import_type == "exec_import":
                                session.run("""
                                    MATCH (f:File {path: $file_id})
                                    MERGE (m:Module {name: 'unknown'})
                                    MERGE (f)-[r:EXEC_IMPORTS]->(m)
                                    SET r.line = $line,
                                        r.exec_import = true
                                """, file_id=file_id,
                                     line=import_info.line_number)
                            
                            else:  # relative imports, namespace imports, etc.
                                session.run("""
                                    MATCH (f:File {path: $file_id})
                                    MATCH (m:Module {name: $module_name})
                                    MERGE (f)-[r:RELATIVE_IMPORTS]->(m)
                                    SET r.line = $line,
                                        r.alias = $alias,
                                        r.import_type = $import_type
                                """, file_id=file_id,
                                     module_name=import_info.module,
                                     line=import_info.line_number,
                                     alias=import_info.alias,
                                     import_type=import_info.import_type)
                        
                        except Exception as e:
                            console.print(f"⚠️ Failed to process import {import_info.module}: {e}", style="yellow")
                            continue
                
        except Exception as e:
            console.print(f"❌ Failed to create dependency relationships: {e}", style="red")
            raise
    
    def _parse_dependency_files(self, repo_path: Path, repo_id: str) -> Dict[str, Any]:
        """Parse external dependency files (requirements.txt, package.json, etc.)"""
        try:
            external_packages = []
            
            # Python requirements
            requirements_files = list(repo_path.glob("**/requirements*.txt"))
            for req_file in requirements_files:
                packages = self._parse_requirements_file(req_file)
                for package in packages:
                    package["source_file"] = str(req_file.relative_to(repo_path))
                    external_packages.append(package)
            
            # Node.js package.json
            package_json_files = list(repo_path.glob("**/package.json"))
            for pkg_file in package_json_files:
                packages = self._parse_package_json(pkg_file)
                for package in packages:
                    package["source_file"] = str(pkg_file.relative_to(repo_path))
                    external_packages.append(package)
            
            # Go go.mod
            go_mod_files = list(repo_path.glob("**/go.mod"))
            for go_mod in go_mod_files:
                packages = self._parse_go_mod(go_mod)
                for package in packages:
                    package["source_file"] = str(go_mod.relative_to(repo_path))
                    external_packages.append(package)
            
            # Create package nodes and relationships
            self._create_package_relationships(external_packages, repo_id)
            
            return {"external_packages": len(external_packages)}
            
        except Exception as e:
            console.print(f"❌ Failed to parse dependency files: {e}", style="red")
            raise
    
    def _parse_requirements_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Parse Python requirements.txt file"""
        packages = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse package specification
                        if '==' in line:
                            name, version = line.split('==', 1)
                        elif '>=' in line:
                            name, version = line.split('>=', 1)
                        elif '<=' in line:
                            name, version = line.split('<=', 1)
                        elif '~=' in line:
                            name, version = line.split('~=', 1)
                        else:
                            name, version = line, None
                        
                        packages.append({
                            "name": name.strip(),
                            "version": version.strip() if version else None,
                            "type": "python_package"
                        })
        except Exception as e:
            console.print(f"⚠️ Failed to parse requirements file {file_path}: {e}", style="yellow")
        
        return packages
    
    def _parse_package_json(self, file_path: Path) -> List[Dict[str, str]]:
        """Parse Node.js package.json file"""
        packages = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Parse dependencies
                for dep_type in ['dependencies', 'devDependencies', 'peerDependencies']:
                    if dep_type in data:
                        for name, version in data[dep_type].items():
                            packages.append({
                                "name": name,
                                "version": version,
                                "type": f"npm_{dep_type.replace('Dependencies', '')}"
                            })
        except Exception as e:
            console.print(f"⚠️ Failed to parse package.json {file_path}: {e}", style="yellow")
        
        return packages
    
    def _parse_go_mod(self, file_path: Path) -> List[Dict[str, str]]:
        """Parse Go go.mod file"""
        packages = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('require '):
                        # Parse require statement
                        parts = line.split()
                        if len(parts) >= 3:
                            name = parts[1]
                            version = parts[2] if len(parts) > 2 else None
                            packages.append({
                                "name": name,
                                "version": version,
                                "type": "go_module"
                            })
        except Exception as e:
            console.print(f"⚠️ Failed to parse go.mod {file_path}: {e}", style="yellow")
        
        return packages
    
    def _create_package_relationships(self, packages: List[Dict[str, str]], repo_id: str):
        """Create package nodes and relationships in Neo4j with deep dependency resolution"""
        try:
            with self.driver.session() as session:
                for package in packages:
                    # Create package node
                    session.run("""
                        MERGE (p:Package {name: $name})
                        SET p.version = $version,
                            p.type = $type,
                            p.last_seen = datetime()
                    """, name=package["name"],
                         version=package.get("version"),
                         type=package["type"])
                    
                    # Create relationship to repository
                    session.run("""
                        MATCH (r:Repository {url: $repo_id})
                        MATCH (p:Package {name: $name})
                        MERGE (r)-[rel:USES_PACKAGE]->(p)
                        SET rel.source_file = $source_file,
                            rel.added_at = datetime()
                    """, repo_id=repo_id,
                         name=package["name"],
                         source_file=package.get("source_file"))
                    
                    # Resolve and create transitive dependencies
                    transitive_deps = self._resolve_deep_dependencies(package["name"])
                    for dep_name in transitive_deps:
                        if dep_name != package["name"]:  # Don't create self-reference
                            # Create transitive package node
                            session.run("""
                                MERGE (tp:Package {name: $name})
                                SET tp.type = 'transitive_dependency',
                                    tp.last_seen = datetime()
                            """, name=dep_name)
                            
                            # Create transitive dependency relationship
                            session.run("""
                                MATCH (p:Package {name: $package_name})
                                MATCH (tp:Package {name: $dep_name})
                                MERGE (p)-[rel:TRANSITIVE_DEPENDS_ON]->(tp)
                                SET rel.added_at = datetime()
                            """, package_name=package["name"], dep_name=dep_name)
                
        except Exception as e:
            console.print(f"❌ Failed to create package relationships: {e}", style="red")
            raise
    
    def _resolve_deep_dependencies(self, package_name: str) -> Set[str]:
        """Resolve all transitive dependencies for a package"""
        try:
            # Try to use pkg_resources for Python packages
            try:
                import pkg_resources
                dist = pkg_resources.get_distribution(package_name)
                if dist:
                    requires = dist.requires() if dist else []
                    all_deps = {package_name}
                    
                    for req in requires:
                        dep_name = req.name
                        all_deps.add(dep_name)
                        # Recursively resolve dependencies (with depth limit)
                        all_deps.update(self._resolve_deep_dependencies_limited(dep_name, depth=2))
                    
                    return all_deps
                    
            except (pkg_resources.DistributionNotFound, ImportError):
                pass
            
            # Fallback: try to import and inspect
            try:
                import importlib
                module = importlib.import_module(package_name)
                return self._analyze_module_imports(module)
            except (ImportError, AttributeError):
                pass
            
            # If all else fails, return just the package itself
            return {package_name}
                
        except Exception as e:
            console.print(f"⚠️ Failed to resolve deep dependencies for {package_name}: {e}", style="yellow")
            return {package_name}
    
    def _resolve_deep_dependencies_limited(self, package_name: str, depth: int) -> Set[str]:
        """Resolve dependencies with depth limit to prevent infinite recursion"""
        if depth <= 0:
            return set()
        
        try:
            import pkg_resources
            dist = pkg_resources.get_distribution(package_name)
            if dist:
                requires = dist.requires() if dist else []
                deps = set()
                
                for req in requires:
                    dep_name = req.name
                    deps.add(dep_name)
                    # Recursively resolve with reduced depth
                    deps.update(self._resolve_deep_dependencies_limited(dep_name, depth - 1))
                
                return deps
                
        except (pkg_resources.DistributionNotFound, ImportError):
            pass
        
        return set()
    
    def _analyze_module_imports(self, module) -> Set[str]:
        """Analyze a module for its imports"""
        try:
            import inspect
            import ast
            
            # Get module source if available
            try:
                source = inspect.getsource(module)
                tree = ast.parse(source)
                
                imports = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])  # Get top-level package
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])
                
                return imports
                
            except (OSError, TypeError):
                # Can't get source, try to analyze module attributes
                return set()
                
        except Exception as e:
            console.print(f"⚠️ Failed to analyze module imports: {e}", style="yellow")
            return set()
    
    def _build_dependency_graph(self, repo_id: str) -> Dict[str, int]:
        """Build and analyze the dependency graph"""
        try:
            with self.driver.session() as session:
                # Count nodes and relationships separately to avoid syntax issues
                result = session.run("""
                    MATCH (r:Repository {url: $repo_id})
                    OPTIONAL MATCH (r)-[:BELONGS_TO]-(f:File)
                    OPTIONAL MATCH (f)-[:IMPORTS]->(m:Module)
                    OPTIONAL MATCH (f)-[:IMPORTS_FROM]->(m2:Module)
                    OPTIONAL MATCH (f)-[:IMPORTS_SPECIFIC]->(func:Function)
                    OPTIONAL MATCH (r)-[:USES_PACKAGE]->(p:Package)
                    RETURN count(DISTINCT f) as files,
                           count(DISTINCT m) + count(DISTINCT m2) as modules,
                           count(DISTINCT p) as packages,
                           count(DISTINCT func) as functions
                """, repo_id=repo_id)
                
                # Count relationships separately
                rel_result = session.run("""
                    MATCH (r:Repository {url: $repo_id})
                    OPTIONAL MATCH (r)-[:BELONGS_TO]-(f:File)
                    OPTIONAL MATCH (f)-[r1:IMPORTS]->()
                    OPTIONAL MATCH (f)-[r2:IMPORTS_FROM]->()
                    OPTIONAL MATCH (f)-[r3:IMPORTS_SPECIFIC]->()
                    RETURN count(r1) + count(r2) + count(r3) as dependencies
                """, repo_id=repo_id)
                
                stats = result.single()
                rel_stats = rel_result.single()
                
                return {
                    "nodes": stats["files"] + stats["modules"] + stats["packages"] + stats["functions"],
                    "relationships": rel_stats["dependencies"]
                }
                
        except Exception as e:
            console.print(f"❌ Failed to build dependency graph: {e}", style="red")
            raise
    
    def get_dependency_statistics(self, repo_url: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about the dependency graph"""
        try:
            with self.driver.session() as session:
                if repo_url:
                    # Statistics for specific repository
                    result = session.run("""
                        MATCH (r:Repository {url: $repo_url})
                        OPTIONAL MATCH (r)-[:BELONGS_TO]-(f:File)
                        OPTIONAL MATCH (f)-[:IMPORTS]->(m:Module)
                        OPTIONAL MATCH (f)-[:IMPORTS_FROM]->(m2:Module)
                        OPTIONAL MATCH (f)-[:IMPORTS_SPECIFIC]->(func:Function)
                        OPTIONAL MATCH (r)-[:USES_PACKAGE]->(p:Package)
                        RETURN count(DISTINCT f) as files,
                               count(DISTINCT m) + count(DISTINCT m2) as modules,
                               count(DISTINCT p) as packages,
                               count(DISTINCT func) as functions
                    """, repo_url=repo_url)
                    
                    # Count relationships separately
                    rel_result = session.run("""
                        MATCH (r:Repository {url: $repo_url})
                        OPTIONAL MATCH (r)-[:BELONGS_TO]-(f:File)
                        OPTIONAL MATCH (f)-[r1:IMPORTS]->()
                        OPTIONAL MATCH (f)-[r2:IMPORTS_FROM]->()
                        OPTIONAL MATCH (f)-[r3:IMPORTS_SPECIFIC]->()
                        RETURN count(r1) + count(r2) + count(r3) as dependencies
                    """, repo_url=repo_url)
                    
                    stats = dict(result.single())
                    stats["dependencies"] = rel_result.single()["dependencies"]
                    return stats
                    
                else:
                    # Global statistics
                    result = session.run("""
                        MATCH (f:File)
                        OPTIONAL MATCH (f)-[:IMPORTS]->(m:Module)
                        OPTIONAL MATCH (f)-[:IMPORTS_FROM]->(m2:Module)
                        OPTIONAL MATCH (f)-[:IMPORTS_SPECIFIC]->(func:Function)
                        OPTIONAL MATCH (r:Repository)-[:USES_PACKAGE]->(p:Package)
                        RETURN count(DISTINCT f) as files,
                               count(DISTINCT m) + count(DISTINCT m2) as modules,
                               count(DISTINCT p) as packages,
                               count(DISTINCT func) as functions,
                               count(DISTINCT r) as repositories
                    """)
                    
                    # Count relationships separately
                    rel_result = session.run("""
                        MATCH (f:File)
                        OPTIONAL MATCH (f)-[r1:IMPORTS]->()
                        OPTIONAL MATCH (f)-[r2:IMPORTS_FROM]->()
                        OPTIONAL MATCH (f)-[r3:IMPORTS_SPECIFIC]->()
                        RETURN count(r1) + count(r2) + count(r3) as dependencies
                    """)
                    
                    stats = dict(result.single())
                    stats["dependencies"] = rel_result.single()["dependencies"]
                    return stats
                
        except Exception as e:
            console.print(f"❌ Failed to get dependency statistics: {e}", style="red")
            raise
    
    def display_statistics_table(self, repo_url: Optional[str] = None):
        """Display dependency statistics in a nice table format"""
        try:
            stats = self.get_dependency_statistics(repo_url)
            
            table = Table(title="Dependency Graph Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", style="magenta")
            
            for key, value in stats.items():
                table.add_row(key.replace('_', ' ').title(), str(value))
            
            console.print(table)
            
        except Exception as e:
            console.print(f"❌ Failed to display statistics: {e}", style="red")
    
    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            console.print("✅ Neo4j connection closed", style="green")


def main():
    """Demo function showing how to use the Dependency Scanner"""
    
    # Test repositories - start with smaller ones
    test_repos = [
        r"C:\Projects\requests",  # Small Python project
        r"C:\Projects\openai-python",  # Medium Python project
        # r"C:\Projects\react",  # Large JS project - commented out for now
        # r"C:\Projects\langchain"  # Very large Python project - commented out for now
    ]
    
    console.print("=== GitHub Repository Dependency Scanner Demo ===\n", style="bold blue")
    console.print("💡 Tip: Press Ctrl+C to stop the scan at any time", style="yellow")
    
    try:
        # Initialize scanner
        scanner = DependencyScanner()
        
        # Scan repositories
        for i, repo_path in enumerate(test_repos, 1):
            console.print(f"\n🔍 Scanning repository {i}/{len(test_repos)}: {repo_path}", style="bold")
            try:
                results = scanner.scan_repository(repo_path)
                console.print(f"✅ Scan completed successfully", style="green")
                
                # Display statistics
                scanner.display_statistics_table(results["repository"])
                
            except KeyboardInterrupt:
                console.print(f"\n⚠️ Scan interrupted by user", style="yellow")
                break
            except Exception as e:
                console.print(f"❌ Failed to scan {repo_path}: {e}", style="red")
                console.print("Continuing with next repository...", style="yellow")
        
        # Display global statistics
        console.print("\n📊 Global Dependency Graph Statistics:", style="bold")
        scanner.display_statistics_table()
        
        scanner.close()
        
    except KeyboardInterrupt:
        console.print(f"\n⚠️ Program interrupted by user", style="yellow")
        try:
            scanner.close()
        except:
            pass
    except Exception as e:
        console.print(f"❌ Demo failed: {e}", style="red")
        console.print("Please check your Neo4j connection and try again.", style="yellow")


if __name__ == "__main__":
    main() 