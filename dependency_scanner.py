"""
GitHub Repository Dependency Scanner

This script scans GitHub repositories and builds dependency graphs using Neo4j.
It parses import statements across different file types and stores the dependency
graph with nodes representing files/modules and edges representing dependencies.

Required packages:
pip install neo4j requests gitpython ast pathlib typing-extensions
"""

import os
import ast
import re
import json
import requests
import git
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
from urllib.parse import urlparse
from datetime import datetime
import logging

# Neo4j imports
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError


class DependencyScanner:
    """
    Scans GitHub repositories and builds dependency graphs in Neo4j
    """
    
    def __init__(self, 
                 neo4j_uri: str,
                 neo4j_username: str,
                 neo4j_password: str,
                 github_token: Optional[str] = None):
        """
        Initialize the dependency scanner
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            github_token: GitHub API token for authenticated requests
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.github_token = github_token
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # File extensions to scan
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
        
        # Initialize database schema
        self._setup_database_schema()
        
    def _setup_database_schema(self):
        """Setup Neo4j database schema with constraints and indexes"""
        try:
            with self.driver.session() as session:
                # Create constraints for unique nodes
                session.run("CREATE CONSTRAINT file_path IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE")
                session.run("CREATE CONSTRAINT module_name IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE")
                session.run("CREATE CONSTRAINT package_name IF NOT EXISTS FOR (p:Package) REQUIRE p.name IS UNIQUE")
                session.run("CREATE CONSTRAINT repository_url IF NOT EXISTS FOR (r:Repository) REQUIRE r.url IS UNIQUE")
                
                # Create indexes for better query performance
                session.run("CREATE INDEX file_extension IF NOT EXISTS FOR (f:File) ON (f.extension)")
                session.run("CREATE INDEX file_language IF NOT EXISTS FOR (f:File) ON (f.language)")
                session.run("CREATE INDEX dependency_type IF NOT EXISTS FOR ()-[d:DEPENDS_ON]-() ON (d.type)")
                
                self.logger.info("✅ Database schema setup completed")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to setup database schema: {e}")
            raise
    
    def scan_github_repository(self, 
                              repo_url: str, 
                              branch: str = "main",
                              local_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Scan a GitHub repository and build its dependency graph
        
        Args:
            repo_url: GitHub repository URL
            branch: Branch to scan (default: main)
            local_path: Optional local path to clone to
            
        Returns:
            Dictionary with scan results and statistics
        """
        try:
            self.logger.info(f"🔍 Starting scan of repository: {repo_url}")
            
            # Clone or update repository
            repo_path = self._get_repository(repo_url, branch, local_path)
            
            # Create repository node
            repo_id = self._create_repository_node(repo_url, repo_path)
            
            # Scan all files in the repository
            scan_results = self._scan_repository_files(repo_path, repo_id)
            
            # Parse dependency files
            dependency_results = self._parse_dependency_files(repo_path, repo_id)
            
            # Build dependency graph
            graph_results = self._build_dependency_graph(repo_id)
            
            # Generate scan summary
            summary = {
                "repository": repo_url,
                "branch": branch,
                "scan_timestamp": datetime.now().isoformat(),
                "files_scanned": scan_results["files_scanned"],
                "dependencies_found": scan_results["dependencies_found"],
                "external_packages": dependency_results["external_packages"],
                "graph_nodes": graph_results["nodes"],
                "graph_relationships": graph_results["relationships"]
            }
            
            self.logger.info(f"✅ Repository scan completed: {summary}")
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Failed to scan repository {repo_url}: {e}")
            raise
    
    def _get_repository(self, repo_url: str, branch: str, local_path: Optional[str]) -> str:
        """Clone or update a GitHub repository"""
        try:
            # Parse repository URL
            parsed_url = urlparse(repo_url)
            repo_name = parsed_url.path.strip('/').split('/')[-1]
            
            # Determine local path
            if local_path is None:
                local_path = f"./temp_repos/{repo_name}"
            
            repo_path = Path(local_path)
            repo_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Clone or pull repository
            if repo_path.exists() and (repo_path / ".git").exists():
                self.logger.info(f"📥 Updating existing repository at {repo_path}")
                repo = git.Repo(repo_path)
                repo.remotes.origin.pull()
            else:
                self.logger.info(f"📥 Cloning repository to {repo_path}")
                repo = git.Repo.clone_from(repo_url, repo_path)
            
            # Checkout specified branch
            repo = git.Repo(repo_path)
            repo.git.checkout(branch)
            
            return str(repo_path)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get repository: {e}")
            raise
    
    def _create_repository_node(self, repo_url: str, repo_path: str) -> str:
        """Create a repository node in Neo4j"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MERGE (r:Repository {url: $url})
                    SET r.path = $path,
                        r.last_scan = datetime(),
                        r.name = $name
                    RETURN r.url as repo_id
                """, url=repo_url, path=repo_path, name=Path(repo_path).name)
                
                repo_id = result.single()["repo_id"]
                self.logger.info(f"✅ Created repository node: {repo_id}")
                return repo_id
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create repository node: {e}")
            raise
    
    def _scan_repository_files(self, repo_path: str, repo_id: str) -> Dict[str, Any]:
        """Scan all files in the repository for dependencies"""
        try:
            repo_path_obj = Path(repo_path)
            files_scanned = 0
            dependencies_found = 0
            
            # Walk through all files in the repository
            for file_path in repo_path_obj.rglob("*"):
                if file_path.is_file() and file_path.suffix in self.supported_extensions:
                    try:
                        # Create file node
                        file_id = self._create_file_node(file_path, repo_id)
                        files_scanned += 1
                        
                        # Parse dependencies
                        dependencies = self._parse_file_dependencies(file_path)
                        if dependencies:
                            self._create_dependency_relationships(file_id, dependencies, repo_id)
                            dependencies_found += len(dependencies)
                            
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to process file {file_path}: {e}")
                        continue
            
            return {
                "files_scanned": files_scanned,
                "dependencies_found": dependencies_found
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to scan repository files: {e}")
            raise
    
    def _create_file_node(self, file_path: Path, repo_id: str) -> str:
        """Create a file node in Neo4j"""
        try:
            with self.driver.session() as session:
                # Get relative path from repository root
                repo_path = Path(repo_path).parent
                relative_path = str(file_path.relative_to(repo_path))
                
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
                     language=self._get_language(file_path.suffix),
                     size=file_path.stat().st_size,
                     repo_id=repo_id)
                
                file_id = result.single()["file_id"]
                return file_id
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create file node: {e}")
            raise
    
    def _get_language(self, extension: str) -> str:
        """Get programming language from file extension"""
        language_map = {
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
        return language_map.get(extension, 'Unknown')
    
    def _parse_file_dependencies(self, file_path: Path) -> List[Dict[str, str]]:
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
            dependencies = parser_func(content, str(file_path))
            
            return dependencies
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to parse dependencies from {file_path}: {e}")
            return []
    
    def _parse_python_imports(self, content: str, file_path: str) -> List[Dict[str, str]]:
        """Parse Python import statements"""
        dependencies = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append({
                            "type": "direct_import",
                            "module": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno
                        })
                        
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        dependencies.append({
                            "type": "from_import",
                            "module": module,
                            "name": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno
                        })
                        
        except SyntaxError:
            # Handle syntax errors by using regex fallback
            dependencies.extend(self._parse_python_imports_regex(content))
        
        return dependencies
    
    def _parse_python_imports_regex(self, content: str) -> List[Dict[str, str]]:
        """Fallback regex parser for Python imports"""
        dependencies = []
        
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
                dependencies.append({
                    "type": "direct_import",
                    "module": module,
                    "alias": alias,
                    "line": line_num
                })
            
            # Check from import statements
            from_match = re.match(from_pattern, line)
            if from_match:
                module = from_match.group(1)
                imports = from_match.group(2)
                
                # Parse multiple imports
                for import_item in imports.split(','):
                    import_item = import_item.strip()
                    if ' as ' in import_item:
                        name, alias = import_item.split(' as ', 1)
                    else:
                        name, alias = import_item, None
                    
                    dependencies.append({
                        "type": "from_import",
                        "module": module,
                        "name": name,
                        "alias": alias,
                        "line": line_num
                    })
        
        return dependencies
    
    def _parse_javascript_imports(self, content: str, file_path: str) -> List[Dict[str, str]]:
        """Parse JavaScript/JSX import statements"""
        dependencies = []
        
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
                            imports, module = match
                            if imports.startswith('{') and imports.endswith('}'):
                                # Named imports
                                named_imports = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*))?', imports)
                                for name, alias in named_imports:
                                    dependencies.append({
                                        "type": "named_import",
                                        "module": module,
                                        "name": name,
                                        "alias": alias,
                                        "line": line_num
                                    })
                            else:
                                # Default import
                                dependencies.append({
                                    "type": "default_import",
                                    "module": module,
                                    "name": imports,
                                    "line": line_num
                                })
                        else:
                            # Namespace import
                            namespace, module = match
                            dependencies.append({
                                "type": "namespace_import",
                                "module": module,
                                "namespace": namespace,
                                "line": line_num
                            })
                    else:
                        # Simple import
                        dependencies.append({
                            "type": "simple_import",
                            "module": match,
                            "line": line_num
                        })
        
        return dependencies
    
    def _parse_typescript_imports(self, content: str, file_path: str) -> List[Dict[str, str]]:
        """Parse TypeScript imports (same as JavaScript for now)"""
        return self._parse_javascript_imports(content, file_path)
    
    def _parse_java_imports(self, content: str, file_path: str) -> List[Dict[str, str]]:
        """Parse Java import statements"""
        dependencies = []
        
        import_pattern = r'^import\s+([a-zA-Z_][a-zA-Z0-9_.]*)(?:\s*;)?'
        
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            
            match = re.match(import_pattern, line)
            if match:
                module = match.group(1)
                dependencies.append({
                    "type": "java_import",
                    "module": module,
                    "line": line_num
                })
        
        return dependencies
    
    def _parse_go_imports(self, content: str, file_path: str) -> List[Dict[str, str]]:
        """Parse Go import statements"""
        dependencies = []
        
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
                        dependencies.append({
                            "type": "go_import",
                            "module": package,
                            "line": 0  # Line number not available for blocks
                        })
        
        # Check for single imports
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            match = re.match(single_import_pattern, line)
            if match:
                package = match.group(1)
                dependencies.append({
                    "type": "go_import",
                    "module": package,
                    "line": line_num
                })
        
        return dependencies
    
    def _parse_rust_imports(self, content: str, file_path: str) -> List[Dict[str, str]]:
        """Parse Rust use statements"""
        dependencies = []
        
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
                        dependencies.append({
                            "type": "rust_use",
                            "module": module,
                            "line": line_num
                        })
                    else:
                        module, item = match.groups()
                        dependencies.append({
                            "type": "rust_use_item",
                            "module": module,
                            "item": item,
                            "line": line_num
                        })
        
        return dependencies
    
    def _parse_php_imports(self, content: str, file_path: str) -> List[Dict[str, str]]:
        """Parse PHP use statements"""
        dependencies = []
        
        use_pattern = r'^use\s+([a-zA-Z_][a-zA-Z0-9_\\]*)(?:\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*))?(?:\s*;)?'
        
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            
            match = re.match(use_pattern, line)
            if match:
                namespace = match.group(1)
                alias = match.group(2)
                dependencies.append({
                    "type": "php_use",
                    "module": namespace,
                    "alias": alias,
                    "line": line_num
                })
        
        return dependencies
    
    def _parse_ruby_imports(self, content: str, file_path: str) -> List[Dict[str, str]]:
        """Parse Ruby require/load statements"""
        dependencies = []
        
        require_pattern = r'^(?:require|load)\s+[\'"]([^\'"]+)[\'"]'
        
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            
            match = re.match(require_pattern, line)
            if match:
                module = match.group(1)
                dependencies.append({
                    "type": "ruby_require",
                    "module": module,
                    "line": line_num
                })
        
        return dependencies
    
    def _create_dependency_relationships(self, file_id: str, dependencies: List[Dict[str, str]], repo_id: str):
        """Create dependency relationships in Neo4j"""
        try:
            with self.driver.session() as session:
                for dep in dependencies:
                    # Create module node
                    session.run("""
                        MERGE (m:Module {name: $module_name})
                        SET m.type = $dep_type,
                            m.last_seen = datetime()
                    """, module_name=dep["module"], dep_type=dep["type"])
                    
                    # Create dependency relationship
                    session.run("""
                        MATCH (f:File {path: $file_id})
                        MATCH (m:Module {name: $module_name})
                        MERGE (f)-[r:DEPENDS_ON]->(m)
                        SET r.type = $dep_type,
                            r.line = $line,
                            r.alias = $alias,
                            r.name = $name
                    """, file_id=file_id,
                         module_name=dep["module"],
                         dep_type=dep["type"],
                         line=dep.get("line", 0),
                         alias=dep.get("alias"),
                         name=dep.get("name"))
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create dependency relationships: {e}")
            raise
    
    def _parse_dependency_files(self, repo_path: str, repo_id: str) -> Dict[str, Any]:
        """Parse external dependency files (requirements.txt, package.json, etc.)"""
        try:
            repo_path_obj = Path(repo_path)
            external_packages = []
            
            # Python requirements
            requirements_files = list(repo_path_obj.glob("**/requirements*.txt"))
            for req_file in requirements_files:
                packages = self._parse_requirements_file(req_file)
                for package in packages:
                    package["source_file"] = str(req_file.relative_to(repo_path_obj))
                    external_packages.append(package)
            
            # Node.js package.json
            package_json_files = list(repo_path_obj.glob("**/package.json"))
            for pkg_file in package_json_files:
                packages = self._parse_package_json(pkg_file)
                for package in packages:
                    package["source_file"] = str(pkg_file.relative_to(repo_path_obj))
                    external_packages.append(package)
            
            # Go go.mod
            go_mod_files = list(repo_path_obj.glob("**/go.mod"))
            for go_mod in go_mod_files:
                packages = self._parse_go_mod(go_mod)
                for package in packages:
                    package["source_file"] = str(go_mod.relative_to(repo_path_obj))
                    external_packages.append(package)
            
            # Create package nodes and relationships
            self._create_package_relationships(external_packages, repo_id)
            
            return {"external_packages": len(external_packages)}
            
        except Exception as e:
            self.logger.error(f"❌ Failed to parse dependency files: {e}")
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
            self.logger.warning(f"⚠️ Failed to parse requirements file {file_path}: {e}")
        
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
            self.logger.warning(f"⚠️ Failed to parse package.json {file_path}: {e}")
        
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
            self.logger.warning(f"⚠️ Failed to parse go.mod {file_path}: {e}")
        
        return packages
    
    def _create_package_relationships(self, packages: List[Dict[str, str]], repo_id: str):
        """Create package nodes and relationships in Neo4j"""
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
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create package relationships: {e}")
            raise
    
    def _build_dependency_graph(self, repo_id: str) -> Dict[str, int]:
        """Build and analyze the dependency graph"""
        try:
            with self.driver.session() as session:
                # Count nodes and relationships
                result = session.run("""
                    MATCH (r:Repository {url: $repo_id})
                    OPTIONAL MATCH (r)-[:BELONGS_TO]-(f:File)
                    OPTIONAL MATCH (f)-[:DEPENDS_ON]->(m:Module)
                    OPTIONAL MATCH (r)-[:USES_PACKAGE]->(p:Package)
                    RETURN count(DISTINCT f) as files,
                           count(DISTINCT m) as modules,
                           count(DISTINCT p) as packages,
                           count(DISTINCT (f)-[:DEPENDS_ON]->(m)) as dependencies
                """, repo_id=repo_id)
                
                stats = result.single()
                
                return {
                    "nodes": stats["files"] + stats["modules"] + stats["packages"],
                    "relationships": stats["dependencies"]
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to build dependency graph: {e}")
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
                        OPTIONAL MATCH (f)-[:DEPENDS_ON]->(m:Module)
                        OPTIONAL MATCH (r)-[:USES_PACKAGE]->(p:Package)
                        RETURN count(DISTINCT f) as files,
                               count(DISTINCT m) as modules,
                               count(DISTINCT p) as packages,
                               count(DISTINCT (f)-[:DEPENDS_ON]->(m)) as dependencies
                    """, repo_url=repo_url)
                else:
                    # Global statistics
                    result = session.run("""
                        MATCH (f:File)
                        OPTIONAL MATCH (f)-[:DEPENDS_ON]->(m:Module)
                        OPTIONAL MATCH (r:Repository)-[:USES_PACKAGE]->(p:Package)
                        RETURN count(DISTINCT f) as files,
                               count(DISTINCT m) as modules,
                               count(DISTINCT p) as packages,
                               count(DISTINCT (f)-[:DEPENDS_ON]->(m)) as dependencies,
                               count(DISTINCT r) as repositories
                    """)
                
                return dict(result.single())
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get dependency statistics: {e}")
            raise
    
    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()


def main():
    """Demo function showing how to use the Dependency Scanner"""
    
    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Optional
    
    # Sample repositories to scan
    sample_repos = [
        "https://github.com/langchain-ai/langchain",
        "https://github.com/openai/openai-python",
        "https://github.com/neo4j/neo4j-python-driver"
    ]
    
    print("=== GitHub Repository Dependency Scanner Demo ===\n")
    
    try:
        # Initialize scanner
        scanner = DependencyScanner(
            neo4j_uri=NEO4J_URI,
            neo4j_username=NEO4J_USERNAME,
            neo4j_password=NEO4J_PASSWORD,
            github_token=GITHUB_TOKEN
        )
        
        # Scan repositories
        for repo_url in sample_repos:
            print(f"\n🔍 Scanning repository: {repo_url}")
            try:
                results = scanner.scan_github_repository(repo_url)
                print(f"✅ Scan completed: {results}")
            except Exception as e:
                print(f"❌ Failed to scan {repo_url}: {e}")
        
        # Get statistics
        print("\n📊 Dependency Graph Statistics:")
        stats = scanner.get_dependency_statistics()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        scanner.close()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Please check your Neo4j connection and try again.")


if __name__ == "__main__":
    main() 