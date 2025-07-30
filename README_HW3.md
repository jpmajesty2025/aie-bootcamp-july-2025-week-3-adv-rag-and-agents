# 📅 Advanced Rag and Agentic AI

### GitHub Repository Dependency Scanner --- file to submit is dependency_scanner.py

- With Python, create a script that scans GitHub repositories and builds dependency graphs using Neo4j
  - Parse import statements across different file types (Python, JavaScript, TypeScript, etc.)
  - Store the dependency graph in Neo4j with nodes representing files/modules and edges representing dependencies
  - Support different dependency types:
    - Direct imports (from module import function)
    - Relative imports (from .local_module import something)
    - External package dependencies (requirements.txt, package.json, etc.)
  - Use Neo4j's graph database capabilities for efficient querying and traversal of dependencies
  - Monitor repository changes and update the dependency graph accordingly


### Agentic Vulnerability Analysis and Remediation --- file to submit is vulnerability_system.py

- With Python, create a comprehensive script that combines vulnerability detection, analysis, and remediation:
  
  **Vulnerability Detection and Analysis:**
  - Integrate with vulnerability databases (CVE, GitHub Security Advisories, Snyk, etc.)
  - Scan external dependencies for known vulnerabilities
  - **Agentic Graph Traversal Agent**: Use AI-powered graph traversal that intelligently navigates the Neo4j dependency graph:
    - **Smart Path Discovery**: Agent determines optimal traversal paths based on vulnerability type and impact severity
    - **Context-Aware Analysis**: AI understands code context to assess real-world impact vs theoretical risk
    - **Dynamic Traversal Strategy**: Agent adapts traversal patterns based on repository structure and dependency complexity
  
  **Remediation Recommendations:**
  - Use Neo4j queries to analyze dependency relationships and suggest optimal fixes
  - Generate intelligent recommendations for dependency version updates with compatibility analysis
  - Identify and suggest alternative packages when updates aren't available
  - Provide detailed remediation suggestions including:
    - Step-by-step fix instructions
    - Potential impact and risks of each suggested change
    - Multiple remediation options ranked by safety and effectiveness
    - Rollback procedures for each suggested fix
  - Generate test recommendations to verify remediation safety before human implementation
  - Generate comprehensive reports showing:
    - Vulnerability severity and details with AI-generated explanations
    - Complete list of affected files ranked by impact
    - Dependency chains that lead to the vulnerability
    - AI-powered risk assessment and remediation urgency recommendations
    - Before/after analysis of recommended changes


### System Testing --- file to submit is test_vulnerability_system.py

- With Python, create a script that creates system tests for the vulnerability analysis and remediation system:
  - You should have at least five tests that guarantee the system returns valid vulnerability analysis results


### Agentic System Architecture

The vulnerability system combines intelligent analysis with human-guided remediation:

- **Agentic Graph Analysis**: The system uses AI agents for intelligent graph traversal and impact assessment
- **Neo4j Integration**: Agents use the dependency graph to make informed decisions about vulnerability impact
- **Integrated Workflow**: Agentic analysis directly feeds into intelligent remediation recommendations
- **Human-Centered Remediation**: System provides detailed suggestions and guidance for humans to implement fixes safely
- **System Testing**: Dedicated system tests ensure reliability of the complete vulnerability analysis and remediation workflow
- **Explainable AI**: All agent decisions and recommendations include detailed reasoning that can be audited and understood by security teams

### Submission

Zip up the three files:
- dependency_scanner.py
- vulnerability_system.py
- test_vulnerability_system.py

And then upload to DataExpert.io
