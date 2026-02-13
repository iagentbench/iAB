"""
Generate Jupyter notebook for viewing graph and data.
"""
import json
from pathlib import Path
from typing import Dict


def generate_viewer_notebook(run_dir: Path, graph_data: Dict) -> Path:
    """
    Generate a concise Jupyter notebook that displays the graph first, then summary info.
    
    Args:
        run_dir: Directory containing the outputs
        graph_data: Dictionary with graph data
    
    Returns:
        Path to generated notebook
    """
    notebook_path = run_dir / "view_graph.ipynb"
    
    # Check if community_details.json exists
    community_json_path = run_dir / "community_details.json"
    has_community_details = community_json_path.exists()
    
    # Get project root - detect structure dynamically
    # Structure can be: output/{keyword}/{timestamp} or output/{batch_name}/{query}/{timestamp}
    # Go up until we find a directory that contains "src" folder
    current = run_dir.parent
    project_root = None
    max_levels = 5  # Safety limit
    for _ in range(max_levels):
        if (current / "src").exists() and (current / "cli").exists():
            project_root = current
            break
        current = current.parent
        if current == current.parent:  # Reached filesystem root
            break
    
    # Fallback: if not found, use old logic (assumes output/keyword/timestamp)
    if project_root is None:
        project_root = run_dir.parent.parent.parent
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# iAgentBench Knowledge Graph Viewer\n",
                    "\n",
                    "Interactive visualization of the generated knowledge graph."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Setup and imports\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "import json\n",
                    "from IPython.display import display\n",
                    "\n",
                    f"# Set paths\n",
                    f"run_dir = Path({repr(str(run_dir))})\n",
                    f"# Find project root by looking for src/ directory\n",
                    f"current = run_dir.parent\n",
                    f"project_root = None\n",
                    f"for _ in range(5):\n",
                    f"    if (current / \"src\").exists() and (current / \"cli\").exists():\n",
                    f"        project_root = current\n",
                    f"        break\n",
                    f"    if current == current.parent:\n",
                    f"        break\n",
                    f"    current = current.parent\n",
                    f"if project_root is None:\n",
                    f"    # Fallback: assume standard structure output/keyword/timestamp\n",
                    f"    project_root = run_dir.parent.parent.parent\n",
                    f"sys.path.insert(0, str(project_root))\n",
                    f"\n",
                    f"# Load graph data\n",
                    f"entities_df = pd.read_parquet(run_dir / \"graphs\" / \"entities.parquet\")\n",
                    f"relationships_df = pd.read_parquet(run_dir / \"graphs\" / \"relationships.parquet\")\n",
                    f"\n",
                    f"print(f\"📊 Loaded {{len(entities_df)}} entities, {{len(relationships_df)}} relationships\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install and setup visualization\n",
                    "try:\n",
                    "    from yfiles_jupyter_graphs import GraphWidget\n",
                    "except ImportError:\n",
                    "    import subprocess\n",
                    "    subprocess.run([\"pip\", \"install\", \"--quiet\", \"yfiles-jupyter-graphs\"], check=True)\n",
                    "    from yfiles_jupyter_graphs import GraphWidget\n",
                    "\n",
                    "from src.visualization import convert_entities_to_dicts, convert_relationships_to_dicts\n",
                    "\n",
                    "# Merge community data if available\n",
                    "communities_parquet = run_dir / \"graphs\" / \"communities.parquet\"\n",
                    "if communities_parquet.exists():\n",
                    "    communities_df = pd.read_parquet(communities_parquet)\n",
                    "    entity_to_community = {}\n",
                    "    for _, comm_row in communities_df.iterrows():\n",
                    "        if 'entity_ids' in comm_row:\n",
                    "            entity_ids = comm_row['entity_ids']\n",
                    "            if isinstance(entity_ids, list) and len(entity_ids) > 0:\n",
                    "                community_id = comm_row.get('community', comm_row.get('id'))\n",
                    "                for eid in entity_ids:\n",
                    "                    entity_to_community[eid] = community_id\n",
                    "    \n",
                    "    if entity_to_community:\n",
                    "        entities_df['community'] = entities_df['id'].map(entity_to_community)\n",
                    "\n",
                    "# Convert to visualization format\n",
                    "nodes = convert_entities_to_dicts(entities_df)\n",
                    "edges = convert_relationships_to_dicts(relationships_df)\n",
                    "\n",
                    "# Create and configure graph widget\n",
                    "w = GraphWidget()\n",
                    "w.directed = True\n",
                    "w.nodes = nodes\n",
                    "w.edges = edges\n",
                    "w.node_label_mapping = \"title\"\n",
                    "w.edge_label_mapping = \"short_label\"\n",
                    "\n",
                    "# Color mapping\n",
                    "def community_to_color(community):\n",
                    "    colors = [\"crimson\", \"darkorange\", \"indigo\", \"cornflowerblue\", \"cyan\", \"teal\", \"green\"]\n",
                    "    return colors[int(community) % len(colors)] if community is not None and not pd.isna(community) else \"lightgray\"\n",
                    "\n",
                    "if 'community' in entities_df.columns:\n",
                    "    w.node_color_mapping = lambda node: community_to_color(node[\"properties\"].get(\"community\"))\n",
                    "elif 'type' in entities_df.columns:\n",
                    "    def type_to_color(entity_type):\n",
                    "        color_map = {\"PERSON\": \"crimson\", \"ORGANIZATION\": \"cornflowerblue\", \"GEO\": \"teal\", \"EVENT\": \"darkorange\"}\n",
                    "        return color_map.get(str(entity_type).upper(), \"lightgray\")\n",
                    "    w.node_color_mapping = lambda node: type_to_color(node[\"properties\"].get(\"type\"))\n",
                    "\n",
                    "# Node size by degree, edge thickness by weight\n",
                    "if 'degree' in entities_df.columns:\n",
                    "    w.node_scale_factor_mapping = lambda node: 0.5 + (node[\"properties\"].get(\"degree\", 1) * 1.5 / 20)\n",
                    "if 'weight' in relationships_df.columns:\n",
                    "    w.edge_thickness_factor_mapping = \"weight\"\n",
                    "\n",
                    "w.circular_layout()\n",
                    "\n",
                    "# Display graph\n",
                    "display(w)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Summary Information"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Graph Summary\n",
                    "print(\"📊 GRAPH SUMMARY\")\n",
                    "print(\"=\" * 80)\n",
                    "print(f\"Entities: {len(entities_df)} | Relationships: {len(relationships_df)}\")\n",
                    "\n",
                    "if 'type' in entities_df.columns:\n",
                    "    type_counts = entities_df['type'].value_counts()\n",
                    "    print(f\"\\nEntity Types:\")\n",
                    "    for etype, count in type_counts.items():\n",
                    "        print(f\"  • {etype or '(No Type)':20s}: {count}\")\n",
                    "\n",
                    "if 'weight' in relationships_df.columns:\n",
                    "    print(f\"\\nRelationship Weights: {relationships_df['weight'].min():.1f} - {relationships_df['weight'].max():.1f} (avg: {relationships_df['weight'].mean():.1f})\")\n",
                    "\n",
                    "# Community Details\n",
                    "comm_json = run_dir / \"community_details.json\"\n",
                    "if comm_json.exists():\n",
                    "    with open(comm_json, 'r') as f:\n",
                    "        comm_data = json.load(f)\n",
                    "    \n",
                    "    # Handle dict with 'communities' key or direct list\n",
                    "    communities = comm_data.get('communities', []) if isinstance(comm_data, dict) else (comm_data if isinstance(comm_data, list) else [])\n",
                    "    \n",
                    "    if communities:\n",
                    "        print(f\"\\n🏘️  COMMUNITIES ({len(communities)})\")\n",
                    "        print(\"=\" * 80)\n",
                    "        for comm in communities:\n",
                    "            if not isinstance(comm, dict):\n",
                    "                continue\n",
                    "            title = comm.get('title', comm.get('name', 'N/A'))\n",
                    "            comm_id = comm.get('community_id', comm.get('id', comm.get('community', 'N/A')))\n",
                    "            rank = comm.get('rank', comm.get('rating', None))\n",
                    "            summary = comm.get('summary', '')\n",
                    "            \n",
                    "            print(f\"\\n{title}\")\n",
                    "            if rank is not None:\n",
                    "                print(f\"  Rank: {rank}/10\")\n",
                    "            if summary:\n",
                    "                words = str(summary).split()\n",
                    "                summary_preview = ' '.join(words[:25]) + ('...' if len(words) > 25 else '')\n",
                    "                print(f\"  Summary: {summary_preview}\")\n",
                    "\n",
                    "# Operational Metrics\n",
                    "metrics_file = run_dir / \"operational_metrics.txt\"\n",
                    "if metrics_file.exists():\n",
                    "    print(f\"\\n📈 OPERATIONAL METRICS\")\n",
                    "    print(\"=\" * 80)\n",
                    "    with open(metrics_file, 'r') as f:\n",
                    "        metrics_content = f.read()\n",
                    "        # Show key metrics (skip header lines)\n",
                    "        lines = metrics_content.split('\\n')\n",
                    "        for line in lines:\n",
                    "            if any(keyword in line for keyword in ['Chat:', 'Embedding:', 'Total:', 'Input:', 'Output:', 'tokens']):\n",
                    "                print(f\"  {line.strip()}\")\n",
                    "\n",
                    "# File Links\n",
                    "print(f\"\\n📁 FILES\")\n",
                    "print(\"=\" * 80)\n",
                    "files = [\n",
                    "    (\"Community Details\", run_dir / \"community_details.json\"),\n",
                    "    (\"Operational Metrics\", run_dir / \"operational_metrics.txt\"),\n",
                    "    (\"Comprehensive Log\", list((run_dir / \"logs\").glob(\"pipeline_*.log\"))[0] if list((run_dir / \"logs\").glob(\"pipeline_*.log\")) else None),\n",
                    "    (\"API Calls Log\", run_dir / \"logs\" / \"bedrock_api_calls_consolidated.jsonl\")\n",
                    "]\n",
                    "for name, path in files:\n",
                    "    if path and path.exists():\n",
                    "        print(f\"  ✓ {name}: {path.name}\")\n",
                    "    else:\n",
                    "        print(f\"  ⚠️  {name}: Not found\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_content, f, indent=1, ensure_ascii=False)
    
    return notebook_path
