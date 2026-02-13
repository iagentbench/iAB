"""
Graph visualization functions.
"""
import re
import json
import math
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def convert_entities_to_dicts(df) -> List[Dict]:
    """Convert entities dataframe to list of dicts for visualization."""
    nodes_dict = {}
    for _, row in df.iterrows():
        node_id = row["title"]
        if isinstance(node_id, str):
            node_id = node_id.strip('"')
        if node_id not in nodes_dict:
            # Convert row to dict and handle any non-serializable types
            props = {}
            for key, value in row.to_dict().items():
                # Handle NaN values first (must be before other conversions)
                if isinstance(value, float) and math.isnan(value):
                    props[key] = None
                    continue
                # Convert pandas types to Python native types
                if hasattr(value, 'item'):  # numpy/pandas scalar or array
                    try:
                        # Check if it's a scalar (size 1) before calling item()
                        if hasattr(value, 'size') and value.size == 1:
                            val = value.item()
                            # Check for NaN after conversion
                            if isinstance(val, float) and math.isnan(val):
                                props[key] = None
                            else:
                                props[key] = val
                        elif hasattr(value, 'shape') and value.shape == ():
                            # 0-dimensional array (scalar)
                            val = value.item()
                            if isinstance(val, float) and math.isnan(val):
                                props[key] = None
                            else:
                                props[key] = val
                        else:
                            # Array with multiple elements - convert to list and handle NaNs
                            arr = value.tolist() if hasattr(value, 'tolist') else list(value)
                            # Replace NaN in list
                            props[key] = [None if isinstance(x, float) and math.isnan(x) else x for x in arr]
                    except (ValueError, AttributeError):
                        # Fallback: try tolist or convert to string
                        try:
                            arr = value.tolist() if hasattr(value, 'tolist') else str(value)
                            if isinstance(arr, list):
                                props[key] = [None if isinstance(x, float) and math.isnan(x) else x for x in arr]
                            else:
                                props[key] = arr
                        except:
                            props[key] = str(value)
                elif hasattr(value, 'tolist'):  # numpy array
                    arr = value.tolist()
                    props[key] = [None if isinstance(x, float) and math.isnan(x) else x for x in arr]
                else:
                    # Final check for NaN in regular values
                    if isinstance(value, float) and math.isnan(value):
                        props[key] = None
                    else:
                        props[key] = value
            nodes_dict[node_id] = {"id": node_id, "properties": props}
    return list(nodes_dict.values())


def convert_relationships_to_dicts(df) -> List[Dict]:
    """Convert relationships dataframe to list of dicts with short labels."""
    relationships = []
    for _, row in df.iterrows():
        source = row["source"]
        target = row["target"]
        if isinstance(source, str):
            source = source.strip('"')
        if isinstance(target, str):
            target = target.strip('"')
        
        # Create short label from description (skip first words, get 2-3 meaningful words) - no LLM calls
        description = row.get("description", "")
        if description:
            desc_str = str(description)
            words = desc_str.split()
            
            # Skip first 2-3 words (often contain node names like "Joe Keery", "Steve Harrington")
            # Then take 2-3 meaningful words
            skip_words = min(3, len(words) // 2)  # Skip up to 3 words or half the sentence
            remaining_words = words[skip_words:]
            
            # Take 2-3 words, avoiding very short words
            meaningful_words = []
            for word in remaining_words[:5]:  # Look at next 5 words
                # Remove punctuation and check if meaningful (not just "is", "a", "the", etc.)
                clean_word = re.sub(r'[^a-zA-Z0-9]', '', word.lower())
                if len(clean_word) > 2 and clean_word not in ['the', 'and', 'for', 'are', 'was', 'has', 'had', 'his', 'her', 'its', 'our', 'their']:
                    meaningful_words.append(word)
                    if len(meaningful_words) >= 3:
                        break
            
            # If we didn't find enough, just take next 2-3 words
            if len(meaningful_words) < 2:
                meaningful_words = remaining_words[:3]
            
            short_label = " ".join(meaningful_words[:3])
            if len(desc_str) > len(short_label) + 20:  # Only add ... if significantly longer
                short_label += "..."
        else:
            short_label = ""
        
        # Convert row to dict and handle any non-serializable types
        rel_dict = {}
        for key, value in row.to_dict().items():
            # Handle NaN values first (must be before other conversions)
            if isinstance(value, float) and math.isnan(value):
                rel_dict[key] = None
                continue
            # Convert pandas types to Python native types
            if hasattr(value, 'item'):  # numpy/pandas scalar or array
                try:
                    # Check if it's a scalar (size 1) before calling item()
                    if hasattr(value, 'size') and value.size == 1:
                        val = value.item()
                        # Check for NaN after conversion
                        if isinstance(val, float) and math.isnan(val):
                            rel_dict[key] = None
                        else:
                            rel_dict[key] = val
                    elif hasattr(value, 'shape') and value.shape == ():
                        # 0-dimensional array (scalar)
                        val = value.item()
                        if isinstance(val, float) and math.isnan(val):
                            rel_dict[key] = None
                        else:
                            rel_dict[key] = val
                    else:
                        # Array with multiple elements - convert to list and handle NaNs
                        arr = value.tolist() if hasattr(value, 'tolist') else list(value)
                        rel_dict[key] = [None if isinstance(x, float) and math.isnan(x) else x for x in arr]
                except (ValueError, AttributeError):
                    # Fallback: try tolist or convert to string
                    try:
                        arr = value.tolist() if hasattr(value, 'tolist') else str(value)
                        if isinstance(arr, list):
                            rel_dict[key] = [None if isinstance(x, float) and math.isnan(x) else x for x in arr]
                        else:
                            rel_dict[key] = arr
                    except:
                        rel_dict[key] = str(value)
            elif hasattr(value, 'tolist'):  # numpy array
                arr = value.tolist()
                rel_dict[key] = [None if isinstance(x, float) and math.isnan(x) else x for x in arr]
            else:
                # Final check for NaN in regular values
                if isinstance(value, float) and math.isnan(value):
                    rel_dict[key] = None
                else:
                    rel_dict[key] = value
        
        rel_dict["short_label"] = short_label
        rel_dict["full_description"] = str(description) if description else ""
        
        relationships.append({"start": source, "end": target, "properties": rel_dict})
    return relationships


def create_graph_visualization(graph_data: Dict, output_path: Path) -> Path:
    """
    Create and save interactive graph visualization.
    Exports both GraphML (for yEd/yFiles) and HTML formats.
    
    Args:
        graph_data: Dictionary with 'entities' and 'relationships' DataFrames
        output_path: Path where HTML file should be saved (GraphML will be saved alongside)
    
    Returns:
        Path to saved HTML file
    """
    if 'entities' not in graph_data or 'relationships' not in graph_data:
        raise ValueError("graph_data must contain 'entities' and 'relationships'")
    
    entities_df = graph_data['entities']
    relationships_df = graph_data['relationships']
    
    # Merge community data into entities if available
    if 'communities' in graph_data:
        communities_df = graph_data['communities']
        entity_to_community = {}
        for _, comm_row in communities_df.iterrows():
            if 'entity_ids' in comm_row:
                entity_ids = comm_row['entity_ids']
                if isinstance(entity_ids, list) and len(entity_ids) > 0:
                    community_id = comm_row.get('community', comm_row.get('id'))
                    for eid in entity_ids:
                        entity_to_community[eid] = community_id
        
        if entity_to_community:
            entities_df = entities_df.copy()
            entities_df['community'] = entities_df['id'].map(entity_to_community)
    
    nodes = convert_entities_to_dicts(entities_df)
    edges = convert_relationships_to_dicts(relationships_df)
    
    # Export GraphML format (can be opened in yEd, Gephi, etc.)
    graphml_path = output_path.parent / (output_path.stem + ".graphml")
    _export_graphml(nodes, edges, graphml_path)
    
    # Export HTML with yFiles-compatible format
    html_content = _generate_yfiles_html(nodes, edges, entities_df, output_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


def _export_graphml(nodes: List[Dict], edges: List[Dict], output_path: Path) -> None:
    """Export graph to GraphML format (can be opened in yEd, Gephi, etc.)."""
    import xml.etree.ElementTree as ET
    
    # Create GraphML root
    graphml = ET.Element('graphml', {
        'xmlns': 'http://graphml.graphdrawing.org/xmlns',
        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'xsi:schemaLocation': 'http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd'
    })
    
    # Define attributes
    key_id = ET.SubElement(graphml, 'key', {'id': 'd0', 'for': 'node', 'attr.name': 'label', 'attr.type': 'string'})
    key_type = ET.SubElement(graphml, 'key', {'id': 'd1', 'for': 'node', 'attr.name': 'type', 'attr.type': 'string'})
    key_community = ET.SubElement(graphml, 'key', {'id': 'd2', 'for': 'node', 'attr.name': 'community', 'attr.type': 'string'})
    key_weight = ET.SubElement(graphml, 'key', {'id': 'd3', 'for': 'edge', 'attr.name': 'weight', 'attr.type': 'double'})
    key_label = ET.SubElement(graphml, 'key', {'id': 'd4', 'for': 'edge', 'attr.name': 'label', 'attr.type': 'string'})
    
    # Create graph element
    graph = ET.SubElement(graphml, 'graph', {'id': 'G', 'edgedefault': 'directed'})
    
    # Add nodes
    node_map = {}
    for i, node in enumerate(nodes):
        node_id = node['id']
        node_elem = ET.SubElement(graph, 'node', {'id': str(i)})
        node_map[node_id] = str(i)
        
        data_label = ET.SubElement(node_elem, 'data', {'key': 'd0'})
        data_label.text = node_id
        
        props = node.get('properties', {})
        if props.get('type'):
            data_type = ET.SubElement(node_elem, 'data', {'key': 'd1'})
            data_type.text = str(props['type'])
        if props.get('community') is not None:
            data_comm = ET.SubElement(node_elem, 'data', {'key': 'd2'})
            data_comm.text = str(props['community'])
    
    # Add edges
    for edge in edges:
        source_id = edge['start']
        target_id = edge['end']
        if source_id in node_map and target_id in node_map:
            edge_elem = ET.SubElement(graph, 'edge', {
                'source': node_map[source_id],
                'target': node_map[target_id]
            })
            
            props = edge.get('properties', {})
            if props.get('weight'):
                data_weight = ET.SubElement(edge_elem, 'data', {'key': 'd3'})
                data_weight.text = str(props['weight'])
            if props.get('short_label'):
                data_label = ET.SubElement(edge_elem, 'data', {'key': 'd4'})
                data_label.text = str(props['short_label'])
    
    # Write to file
    tree = ET.ElementTree(graphml)
    ET.indent(tree, space='  ')
    tree.write(output_path, encoding='utf-8', xml_declaration=True)


def _generate_yfiles_html(nodes: List[Dict], edges: List[Dict], entities_df, output_path: Path) -> str:
    """Generate HTML with yFiles library embedded (standalone, works offline)."""
    nodes_json = json.dumps(nodes, indent=2, default=str)
    edges_json = json.dumps(edges, indent=2, default=str)
    
    # Escape JSON for embedding
    nodes_json_escaped = nodes_json.replace('</script>', '<\\/script>')
    edges_json_escaped = edges_json.replace('</script>', '<\\/script>')
    
    graphml_filename = output_path.stem + ".graphml"
    
    # Use yFiles library from CDN (or can be bundled)
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>iAgentBench Knowledge Graph (yFiles)</title>
    <script src="https://cdn.jsdelivr.net/npm/yfiles@2.6.0/dist/yfiles.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: #f5f5f5;
        }}
        #graphContainer {{
            width: 100%;
            height: 100vh;
            background: white;
        }}
        .info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255, 255, 255, 0.9);
            padding: 10px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 1000;
        }}
    </style>
</head>
<body>
    <div class="info">Nodes: {num_nodes} | Edges: {num_edges} | <a href="{graphml_file}" download>Download GraphML (for yEd/yFiles)</a></div>
    <div id="graphContainer"></div>
    
    <script>
        // Graph data
        const nodes = {nodes_data};
        const edges = {edges_data};
        
        // Initialize yFiles graph
        const graphComponent = new yfiles.view.GraphComponent('graphContainer');
        const graph = graphComponent.graph;
        
        // Create node map
        const nodeMap = {{}};
        nodes.forEach(node => {{
            const yNode = graph.createNode({{
                layout: new yfiles.geometry.Rect(0, 0, 100, 50),
                tag: node
            }});
            nodeMap[node.id] = yNode;
            
            // Set label
            const label = graph.addLabel(yNode, node.id);
            graph.setLabelStyle(label, new yfiles.styles.DefaultLabelStyle({{
                font: new yfiles.styles.Font('Arial', 12)
            }}));
        }});
        
        // Create edges
        edges.forEach(edge => {{
            const source = nodeMap[edge.start];
            const target = nodeMap[edge.end];
            if (source && target) {{
                const yEdge = graph.createEdge(source, target);
                if (edge.properties && edge.properties.short_label) {{
                    const label = graph.addLabel(yEdge, edge.properties.short_label);
                }}
            }}
        }});
        
        // Apply layout
        const layout = new yfiles.layout.hierarchic.HierarchicLayout();
        graphComponent.morphLayout(layout);
        
        // Fit graph to view
        graphComponent.fitGraphBounds();
    </script>
</body>
</html>"""
    
    graphml_file = output_path.stem + ".graphml"
    return html_template.format(
        num_nodes=len(nodes),
        num_edges=len(edges),
        nodes_data=nodes_json_escaped,
        edges_data=edges_json_escaped,
        graphml_file=graphml_filename
    )


def _generate_html_graph(nodes: List[Dict], edges: List[Dict], entities_df) -> str:
    """Generate standalone HTML file with graph visualization."""
    # Convert nodes and edges to JSON for embedding
    # Use default=str to handle any non-serializable objects (like pandas Series)
    nodes_json = json.dumps(nodes, indent=2, default=str)
    edges_json = json.dumps(edges, indent=2, default=str)
    
    # Escape JSON for embedding in HTML/JS (prevent </script> from breaking HTML)
    nodes_json_escaped = nodes_json.replace('</script>', '<\\/script>')
    edges_json_escaped = edges_json.replace('</script>', '<\\/script>')
    
    # Use .format() instead of f-string to avoid Python parsing JavaScript code
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>iAgentBench Knowledge Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background: #f5f5f5;
        }}
        #graph-container {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .node {{
            cursor: pointer;
        }}
        .node circle {{
            stroke: #333;
            stroke-width: 2px;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
            stroke-width: 2px;
        }}
        .link-label {{
            font-size: 10px;
            fill: #666;
        }}
        .node-label {{
            font-size: 12px;
            font-weight: bold;
            fill: #333;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
        }}
        h1 {{
            margin-top: 0;
            color: #333;
        }}
        .info {{
            margin-bottom: 20px;
            color: #666;
        }}
    </style>
</head>
<body>
    <h1>iAgentBench Knowledge Graph</h1>
    <div class="info">Nodes: {num_nodes} | Edges: {num_edges}</div>
    <div id="graph-container"></div>
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        const nodes = {nodes_data};
        const edges = {edges_data};
        
        const width = document.getElementById('graph-container').clientWidth - 40;
        const height = Math.max(600, window.innerHeight - 200);
        
        const svg = d3.select('#graph-container')
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        const tooltip = d3.select('#tooltip');
        
        // Create force simulation
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(edges).id(function(d) {{ return d.id; }}).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(30));
        
        // Add links
        const link = svg.append('g')
            .selectAll('line')
            .data(edges)
            .enter().append('line')
            .attr('class', 'link')
            .attr('stroke-width', function(d) {{
                var weight = d.properties && d.properties.weight ? d.properties.weight : 1;
                return Math.sqrt(weight) * 2;
            }});
        
        // Add link labels
        const linkLabels = svg.append('g')
            .selectAll('text')
            .data(edges)
            .enter().append('text')
            .attr('class', 'link-label')
            .text(function(d) {{
                return (d.properties && d.properties.short_label) ? d.properties.short_label : '';
            }})
            .attr('dx', 5)
            .attr('dy', -5);
        
        // Add nodes
        const node = svg.append('g')
            .selectAll('circle')
            .data(nodes)
            .enter().append('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));
        
        node.append('circle')
            .attr('r', function(d) {{
                var degree = (d.properties && d.properties.degree) ? d.properties.degree : 1;
                return 5 + Math.sqrt(degree) * 3;
            }})
            .attr('fill', function(d) {{
                var community = (d.properties && d.properties.community !== null && d.properties.community !== undefined) ? d.properties.community : null;
                var colors = ['crimson', 'darkorange', 'indigo', 'cornflowerblue', 'cyan', 'teal', 'green'];
                if (community !== null && community !== undefined && !isNaN(community)) {{
                    return colors[Math.floor(community) % colors.length];
                }}
                return 'lightgray';
            }});
        
        node.append('text')
            .attr('class', 'node-label')
            .text(function(d) {{ return d.id; }})
            .attr('dx', 10)
            .attr('dy', 5);
        
        // Tooltip on hover
        node.on('mouseover', function(event, d) {{
            tooltip.transition().duration(200).style('opacity', 1);
            var nodeType = (d.properties && d.properties.type) ? d.properties.type : 'N/A';
            var nodeCommunity = (d.properties && d.properties.community !== null && d.properties.community !== undefined) ? d.properties.community : 'N/A';
            tooltip.html('<strong>' + d.id + '</strong><br/>Type: ' + nodeType + '<br/>Community: ' + nodeCommunity)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px');
        }})
        .on('mouseout', function() {{
            tooltip.transition().duration(200).style('opacity', 0);
        }});
        
        // Update positions on simulation tick
        simulation.on('tick', function() {{
            link
                .attr('x1', function(d) {{ return d.source.x; }})
                .attr('y1', function(d) {{ return d.source.y; }})
                .attr('x2', function(d) {{ return d.target.x; }})
                .attr('y2', function(d) {{ return d.target.y; }});
            
            linkLabels
                .attr('x', function(d) {{ return (d.source.x + d.target.x) / 2; }})
                .attr('y', function(d) {{ return (d.source.y + d.target.y) / 2; }});
            
            node.attr('transform', function(d) {{ return 'translate(' + d.x + ',' + d.y + ')'; }});
        }});
        
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>"""
    
    # Format the template with actual values (using .format() to avoid f-string parsing issues)
    html_content = html_template.format(
        num_nodes=len(nodes),
        num_edges=len(edges),
        nodes_data=nodes_json_escaped,
        edges_data=edges_json_escaped
    )
    
    return html_content
