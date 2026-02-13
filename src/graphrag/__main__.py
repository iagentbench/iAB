# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The GraphRAG package."""

# CLI main removed - we use our own CLI in cli/main.py
# This file is not used in the indexing workflow
try:
    from graphrag.cli.main import app
except ImportError:
    # If typer not available, create minimal stub
    class MinimalApp:
        def __call__(self, *args, **kwargs):
            print("GraphRAG CLI not available. Use cli/main.py instead.")
    app = MinimalApp()

app(prog_name="graphrag")
