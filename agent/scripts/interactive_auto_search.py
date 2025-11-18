#!/usr/bin/env python3
"""
Interactive CLI chat script using the auto_search workflow.

This script provides an interactive interface to the AutoReasonSearchWorkflow,
allowing you to chat with the model using the same sophisticated search and answer
agents used in the auto_search workflow.

Example usage:
    python scripts/interactive_auto_search.py --config workflows/trained/auto_search_sft.yaml
    python scripts/interactive_auto_search.py --config workflows/trained/auto_search_sft.yaml --dataset-name sqav2
"""

import asyncio
import sys
import time
import random
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console, Group
    from rich.markdown import Markdown
    from rich.text import Text
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.live import Live
    from rich.spinner import Spinner
    HAS_RICH = True
except ImportError:
    print("Warning: rich/typer not available. Install with: pip install typer rich")
    HAS_RICH = False
    # Fallback for basic functionality
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
        def status(self, *args, **kwargs):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    class Prompt:
        @staticmethod
        def ask(prompt_text):
            return input(prompt_text + " ")
    
    class Panel:
        def __init__(self, content, **kwargs):
            self.content = content
    
    class Markdown:
        def __init__(self, text):
            self.text = text

# Import the workflow
sys.path.insert(0, str(Path(__file__).parent.parent))
from workflows.auto_search_sft import AutoReasonSearchWorkflow

if HAS_RICH:
    app = typer.Typer(help="Interactive chat using auto_search workflow")
    console = Console()
else:
    # Fallback app structure
    class App:
        def command(self, **kwargs):
            def decorator(f):
                return f
            return decorator
    app = App()
    console = Console()


async def chat_loop(
    workflow: AutoReasonSearchWorkflow,
    dataset_name: Optional[str] = None,
    verbose: bool = False,
    show_full_tool_output: bool = False,
):
    """Run the interactive chat loop using the auto_search workflow."""
    console.print("\n[bold green]Starting interactive auto_search chat session[/bold green]")
    console.print("\n[dim]Type 'exit' or 'quit' to end the session[/dim]\n")
    
    if dataset_name:
        console.print(f"[dim]Using dataset configuration: {dataset_name}[/dim]\n")

    import re

    # Loading text variations
    LOADING_TEXTS = [
        "Infodigging", "Factgrabbing", "Knowledging", "Studifying", 
        "Learninating", "Wisdomizing", "Thoughtsifting", "Bookworming", 
        "Researching", "Cognitating", "Discoverifying", "Deepthinking", 
        "Sciencifying", "Scholarizing", "Ideahunting", "Thinkworking", 
        "Smartifying", "Conclusionizing", "Insightfarming", "Infohoovering", 
        "Factstacking", "Databinging", "Factsniffing", "Mindcooking", 
        "Factweaving", "Infopiling"
    ]

    # State for segmented live display
    last_processed_text_len = 0
    current_segment_text = ""
    active_live = None
    is_answering = False

    # Helper to reduce newlines
    def clean_text(t):
        return re.sub(r'\n{3,}', '\n\n', t)
    
    # Helper to format citations and think tags with Rich markup
    def format_citations(t):
        # Format citation tags
        def format_cite(match):
            cite_id = match.group(2)
            cite_content = match.group(3)
            return f'[dim]<cite id="[/dim][dim cyan]{cite_id}[/dim cyan][dim]">[/dim][cyan bold]{cite_content}[/cyan bold][dim]</cite>[/dim]'
        t = re.sub(r'<cite\s+id=(["\']?)([^"\'>\s]+)\1[^>]*>([^<]+)</cite>', format_cite, t)
        
        # Format think tags (dimmed)
        t = t.replace("<think>", "[dim]<think>[/dim]")
        t = t.replace("</think>", "[dim]</think>[/dim]")
        
        return t
        
    # Generic helper to render a panel
    def render_panel(content, title, style, is_active=True, spinner_text="Thinking..."):
        renderables = []
        
        # Clean and strip content
        content = clean_text(content).strip()
        
        if content:
            formatted = format_citations(content)
            renderable = Text.from_markup(formatted)
            
            panel = Panel(
                renderable,
                title=f"[{style}]{title}[/{style}]",
                title_align="left",
                border_style=style
            )
            renderables.append(panel)
        
        if not content and title == "Thinking":
            # Rotate loading text every 10 seconds randomly
            # Seed random generator with time bucket to ensure consistency across renders within the same bucket
            # but variety over time
            bucket = int(time.time() / 10)
            rng = random.Random(bucket)
            spinner_text = f"{rng.choice(LOADING_TEXTS)}..."

        # Add spinner at the bottom if active
        if is_active:
            if renderables:
                renderables.append(Text(" ")) 
            renderables.append(Spinner("dots", text=spinner_text, style="cyan"))

        return Group(*renderables)

    # Define callback to print step updates
    def print_step_update(text, tool_calls):
        nonlocal last_processed_text_len, current_segment_text, active_live, is_answering
        
        # Detect if text stream has reset (new generation started, e.g. next agent)
        if text and len(text) < last_processed_text_len:
            last_processed_text_len = 0
            # If we have an active panel, finalize it as it belongs to previous generation
            if active_live:
                # If we were answering, finalize as answer. If thinking, finalize as thinking.
                if is_answering:
                    active_live.update(render_panel(current_segment_text, "Answer", "green", is_active=False))
                else:
                    active_live.update(render_panel(current_segment_text, "Thinking", "yellow", is_active=False))
                active_live.stop()
                active_live = None
            current_segment_text = ""
            is_answering = False
            
            # Start new live display for new generation (default to Thinking)
            active_live = Live(
                render_panel("", "Thinking", "yellow", is_active=True),
                console=console,
                auto_refresh=True,
                refresh_per_second=10,
                vertical_overflow="visible"
            )
            active_live.start()
        
        # Handle text updates
        if text and len(text) > last_processed_text_len:
            new_chunk = text[last_processed_text_len:]
            # Check if this chunk is just a newline after a tool call, skip if so to avoid empty boxes
            if not current_segment_text and not new_chunk.strip():
                last_processed_text_len = len(text)
            else:
                current_segment_text += new_chunk
                last_processed_text_len = len(text)
                
                # Check for answer tag split
                answer_match = re.search(r'(?s)(.*)<answer>(.*)', current_segment_text)
                
                if answer_match:
                    thinking_text = answer_match.group(1)
                    answer_text = answer_match.group(2).replace("</answer>", "")
                    
                    if not is_answering:
                        # Transition from Thinking to Answering detected!
                        if active_live:
                            # Finalize thinking panel
                            active_live.update(render_panel(thinking_text, "Thinking", "yellow", is_active=False))
                            active_live.stop()
                        
                        # Start new live display for Answer
                        is_answering = True
                        
                        current_segment_text = answer_text
                        
                        active_live = Live(
                            render_panel(current_segment_text, "Answer", "green", is_active=True, spinner_text="Generating Answer..."),
                            console=console,
                            auto_refresh=True,
                            refresh_per_second=10,
                            vertical_overflow="visible"
                        )
                        active_live.start()
                    else:
                        # Already answering, just update content
                        # Be careful not to update if we already have a cleaner version?
                        # No, current_segment_text accumulates, so we must re-extract answer part
                        # to avoid duplicating "Answer" content if `answer_match` found it again in accumulated string
                        current_segment_text = answer_text # Update with latest answer text
                        active_live.update(render_panel(current_segment_text, "Answer", "green", is_active=True, spinner_text="Generating Answer..."))
                
                else:
                    # No answer tag yet
                    if not active_live:
                         active_live = Live(
                            render_panel(current_segment_text, "Thinking", "yellow", is_active=True),
                            console=console,
                            auto_refresh=True,
                            refresh_per_second=10,
                            vertical_overflow="visible"
                        )
                         active_live.start()
                    else:
                        active_live.update(render_panel(current_segment_text, "Thinking", "yellow", is_active=True))
        
        # Handle tool calls
        if tool_calls:
            # Finalize current thinking block
            if active_live:
                if is_answering:
                     active_live.update(render_panel(current_segment_text, "Answer", "green", is_active=False))
                else:
                     active_live.update(render_panel(current_segment_text, "Thinking", "yellow", is_active=False))
                active_live.stop()
                active_live = None
            
            # Print tool calls
            for tool_call in tool_calls:
                tool_name = tool_call.tool_name
                console.print(f"\n[bold magenta]Tool Call: {tool_name}[/bold magenta]")
                
                output = tool_call.output
                if not show_full_tool_output and len(output) > 500:
                    output = output[:500] + "... [truncated]"
                
                output = clean_text(output)
                console.print(Panel(output, title="[green]Output[/green]", border_style="green"))
                
            # Reset segment for next block (next iteration)
            current_segment_text = ""
            last_processed_text_len = 0 # Reset text tracking for new iteration
            is_answering = False # Reset answering state
            
            # Immediately start a new live display for the next thinking segment
            active_live = Live(
                render_panel("", "Thinking", "yellow", is_active=True),
                console=console,
                auto_refresh=True,
                refresh_per_second=10,
                vertical_overflow="visible"
            )
            active_live.start()

    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            
            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("\n[bold yellow]Goodbye![/bold yellow]")
                break
            
            if not user_input.strip():
                continue
            
            # Reset state
            last_processed_text_len = 0
            current_segment_text = ""
            active_live = None
            
            console.print("\n[bold blue]Search & Reasoning Trace:[/bold blue]")
            
            # Start initial live display to show "Thinking" immediately
            active_live = Live(
                render_panel("", "Thinking", "yellow", is_active=True),
                console=console,
                auto_refresh=True,
                refresh_per_second=10,
                vertical_overflow="visible"
            )
            active_live.start()
            
            # Run workflow
            result = await workflow(
                problem=user_input,
                dataset_name=dataset_name,
                verbose=verbose,
                step_callback=print_step_update,
            )
            
            # Finalize any remaining live display
            if active_live:
                if is_answering:
                    active_live.update(render_panel(current_segment_text, "Answer", "green", is_active=False))
                else:
                    active_live.update(render_panel(current_segment_text, "Thinking", "yellow", is_active=False))
                active_live.stop()
                active_live = None
            
            # Don't print final_response again - it's already been printed via step_callback
            
            # Show detailed tool usage info
            browsed_links = result.get("browsed_links", [])
            searched_links = result.get("searched_links", [])
            total_tool_calls = result.get("total_tool_calls", 0)
            failed_tool_calls = result.get("total_failed_tool_calls", 0)
            
            if browsed_links or searched_links or total_tool_calls > 0:
                info_parts = []
                if searched_links:
                    info_parts.append(f"Searched: {len(searched_links)} links")
                if browsed_links:
                    info_parts.append(f"Browsed: {len(browsed_links)} links")
                if total_tool_calls > 0:
                    info_parts.append(f"Tool calls: {total_tool_calls}")
                if failed_tool_calls > 0:
                    info_parts.append(f"Failed: {failed_tool_calls}")
                if info_parts:
                    console.print(f"[dim]{' | '.join(info_parts)}[/dim]")
            
            # Show links if verbose
            if verbose and searched_links:
                console.print("\n[dim]Searched links:[/dim]")
                for link in searched_links[:5]:  # Show first 5
                    console.print(f"[dim]  - {link}[/dim]")
                if len(searched_links) > 5:
                    console.print(f"[dim]  ... and {len(searched_links) - 5} more[/dim]")
            
            if verbose and browsed_links:
                console.print("\n[dim]Browsed links:[/dim]")
                for link in browsed_links[:5]:  # Show first 5
                    console.print(f"[dim]  - {link}[/dim]")
                if len(browsed_links) > 5:
                    console.print(f"[dim]  ... and {len(browsed_links) - 5} more[/dim]")
            
            console.print()  # Empty line for spacing
            
        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]Interrupted. Type 'exit' to quit or continue chatting.[/bold yellow]\n")
        except Exception as e:
            console.print(f"\n[bold red]Error: {e}[/bold red]\n")
            if verbose:
                import traceback
                console.print(traceback.format_exc())


def chat(
    config: str,
    dataset_name: Optional[str] = None,
    verbose: bool = False,
    config_overrides: Optional[str] = None,
    show_full_tool_output: bool = False,
):
    """
    Start an interactive chat session using the auto_search workflow.
    
    Args:
        config: Path to workflow configuration YAML file
        dataset_name: Optional dataset name for dataset-specific instructions
        verbose: Enable verbose output
        config_overrides: Comma-separated config overrides (e.g., 'param1=value1,param2=value2')
        show_full_tool_output: Show full tool output instead of truncating
    """
    # Parse config overrides
    overrides = {}
    if config_overrides:
        for pair in config_overrides.split(","):
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            
            # Try to convert to appropriate type
            if value.lower() == "true":
                overrides[key] = True
            elif value.lower() == "false":
                overrides[key] = False
            elif value.lower() in ["none", "null"]:
                overrides[key] = None
            elif value.isdigit():
                overrides[key] = int(value)
            else:
                try:
                    overrides[key] = float(value)
                except ValueError:
                    overrides[key] = value
    
    # Set browse_timeout to 10s for interactive chat (unless explicitly overridden)
    if "browse_timeout" not in overrides:
        overrides["browse_timeout"] = 10
    
    # Create workflow
    try:
        workflow = AutoReasonSearchWorkflow(
            configuration=config,
            **overrides
        )
        console.print(f"[green]Loaded workflow from: {config}[/green]\n")
    except Exception as e:
        console.print(f"[bold red]Failed to create workflow: {e}[/bold red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)
    
    # Run chat loop
    try:
        asyncio.run(chat_loop(workflow, dataset_name=dataset_name, verbose=verbose, show_full_tool_output=show_full_tool_output))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted. Exiting.[/bold yellow]")
        sys.exit(0)


if __name__ == "__main__":
    if HAS_RICH:
        # Use typer for CLI parsing
        @app.command()
        def chat_command(
            config: str = typer.Option(
                ...,
                "--config",
                "-c",
                help="Path to workflow configuration YAML file",
            ),
            dataset_name: Optional[str] = typer.Option(
                None,
                "--dataset-name",
                "-d",
                help="Dataset name for dataset-specific instructions (e.g., 'sqav2', 'simpleqa')",
            ),
            verbose: bool = typer.Option(
                False,
                "--verbose",
                "-v",
                help="Enable verbose output (shows tool calls and debugging info)",
            ),
            show_full_tool_output: bool = typer.Option(
                False,
                "--show-full-tool-output",
                help="Show full tool output instead of truncating to 500 chars",
            ),
            config_overrides: Optional[str] = typer.Option(
                None,
                "--config-overrides",
                help="Override configuration parameters in format 'param1=value1,param2=value2'",
            ),
        ):
            """Start an interactive chat session using the auto_search workflow."""
            chat(
                config=config,
                dataset_name=dataset_name,
                verbose=verbose,
                config_overrides=config_overrides,
                show_full_tool_output=show_full_tool_output,
            )
        
        app()
    else:
        # Fallback to argparse for basic CLI
        import argparse
        parser = argparse.ArgumentParser(description="Interactive chat using auto_search workflow")
        parser.add_argument("--config", "-c", required=True, help="Path to workflow configuration YAML file")
        parser.add_argument("--dataset-name", "-d", help="Dataset name for dataset-specific instructions")
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
        parser.add_argument("--show-full-tool-output", action="store_true", help="Show full tool output instead of truncating to 500 chars")
        parser.add_argument("--config-overrides", help="Config overrides in format 'param1=value1,param2=value2'")
        args = parser.parse_args()
        
        chat(
            config=args.config,
            dataset_name=args.dataset_name,
            verbose=args.verbose,
            config_overrides=args.config_overrides,
            show_full_tool_output=args.show_full_tool_output,
        )

