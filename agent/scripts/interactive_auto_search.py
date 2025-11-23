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
from dr_agent.tool_interface.data_types import DocumentToolOutput

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
    final_answer_text = ""  # Store final answer for bibliography extraction
    thinking_text = ""  # Store final thinking text for display
    
    # Track snippets for bibliography (snippet_id -> snippet_info dict)
    snippets_dict = {}

    # Helper to convert number to letter sequence (0->A, 1->B, ..., 25->Z, 26->AA, ...)
    def number_to_letters(n):
        """Convert a number to letter sequence: 0->A, 1->B, ..., 25->Z, 26->AA, 27->AB, ..."""
        result = ""
        while n >= 0:
            result = chr(65 + (n % 26)) + result  # 65 is 'A'
            n = n // 26 - 1
            if n < 0:
                break
        return result
    
    # Helper to reduce newlines
    def clean_text(t):
        return re.sub(r'\n{3,}', '\n\n', t)
    
    # Helper to format citations and think tags with Rich markup
    def format_citations(t):
        # Format citation tags (handles both id= and ids=)
        def format_cite(match):
            cite_id = match.group(2)
            cite_content = match.group(3)
            # Format citation: show cite_id in cyan, content in bold cyan
            # Note: cite_id should not contain brackets, but if it does, they need to be escaped for Rich
            # For IDs like "A-4", no escaping is needed
            return f'[dim]<cite id="[/dim][cyan]{cite_id}[/cyan][dim]">[/dim][bold cyan]{cite_content}[/bold cyan][dim]</cite>[/dim]'
        # Match both id= and ids=, handle comma-separated IDs
        t = re.sub(r'<cite\s+ids?=(["\']?)([^"\'>\s]+)\1[^>]*>([^<]+)</cite>', format_cite, t)
        
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
        nonlocal last_processed_text_len, current_segment_text, active_live, is_answering, snippets_dict, final_answer_text
        
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
                        if is_answering:
                            # This shouldn't happen normally if is_answering is true but no match found in accumulated text?
                            # It means we lost the <answer> tag from the accumulated text?
                            # But current_segment_text is reset to answer_text on transition.
                            # Ah, wait. If is_answering is True, `current_segment_text` ALREADY contains just the answer part from previous iterations.
                            # But `new_chunk` is appended to it at the start of this block: `current_segment_text += new_chunk`.
                            # So `current_segment_text` grows.
                            # The issue is that `answer_match` regex `(.*)<answer>(.*)` expects `<answer>` to be present.
                            # But we STRIPPED `<answer>` when we reset `current_segment_text = answer_text`!
                            # So `answer_match` will fail in subsequent iterations!
                            
                            # FIX: We are already in answering mode. We just append new chunk and render.
                            active_live.update(render_panel(current_segment_text, "Answer", "green", is_active=True, spinner_text="Generating Answer..."))
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
                call_id = getattr(tool_call, "call_id", None)
                header = f"Tool Call: {tool_name}"
                if call_id:
                    header += f" (id={call_id})"
                console.print(f"\n[bold magenta]{header}[/bold magenta]")
                
                # Build snippet info if available
                snippet_sections = []
                if isinstance(tool_call, DocumentToolOutput) and tool_call.documents:
                    snippet_blocks = []
                    for idx, doc in enumerate(tool_call.documents):
                        snippet_id = (
                            f"{tool_call.call_id}-{idx}"
                            if getattr(tool_call, "call_id", None)
                            else doc.id
                        )
                        snippet_content = clean_text(doc.stringify())
                        # Store snippet for bibliography
                        snippets_dict[snippet_id] = {
                            "content": snippet_content,
                            "id": snippet_id,
                            "tool_name": tool_name
                        }
                        snippet_blocks.append(
                            f"[bold]{idx + 1}. Snippet[/bold] [dim](id={snippet_id})[/dim]\n{snippet_content}"
                        )
                    if snippet_blocks:
                        snippet_sections.append("\n\n".join(snippet_blocks))

                if snippet_sections:
                    content = "[cyan]Retrieved Documents[/cyan]\n" + "\n\n".join(
                        snippet_sections
                    )
                else:
                    output = tool_call.output or ""
                    if (
                        not show_full_tool_output
                        and isinstance(output, str)
                        and len(output) > 500
                    ):
                        output = output[:500] + "... [truncated]"
                    content = clean_text(output)

                console.print(
                    Panel(
                        content,
                        title="[green]Output[/green]",
                        border_style="green",
                    )
                )
                
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
            final_answer_text = ""
            thinking_text = ""
            snippets_dict.clear()  # Reset snippets for each query
            
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
            
            # Store final text before processing (answer or thinking)
            if active_live:
                if is_answering:
                    final_answer_text = current_segment_text  # Store final answer for bibliography
                else:
                    thinking_text = current_segment_text  # Store thinking text for display
            
            # Variables for bibliography
            cited_snippet_ids = []
            id_mapping = {}
            
            # Extract citation IDs from final answer and rewrite them
            if final_answer_text:
                # Extract all citation IDs from the answer
                # Pattern matches: <cite id="ID"> or <cite id='ID'> or <cite id=ID> or <cite ids="ID1,ID2">
                citation_pattern = r'<cite\s+ids?=(["\']?)([^"\'>\s]+)\1[^>]*>'
                citation_matches = re.findall(citation_pattern, final_answer_text)
                # Add IDs in order of appearance in the text
                for quote_char, cite_id in citation_matches:
                    # Handle comma-separated IDs (e.g., <cite id="ID1,ID2">)
                    ids = [id.strip() for id in cite_id.split(',')]
                    for id in ids:
                        if id not in cited_snippet_ids:
                            cited_snippet_ids += [id]
                
                # Create mapping from original IDs to letter-based IDs
                # Map by prefix (part before dash) to preserve suffix
                prefix_to_letter = {}
                id_mapping = {}  # original_id -> letter_based_id (already initialized above)
                
                for idx, original_id in enumerate(cited_snippet_ids):
                    # Split ID into prefix and suffix (e.g., "36a93066-7" -> prefix="36a93066", suffix="7")
                    if '-' in original_id:
                        prefix, suffix = original_id.rsplit('-', 1)
                    else:
                        # If no dash, treat entire ID as prefix with empty suffix
                        prefix, suffix = original_id, ""
                    
                    # Get or create letter for this prefix           
                    if prefix not in prefix_to_letter:         
                        prefix_to_letter[prefix] = number_to_letters(idx)
                    
                    letter = prefix_to_letter[prefix]
                    # Reconstruct with letter prefix and original suffix
                    if suffix:
                        new_id = f"{letter}-{suffix}"
                    else:
                        new_id = letter
                    
                    id_mapping[original_id] = new_id
                
                # Replace IDs in final_answer_text
                def replace_cite_id(match):
                    quote_char = match.group(1)
                    original_ids_str = match.group(2)
                    # Handle comma-separated IDs
                    original_ids = [id.strip() for id in original_ids_str.split(',')]
                    new_ids = [id_mapping.get(id, id) for id in original_ids]
                    new_ids_str = ','.join(new_ids)
                    # Use 'ids' if multiple IDs, 'id' if single
                    attr_name = 'ids' if len(new_ids) > 1 else 'id'
                    return f'<cite {attr_name}={quote_char}{new_ids_str}{quote_char}>'
                
                # Replace all citation IDs in the text
                final_answer_text = re.sub(
                    r'<cite\s+ids?=(["\']?)([^"\'>\s]+)\1[^>]*>',
                    replace_cite_id,
                    final_answer_text
                )
                
            
            # Finalize any remaining live display (if not already done)
            # Note: If we had final_answer_text with citations, we already updated and stopped active_live above
            if active_live:
                if is_answering:
                    active_live.update(render_panel(format_citations(final_answer_text), "Answer", "green", is_active=False))
                else:
                    if thinking_text:
                        active_live.update(render_panel(thinking_text, "Thinking", "yellow", is_active=False))
                active_live.stop()
                active_live = None
            
            # Display bibliography after the final answer (if we have citations)
            if final_answer_text and cited_snippet_ids and snippets_dict:
                bibliography_items = []
                for original_id in cited_snippet_ids:
                    if original_id in snippets_dict:
                        snippet_info = snippets_dict[original_id]
                        snippet_content = snippet_info["content"]
                        tool_name = snippet_info["tool_name"]
                        # Use letter-based ID for display
                        display_id = id_mapping.get(original_id, original_id)
                        # Truncate snippet_content after the URL line
                        lines = snippet_content.split('\n')
                        truncated_lines = []
                        url_line_found = False
                        for line in lines:
                            truncated_lines.append(line)
                            # Check if this line contains "URL:" (case-insensitive)
                            if line.strip().upper().startswith('URL:'):
                                url_line_found = True
                                break
                        # If URL line was found, use truncated version; otherwise use original
                        if url_line_found:
                            snippet_content = '\n'.join(truncated_lines)
                        bibliography_items.append(
                            f"[bold]{display_id}[/bold] ({tool_name})\nOriginal ID: {original_id}\n{snippet_content}"
                        )
                
                if bibliography_items:
                    bibliography_text = "\n\n".join(bibliography_items)
                    console.print(
                        Panel(
                            bibliography_text,
                            title="[cyan]Bibliography[/cyan]",
                            border_style="cyan",
                        )
                    )
                    console.print()  # Empty line for spacing
            
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
    
    # Set prompt version for CLI agent
    if "prompt_version" not in overrides:
        overrides["prompt_version"] = "cli"

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
        parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
        parser.add_argument("--show-full-tool-output", action="store_true", help="Show full tool output instead of truncating to 500 chars")
        parser.add_argument("--config-overrides", help="Config overrides in format 'param1=value1,param2=value2'")
        args = parser.parse_args()
        
        # hardcode dataset name to long_form
        chat(
            config=args.config,
            dataset_name="long_form",
            verbose=args.verbose,
            config_overrides=args.config_overrides,
            show_full_tool_output=args.show_full_tool_output,
        )

