import asyncio
import json
import click
import httpx
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .models.schemas import BriefRequest, QueryDepth
from .graph.workflow import ResearchWorkflow
from .config import settings

console = Console()


@click.group()
def cli():
    """Research Assistant CLI - Generate context-aware research briefs."""
    pass


@cli.command()
@click.argument('topic')
@click.option('--depth', type=click.Choice(['1', '2', '3']), default='2', 
              help='Research depth: 1=Shallow, 2=Medium, 3=Deep')
@click.option('--follow-up', is_flag=True, help='Mark this as a follow-up query')
@click.option('--user-id', required=True, help='User ID for context tracking')
@click.option('--output', '-o', type=click.Path(), help='Save output to file')
@click.option('--api-url', help='API URL (if using remote API)')
def research(topic: str, depth: str, follow_up: bool, user_id: str, 
             output: Optional[str], api_url: Optional[str]):
    """Generate a research brief for the given topic."""
    asyncio.run(_research_async(topic, depth, follow_up, user_id, output, api_url))


async def _research_async(topic: str, depth: str, follow_up: bool, user_id: str,
                         output: Optional[str], api_url: Optional[str]):
    """Async research execution."""
    
    request = BriefRequest(
        topic=topic,
        depth=QueryDepth(int(depth)),
        follow_up=follow_up,
        user_id=user_id
    )
    
    console.print(f"Researching: [bold blue]{topic}[/bold blue]")
    console.print(f" Depth: [yellow]{QueryDepth(int(depth)).name}[/yellow]")
    console.print(f" User: [green]{user_id}[/green]")
    console.print(f" Follow-up: [cyan]{follow_up}[/cyan]")
    console.print()
    
    if api_url:
        await _call_remote_api(request, api_url, output)
    else:
        await _execute_local_workflow(request, output)


async def _call_remote_api(request: BriefRequest, api_url: str, output: Optional[str]):
    """Call remote API for research."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating research brief...", total=None)
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.post(
                    f"{api_url.rstrip('/')}/brief",
                    json=request.model_dump()
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result["success"]:
                        brief = result["data"]
                        _display_brief(brief)
                        
                        if output:
                            _save_brief(brief, output)
                        
                        console.print(f"\n [green]Research completed successfully![/green]")
                        console.print(f" Trace ID: [dim]{result.get('trace_id', 'N/A')}[/dim]")
                        console.print(f"  Processing time: [dim]{result.get('processing_time', 0):.2f}s[/dim]")
                    else:
                        console.print(f" [red]Research failed: {result['error']}[/red]")
                else:
                    console.print(f" [red]API error: {response.status_code}[/red]")
                    
            except httpx.TimeoutException:
                console.print(" [red]Request timed out. The research may still be processing.[/red]")
            except Exception as e:
                console.print(f" [red]Error: {str(e)}[/red]")


async def _execute_local_workflow(request: BriefRequest, output: Optional[str]):
    """Execute workflow locally."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating research brief...", total=None)
        
        try:
            workflow = ResearchWorkflow()
            
            result = await workflow.execute(request)
            
            if result["success"]:
                brief = result["brief"]
                _display_brief(brief.model_dump())
                
                if output:
                    _save_brief(brief.model_dump(), output)
                
                console.print(f"\n [green]Research completed successfully![/green]")
                console.print(f"Trace ID: [dim]{result.get('trace_id', 'N/A')}[/dim]")
                console.print(f"⏱ Processing time: [dim]{result.get('processing_time', 0):.2f}s[/dim]")
                
                if result.get("errors"):
                    console.print(f"  [yellow]Warnings: {len(result['errors'])}[/yellow]")
                    for error in result["errors"]:
                        console.print(f"   • [dim]{error}[/dim]")
            else:
                console.print(f" [red]Research failed: {result['error']}[/red]")
                
        except Exception as e:
            console.print(f" [red]Error: {str(e)}[/red]")


def _display_brief(brief_data: dict):
    """Display research brief in a formatted way."""
    console.print("\n" + "="*80)
    console.print(Panel.fit(f"[bold]{brief_data['topic']}[/bold]", title="Research Brief"))
    
    console.print(Panel(brief_data['summary'], title=" Summary"))
    
    console.print("\n[bold] Key Findings:[/bold]")
    for i, finding in enumerate(brief_data['key_findings'], 1):
        console.print(f"  {i}. {finding}")
    
    console.print(Panel(brief_data['detailed_analysis'], title=" Detailed Analysis"))
    
    if brief_data.get('references'):
        console.print("\n[bold]References:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Title", style="cyan")
        table.add_column("URL", style="blue")
        table.add_column("Excerpt", style="dim")
        
        for ref in brief_data['references']:
            table.add_row(
                ref['title'][:50] + "..." if len(ref['title']) > 50 else ref['title'],
                ref['url'][:60] + "..." if len(ref['url']) > 60 else ref['url'],
                ref['excerpt'][:100] + "..." if len(ref['excerpt']) > 100 else ref['excerpt']
            )
        
        console.print(table)
    
    console.print(f"\n[dim]Confidence Score: {brief_data.get('confidence_score', 0):.2f}")
    console.print(f"Generated: {brief_data.get('generated_at', 'N/A')}[/dim]")


def _save_brief(brief_data: dict, output_path: str):
    """Save brief to file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            if output_path.endswith('.json'):
                json.dump(brief_data, f, indent=2, default=str)
            else:
                f.write(f"# {brief_data['topic']}\n\n")
                f.write(f"## Summary\n{brief_data['summary']}\n\n")
                f.write("## Key Findings\n")
                for i, finding in enumerate(brief_data['key_findings'], 1):
                    f.write(f"{i}. {finding}\n")
                f.write(f"\n## Detailed Analysis\n{brief_data['detailed_analysis']}\n\n")
                
                if brief_data.get('references'):
                    f.write("## References\n")
                    for ref in brief_data['references']:
                        f.write(f"- [{ref['title']}]({ref['url']})\n")
                        f.write(f"  {ref['excerpt']}\n\n")
        
        console.print(f"[green]Brief saved to: {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to save brief: {str(e)}[/red]")


@cli.command()
@click.option('--user-id', required=True, help='User ID')
@click.option('--api-url', help='API URL (if using remote API)')
def history(user_id: str, api_url: Optional[str]):
    """Show user's research history."""
    asyncio.run(_show_history(user_id, api_url))


async def _show_history(user_id: str, api_url: Optional[str]):
    """Show user history."""
    if api_url:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{api_url.rstrip('/')}/history/{user_id}")
                if response.status_code == 200:
                    result = response.json()
                    if result["success"] and result["data"]:
                        history_data = result["data"]
                        _display_history(history_data)
                    else:
                        console.print(f"[yellow]No history found for user: {user_id}[/yellow]")
                else:
                    console.print(f"[red]API error: {response.status_code}[/red]")
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
    else:
        console.print("[red]Local history viewing not implemented. Use --api-url.[/red]")


def _display_history(history_data: dict):
    """Display user history."""
    briefs = history_data.get('briefs', [])
    
    console.print(f"\n[bold]Research History for {history_data['user_id']}[/bold]")
    console.print(f"Total briefs: {len(briefs)}")
    console.print(f"Account created: {history_data['created_at']}")
    console.print()
    
    if briefs:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Date", style="dim")
        table.add_column("Topic", style="cyan")
        table.add_column("Confidence", style="green")
        table.add_column("Processing Time", style="yellow")
        
        for brief in briefs[-10:]: 
            table.add_row(
                brief['generated_at'][:10],
                brief['topic'][:50] + "..." if len(brief['topic']) > 50 else brief['topic'],
                f"{brief.get('confidence_score', 0):.2f}",
                f"{brief.get('processing_time', 0):.1f}s"
            )
        
        console.print(table)


@cli.command()
def serve():
    """Start the API server."""
    import uvicorn
    console.print(f"Starting Research Assistant API server...")
    console.print(f"Host: {settings.api_host}")
    console.print(f"Port: {settings.api_port}")
    console.print(f"URL: http://{settings.api_host}:{settings.api_port}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()