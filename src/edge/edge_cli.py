# EDGE-01 MVP - Offline edge-analyze CLI

import click
from src.edge.local_inference import EdgeAnalyzer

@click.command()
@click.argument('audio_path')
def edge_analyze(audio_path):
    """Offline sentiment analysis on edge device."""
    analyzer = EdgeAnalyzer()
    result = analyzer.analyze(audio_path)
    click.echo(result)

if __name__ == '__main__':
    edge_analyze()