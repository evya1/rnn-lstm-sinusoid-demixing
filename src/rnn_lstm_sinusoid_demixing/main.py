"""CLI entry point for the sinusoid demixing project."""

import typer

app = typer.Typer(help="RNN-LSTM Sinusoid Demixing — Exercise 01")


@app.command()
def run() -> None:
    """Run a full experiment: generate data, train models, evaluate, plot."""
    typer.echo("rnn-lstm-sinusoid-demixing: project skeleton ready.")
    typer.echo("Implementation coming in Phase 4+.")


def main() -> None:
    app()
