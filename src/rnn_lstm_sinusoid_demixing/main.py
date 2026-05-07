"""CLI entry point for the sinusoid demixing project."""

import typer

from rnn_lstm_sinusoid_demixing.sdk.sdk import DemixingSDK
from rnn_lstm_sinusoid_demixing.shared.paths import results_dir

app = typer.Typer(help="RNN-LSTM Sinusoid Demixing — Exercise 01")


@app.command()
def run() -> None:
    """Run the full experiment: generate data, train models, evaluate, plot."""
    typer.echo("Starting sinusoid demixing experiment...")
    summary = DemixingSDK().run()
    typer.echo("\nTest MSE results (noise_level=0.1):")
    for model_type, mse in sorted(summary.items(), key=lambda x: x[1]):
        typer.echo(f"  {model_type.upper():6s}  MSE = {mse:.6f}")
    typer.echo(f"\nAll plots and metrics saved to: {results_dir()}")


def main() -> None:
    app()
