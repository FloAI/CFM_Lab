import argparse
import sys
from CFM_Lab import generate_samples_from_csv

def main():
    parser = argparse.ArgumentParser(
        description="🧬 CFM_Lab: Conditional Flow Matching CLI for Tabular & Scientific Data"
    )

    # --- Required Arguments ---
    parser.add_argument("--data", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--out", type=str, default="./output", help="Directory to save results.")

    # --- Data Config ---
    parser.add_argument("--cond_col", type=str, default=None, help="Name of column to condition on (Single File Mode).")
    parser.add_argument("--cond_path", type=str, default=None, help="Path to separate metadata CSV (Two File Mode).")
    
    # --- Model Config ---
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "transformer"], help="Model architecture.")
    parser.add_argument("--coupling", type=str, default="independent", 
                        choices=["independent", "exact_optimal_transport_cfm", "schrodinger_bridge_cfm"], 
                        help="Flow coupling strategy.")
    
    # --- Training Config ---
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=128, help="Batch size.")
    parser.add_argument("--samples", type=int, default=100, help="Number of synthetic samples to generate.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    # --- Actions ---
    parser.add_argument("--eval", action="store_true", help="Generate evaluation report (plots & metrics).")

    args = parser.parse_args()

    print(f"\n🧪 Starting CFM_Lab Pipeline...")
    print(f"   Input: {args.data}")
    print(f"   Model: {args.model.upper()} with {args.coupling} coupling")

    try:
        # Call the main library function
        generate_samples_from_csv(
            data_path=args.data,
            num_samples=args.samples,
            condition_column_name=args.cond_col,
            cond_path=args.cond_path,
            save_dir=args.out,
            model_type=args.model,
            interpolant=args.coupling,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            seed=args.seed,
            evaluate=args.eval
        )
        print(f"\n Success! Results saved to: {args.out}")
    
    except Exception as e:
        print(f"\n Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
