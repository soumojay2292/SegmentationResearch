import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

def get_latest_results_dir(base="results"):
    subdirs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if not subdirs:
        return None
    latest = max(subdirs, key=os.path.getmtime)
    return latest

def analyze_results():
    latest_results = get_latest_results_dir()
    if latest_results is None:
        print("❌ No results folder found.")
        return

    summary_path = os.path.join(latest_results, "summary.csv")
    if not os.path.exists(summary_path):
        print(f"❌ No summary.csv found in {latest_results}")
        return

    print(f"\n📂 Analyzing latest results: {latest_results}\n")

    # Load CSV into DataFrame
    df = pd.read_csv(summary_path)
    print("\n📊 Summary of Results:\n")
    print(df)

    # Create PDF report
    pdf_path = os.path.join(latest_results, "report.pdf")
    with PdfPages(pdf_path) as pdf:
        # Plot Dice scores per model
        plt.figure(figsize=(8, 5))
        df.plot(x="Model", y="Final Val Dice", kind="bar", title="Validation Dice per Model")
        plt.tight_layout()
        pdf.savefig()  # save to PDF
        plt.close()

        # Plot IoU scores per model
        plt.figure(figsize=(8, 5))
        df.plot(x="Model", y="Final Val IoU", kind="bar", title="Validation IoU per Model")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Per-dataset Dice plot
        plt.figure(figsize=(8, 5))
        df.groupby(["Dataset", "Model"])["Final Val Dice"].mean().unstack().plot(kind="bar", title="Dice Scores per Dataset")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Per-dataset IoU plot
        plt.figure(figsize=(8, 5))
        df.groupby(["Dataset", "Model"])["Final Val IoU"].mean().unstack().plot(kind="bar", title="IoU Scores per Dataset")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Add summary table as text page
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        table_text = df.to_string(index=False)
        ax.text(0, 1, "Summary Table:\n\n" + table_text, va="top", ha="left", fontsize=10, family="monospace")
        pdf.savefig()
        plt.close()

    print(f"📑 Combined PDF report saved: {pdf_path}")

if __name__ == "__main__":
    analyze_results()
