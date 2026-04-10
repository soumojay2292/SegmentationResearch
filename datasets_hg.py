import sys
import importlib

# Force Python to ignore the local 'datasets' folder
sys.path = [p for p in sys.path if "SegmentationResearch" not in p]

# Now import Hugging Face's datasets library
datasets_lib = importlib.import_module("datasets")
dataset = datasets_lib.load_dataset("ronnieee22/segmentation_research")

print(dataset)
