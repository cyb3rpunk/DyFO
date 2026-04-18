import sys
import os

# Add project root to path so 'dyfo' can be imported
sys.path.append(os.getcwd())

from dyfo.core.model_variants import build_encoder
from dyfo.config import DyFOConfig

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    config = DyFOConfig()
    num_nodes = 30
    
    print(f"{'Variant':<15} | {'Parameters':<12}")
    print("-" * 30)
    
    variants = ["tgn", "tgat", "ra_htgn", "gat_static", "roland", "temporal_kg"]
    
    for v in variants:
        try:
            encoder = build_encoder(config, num_nodes=num_nodes, variant=v)
            params = count_parameters(encoder)
            print(f"{v:<15} | {params:,}")
        except Exception as e:
            print(f"{v:<15} | Error: {e}")

if __name__ == "__main__":
    main()