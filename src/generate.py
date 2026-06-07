import argparse
import torch
from src.config import NeuraCoderConfig
from src.model import NeuraCoder
from src.tokenizer import NeuraCoderTokenizer

def main():
    parser = argparse.ArgumentParser(description="Generate code with NeuraCoder")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model (config.json + model.pt)")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for code generation")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = NeuraCoderConfig.from_pretrained(args.model_path)
    model = NeuraCoder(config).to(device)
    model.load_state_dict(torch.load(f"{args.model_path}/model.pt", map_location=device))
    model.eval()
    tokenizer = NeuraCoderTokenizer()

    input_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )

    generated = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    print("\n" + "="*50)
    print("Generated Code:\n")
    print(generated)
    print("="*50)

if __name__ == "__main__":
    main()
