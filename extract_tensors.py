
import torch
import sys
import os
from pathlib import Path
from collections import OrderedDict

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:] 
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def create_tensor_files(model_path, output_dir=None):
    
    print(f"Loading model: {model_path}")
    
    try:
      
        data = torch.load(model_path, map_location='cpu')
        print("Successfully loaded model")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False

  
    if not isinstance(data, (dict, OrderedDict)):
        print("Model is not a state dictionary")
        return False

   
    if any(key.startswith('module.') for key in data.keys()):
        print("Removing 'module.' prefix from DataParallel model...")
        clean_state_dict = remove_module_prefix(data)
    else:
        clean_state_dict = data

    
    tensor_dict = {k: v for k, v in clean_state_dict.items() if isinstance(v, torch.Tensor)}

    if not tensor_dict:
        print("No tensors found in the model")
        return False

    print(f"Found {len(tensor_dict)} tensors to extract")

   
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_dir = f"{base_name}_tensors"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Creating tensor files in: {output_path}")

  
    print("Creating metadata files...")
    with open(output_path / "metadata.txt", "w") as f:
        f.write("# CAESAR Model Tensor Metadata\n")
        f.write(f"# Total parameters: {len(tensor_dict)}\n")
        f.write(f"# Original model: {model_path}\n\n")
        for name, tensor in tensor_dict.items():
            f.write(f"{name}: shape={list(tensor.shape)}, dtype={tensor.dtype}, numel={tensor.numel()}\n")

 
    with open(output_path / "parameter_map.txt", "w") as f:
        f.write("# Parameter name mapping for C++ LibTorch loader\n")
        f.write("# Format: original_name -> safe_filename\n\n")
        for name in tensor_dict.keys():
            safe_name = name.replace(".", "_").replace("/", "_")
            f.write(f"{name} -> {safe_name}.pt\n")

 
    print("Extracting individual tensor files...")
    saved_count = 0
    failed_count = 0
    
    for name, tensor in tensor_dict.items():

        safe_name = name.replace(".", "_").replace("/", "_")
        tensor_path = output_path / f"{safe_name}.pt"

        try:
            torch.save(tensor, tensor_path)
            saved_count += 1
            
          
            if saved_count <= 5:
                shape_str = "x".join(map(str, tensor.shape))
                print(f"{name} -> {tensor_path.name} [{shape_str}]")
            elif saved_count == 6:
                print(f"  ... continuing to extract remaining {len(tensor_dict) - 5} tensors")
                
        except Exception as e:
            print(f"Failed to save {name}: {e}")
            failed_count += 1

   
    print(f"\n=== Extraction Complete ===")
    print(f"Successfully saved: {saved_count}/{len(tensor_dict)} tensors")
    if failed_count > 0:
        print(f"Failed to save: {failed_count} tensors")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Files created:")
    print(f"  - metadata.txt (tensor information)")
    print(f"  - parameter_map.txt (name mapping for C++)")
    print(f"  - {saved_count} individual .pt tensor files")
    
    if saved_count > 0:
        print(f"\nYou can now use these files with the C++ CaesarModelLoader:")
        print(f"  ./test_caesar_reader {output_path.absolute()}")
    
    return saved_count > 0

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python extract_tensors.py <model_path> [output_directory]")
        print("Example: python extract_tensors.py caesar_model.pth")
        print("Example: python extract_tensors.py caesar_model.pth my_tensors/")
        sys.exit(1)

    model_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) == 3 else None

    if not os.path.exists(model_path):
        print(f"✗ Model file does not exist: {model_path}")
        sys.exit(1)

    success = create_tensor_files(model_path, output_dir)
    
    if not success:
        print("Failed to extract tensors")
        sys.exit(1)
    else:
        print("Tensor extraction completed successfully!")

if __name__ == "__main__":
    main()
