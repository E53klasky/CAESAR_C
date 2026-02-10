#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "data_utils.h"
#include "dataset/dataset.h"
#include "models/array_utils.h"
#include "models/caesar_compress.h"
#include "models/caesar_decompress.h"

void print_usage(const char* program_name) {
  std::cout << "CAESAR - Compression Tool for Scientific Data\n\n";
  std::cout << "Usage:\n";
  std::cout << "  " << program_name << " compress <input.bin> [options]\n";
  std::cout << "  " << program_name << " decompress <input.cae> [options]\n\n";
  std::cout << "Commands:\n";
  std::cout << "  compress       Compress a raw binary file\n";
  std::cout << "  decompress     Decompress a .cae file\n\n";
  std::cout << "Options:\n";
  std::cout << "  -o, --output <file>          Output file path (default: "
               "auto-generated)\n";
  std::cout << "  -s, --shape <T,C,H,W> or <T,H,W>  Data shape (required for "
               "compress)\n";
  std::cout << "  -e, --error-bound <value>    Relative error bound (default: "
               "0.001)\n";
  std::cout << "  -b, --batch-size <size>      Batch size (default: 128)\n";
  std::cout << "  -f, --n-frame <frames>       Number of frames (default: "
               "8)\n";
  std::cout << "  -m, --model <type>           Model type: V or D (default: "
               "V) [D not yet implemented]\n";
  std::cout << "  --compress-device <device>   Device for compression: cpu or "
               "cuda (default: auto)\n";
  std::cout << "  --decompress-device <device> Device for decompression: cpu "
               "or cuda (default: auto)\n";
  std::cout << "  -t, --timing                 Show timing information\n";
  std::cout << "  --metadata                   Show detailed metadata "
               "statistics\n";
  std::cout << "  -v, --verbose                Verbose output\n";
  std::cout << "  -q, --quiet                  Minimal output\n";
  std::cout << "  --verify                     Verify decompression by "
               "computing metrics\n";
  std::cout << "  --metrics-csv <file>         Save metrics to CSV file\n";
  std::cout << "  --preset <level>             Compression preset: fast "
               "(0.01), balanced (0.001), best (0.0001)\n";
  std::cout << "  --force-padding              Force spatial padding\n";
  std::cout << "  -h, --help                   Show this help message\n\n";
  std::cout << "Examples:\n";
  std::cout << "  # Compress with shape 20,256,256\n";
  std::cout << "  " << program_name
            << " compress data.bin -s 20,256,256 -o compressed.cae\n\n";
  std::cout << "  # Compress with custom error bound and timing\n";
  std::cout << "  " << program_name
            << " compress data.bin -s 1,1,20,256,256 -e 0.0001 -t "
               "--metadata\n\n";
  std::cout << "  # Decompress with verification\n";
  std::cout << "  " << program_name
            << " decompress compressed.cae -o output.bin --verify\n\n";
  std::cout << "  # Use CUDA for compression, CPU for decompression\n";
  std::cout << "  " << program_name
            << " compress data.bin -s 20,256,256 --compress-device cuda\n\n";
}

std::vector<int64_t> parse_shape(const std::string& shape_str) {
  std::vector<int64_t> shape;
  std::stringstream ss(shape_str);
  std::string token;
  while (std::getline(ss, token, ',')) {
    shape.push_back(std::stoll(token));
  }
  return shape;
}

torch::Device parse_device(const std::string& device_str) {
  std::string lower = device_str;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  if (lower == "cpu") {
    return torch::Device(torch::kCPU);
  } else if (lower == "cuda" || lower == "gpu") {
    if (torch::cuda::is_available()) {
      return torch::Device(torch::kCUDA);
    } else {
      std::cerr << "Warning: CUDA requested but not available, using CPU\n";
      return torch::Device(torch::kCPU);
    }
  } else {
    throw std::runtime_error("Unknown device: " + device_str);
  }
}

torch::Device auto_select_device() {
  if (torch::cuda::is_available()) {
    return torch::Device(torch::kCUDA);
  }
  return torch::Device(torch::kCPU);
}

torch::Tensor load_raw_binary(const std::string& bin_path,
                              const std::vector<int64_t>& shape,
                              bool verbose = false) {
  std::ifstream file(bin_path, std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Cannot open binary file: " + bin_path);

  size_t num_elems = 1;
  for (auto d : shape) {
    if (d <= 0) throw std::runtime_error("Invalid shape dimension");
    num_elems *= static_cast<size_t>(d);
  }

  std::vector<float> buf(num_elems);
  file.read(reinterpret_cast<char*>(buf.data()),
            static_cast<std::streamsize>(num_elems * sizeof(float)));
  if (!file)
    throw std::runtime_error("Failed to read expected floats from " + bin_path);
  file.close();

  torch::Tensor t =
      torch::from_blob(buf.data(), torch::IntArrayRef(shape), torch::kFloat32)
          .clone();

  if (verbose) {
    std::cout << "Loaded " << bin_path << " with shape " << t.sizes() << "\n";
    std::cout << "  Min: " << t.min().item<float>()
              << ", Max: " << t.max().item<float>() << "\n";
  }
  return t;
}

void save_tensor_to_bin(const torch::Tensor& tensor,
                        const std::string& filename, bool verbose = false) {
  torch::Tensor cpu = tensor.to(torch::kCPU).contiguous();
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Error opening " + filename + " for write");
  }
  file.write(reinterpret_cast<const char*>(cpu.data_ptr<float>()),
             static_cast<std::streamsize>(cpu.numel() * sizeof(float)));
  file.close();
  if (verbose) {
    std::cout << "Saved tensor to " << filename << "\n";
  }
}

bool save_encoded_streams(const std::vector<std::string>& streams,
                          const std::string& filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file to write: " << filename << std::endl;
    return false;
  }
  for (const auto& s : streams) {
    uint64_t len = static_cast<uint64_t>(s.size());
    file.write(reinterpret_cast<const char*>(&len), sizeof(len));
    if (len) file.write(s.data(), static_cast<std::streamsize>(len));
  }
  file.close();
  return true;
}

std::vector<std::string> load_encoded_streams(const std::string& filename) {
  std::vector<std::string> out;
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file to read: " << filename << std::endl;
    return out;
  }
  uint64_t len;
  while (file.read(reinterpret_cast<char*>(&len), sizeof(len))) {
    std::string s;
    if (len) {
      s.resize(len);
      if (!file.read(&s[0], static_cast<std::streamsize>(len))) {
        std::cerr << "Error: truncated read while reading " << filename
                  << std::endl;
        break;
      }
    }
    out.push_back(std::move(s));
  }
  file.close();
  return out;
}

template <typename T>
size_t get_vector_data_size(const std::vector<T>& vec) {
  if (vec.empty()) return 0;
  return vec.size() * sizeof(T);
}

template <typename T>
size_t get_2d_vector_data_size(const std::vector<std::vector<T>>& vec_2d) {
  size_t total_bytes = 0;
  for (const auto& inner_vec : vec_2d) {
    total_bytes += inner_vec.size() * sizeof(T);
  }
  return total_bytes;
}

size_t calculate_metadata_size(const CompressionResult& result) {
  size_t total_bytes = 0;
  total_bytes += get_vector_data_size(result.gae_comp_data);
  total_bytes += sizeof(result.num_samples);
  total_bytes += sizeof(result.num_batches);

  const auto& meta = result.compressionMetaData;
  total_bytes += get_vector_data_size(meta.offsets);
  total_bytes += get_vector_data_size(meta.scales);
  total_bytes += get_2d_vector_data_size(meta.indexes);
  total_bytes += sizeof(std::get<0>(meta.block_info));
  total_bytes += sizeof(std::get<1>(meta.block_info));
  total_bytes += get_vector_data_size(std::get<2>(meta.block_info));
  total_bytes += get_vector_data_size(meta.data_input_shape);
  total_bytes += get_vector_data_size(meta.filtered_blocks);
  total_bytes += sizeof(meta.global_scale);
  total_bytes += sizeof(meta.global_offset);
  total_bytes += sizeof(meta.pad_T);

  const auto& gae_meta = result.gaeMetaData;
  total_bytes += sizeof(gae_meta.GAE_correction_occur);
  total_bytes += get_vector_data_size(gae_meta.padding_recon_info);
  total_bytes += get_2d_vector_data_size(gae_meta.pcaBasis);
  total_bytes += get_vector_data_size(gae_meta.uniqueVals);
  total_bytes += sizeof(gae_meta.quanBin);
  total_bytes += sizeof(gae_meta.nVec);
  total_bytes += sizeof(gae_meta.prefixLength);
  total_bytes += sizeof(gae_meta.dataBytes);
  total_bytes += sizeof(gae_meta.coeffIntBytes);

  return total_bytes;
}

void print_metadata_stats(const CompressionResult& result) {
  std::cout << "\n=== DETAILED METADATA BREAKDOWN ===\n";

  const auto& meta = result.compressionMetaData;
  const auto& gae_meta = result.gaeMetaData;

  std::cout << "Compression Metadata:\n";
  std::cout << "  - Offsets vector size: " << meta.offsets.size() << " ("
            << get_vector_data_size(meta.offsets) << " bytes)\n";
  std::cout << "  - Scales vector size: " << meta.scales.size() << " ("
            << get_vector_data_size(meta.scales) << " bytes)\n";
  std::cout << "  - Indexes size: " << meta.indexes.size() << " ("
            << get_2d_vector_data_size(meta.indexes) << " bytes)\n";
  std::cout << "  - Block info (nH, nW): (" << std::get<0>(meta.block_info)
            << ", " << std::get<1>(meta.block_info) << ")\n";
  std::cout << "  - Padding info size: " << std::get<2>(meta.block_info).size()
            << " (" << get_vector_data_size(std::get<2>(meta.block_info))
            << " bytes)\n";
  std::cout << "  - Input shape: [";
  for (size_t i = 0; i < meta.data_input_shape.size(); ++i) {
    std::cout << meta.data_input_shape[i];
    if (i < meta.data_input_shape.size() - 1) std::cout << ", ";
  }
  std::cout << "]\n";
  std::cout << "  - Filtered blocks: " << meta.filtered_blocks.size() << " ("
            << get_vector_data_size(meta.filtered_blocks) << " bytes)\n";
  std::cout << "  - Global scale: " << meta.global_scale << "\n";
  std::cout << "  - Global offset: " << meta.global_offset << "\n";
  std::cout << "  - Pad T: " << meta.pad_T << "\n";

  std::cout << "\nGAE Metadata:\n";
  std::cout << "  - GAE correction occurred: "
            << (gae_meta.GAE_correction_occur ? "Yes" : "No") << "\n";
  std::cout << "  - Padding recon info size: "
            << gae_meta.padding_recon_info.size() << " ("
            << get_vector_data_size(gae_meta.padding_recon_info) << " bytes)\n";
  std::cout << "  - PCA Basis dimensions: " << gae_meta.pcaBasis.size();
  if (!gae_meta.pcaBasis.empty()) {
    std::cout << " x " << gae_meta.pcaBasis[0].size();
  }
  std::cout << " (" << get_2d_vector_data_size(gae_meta.pcaBasis)
            << " bytes)\n";
  std::cout << "  - Unique values: " << gae_meta.uniqueVals.size() << " ("
            << get_vector_data_size(gae_meta.uniqueVals) << " bytes)\n";
  std::cout << "  - Quantization bin: " << gae_meta.quanBin << "\n";
  std::cout << "  - nVec: " << gae_meta.nVec << "\n";
  std::cout << "  - Prefix length: " << gae_meta.prefixLength << "\n";
  std::cout << "  - Data bytes: " << gae_meta.dataBytes << "\n";
  std::cout << "  - Coeff int bytes: " << gae_meta.coeffIntBytes << "\n";

  std::cout << "\nOther Compression Data:\n";
  std::cout << "  - GAE compressed data: " << result.gae_comp_data.size()
            << " bytes\n";
  std::cout << "  - Number of samples: " << result.num_samples << "\n";
  std::cout << "  - Number of batches: " << result.num_batches << "\n";
  std::cout << "  - Encoded latents streams: " << result.encoded_latents.size()
            << "\n";
  std::cout << "  - Encoded hyper latents streams: "
            << result.encoded_hyper_latents.size() << "\n";
}

double calculate_psnr(const torch::Tensor& original,
                      const torch::Tensor& reconstructed) {
  torch::Tensor orig_cpu = original.to(torch::kCPU);
  torch::Tensor recon_cpu = reconstructed.to(torch::kCPU);

  double max_val = orig_cpu.max().item<double>();
  double min_val = orig_cpu.min().item<double>();
  double range = max_val - min_val;

  torch::Tensor diff = recon_cpu - orig_cpu;
  double mse = diff.pow(2).mean().item<double>();

  if (mse == 0.0) return std::numeric_limits<double>::infinity();

  double psnr = 20.0 * std::log10(range) - 10.0 * std::log10(mse);
  return psnr;
}

void save_metrics_to_csv(
    const std::string& filename, const std::string& input_file,
    const std::vector<int64_t>& shape, double compression_time,
    double decompression_time, uint64_t uncompressed_bytes,
    uint64_t compressed_bytes, size_t metadata_bytes, double cr_with_meta,
    double cr_without_meta, double nrmse, double psnr, float error_bound,
    int batch_size, int n_frame, const std::string& model_type,
    const std::string& compress_device, const std::string& decompress_device) {
  bool file_exists = std::filesystem::exists(filename);
  std::ofstream file(filename, std::ios::app);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open CSV file: " << filename << std::endl;
    return;
  }

  if (!file_exists) {
    file << "timestamp,input_file,shape,error_bound,batch_size,n_frame,model,"
         << "compress_device,decompress_device,"
         << "uncompressed_bytes,compressed_bytes,metadata_bytes,"
         << "cr_with_meta,cr_without_meta,"
         << "compression_time_s,decompression_time_s,total_time_s,"
         << "nrmse,psnr\n";
  }

  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  std::stringstream timestamp;
  timestamp << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");

  std::stringstream shape_str;
  shape_str << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    shape_str << shape[i];
    if (i < shape.size() - 1) shape_str << "x";
  }
  shape_str << "]";

  file << timestamp.str() << "," << input_file << "," << shape_str.str() << ","
       << error_bound << "," << batch_size << "," << n_frame << ","
       << model_type << "," << compress_device << "," << decompress_device
       << "," << uncompressed_bytes << "," << compressed_bytes << ","
       << metadata_bytes << "," << std::fixed << std::setprecision(4)
       << cr_with_meta << "," << cr_without_meta << "," << std::setprecision(6)
       << compression_time << "," << decompression_time << ","
       << (compression_time + decompression_time) << "," << std::setprecision(8)
       << nrmse << "," << std::setprecision(4) << psnr << "\n";

  file.close();
  std::cout << "Metrics saved to " << filename << "\n";
}

int compress_file(const std::string& input_file, const std::string& output_file,
                  const std::vector<int64_t>& shape, float error_bound,
                  int batch_size, int n_frame, const std::string& model_type,
                  torch::Device compress_device, bool show_timing,
                  bool show_metadata, bool verbose, bool quiet,
                  bool force_padding, const std::string& metrics_csv) {
  if (!quiet) {
    std::cout << "=== CAESAR COMPRESSION ===\n";
    std::cout << "Input file: " << input_file << "\n";
    std::cout << "Output file: " << output_file << "\n";
    std::cout << "Model: CAESAR-" << model_type << "\n";
    std::cout << "Compression device: " << compress_device << "\n";
    std::cout << "Error bound: " << error_bound << "\n";
    std::cout << "Batch size: " << batch_size << "\n";
    std::cout << "N-frame: " << n_frame << "\n\n";
  }

  torch::Tensor raw = load_raw_binary(input_file, shape, verbose);
  raw = raw.squeeze();

  if (verbose) {
    std::cout << "After squeeze, shape: " << raw.sizes() << "\n";
  }

  torch::Tensor raw_copy = raw.clone();

  torch::Tensor raw_5d;
  PaddingInfo padding_info;

  if (shape.size() >= 5 && shape[3] >= 128 && shape[4] >= 128) {
    std::tie(raw_5d, padding_info) =
        to_5d_and_pad(raw, shape[3], shape[4], force_padding);
  } else if (shape.size() == 4 || shape.size() == 3) {
    std::tie(raw_5d, padding_info) =
        to_5d_and_pad(raw, 128, 128, force_padding);
  } else {
    std::tie(raw_5d, padding_info) =
        to_5d_and_pad(raw, 256, 256, force_padding);
  }

  Compressor compressor(compress_device);

  DatasetConfig config;
  config.memory_data = raw_5d;
  config.device = compress_device;
  config.variable_idx = 0;
  config.n_frame = n_frame;
  config.dataset_name = "CAESAR Compression Dataset";
  config.section_range = std::nullopt;
  config.frame_range = std::nullopt;
  config.train_size = 256;
  config.inst_norm = true;
  config.norm_type = "mean_range";
  config.train_mode = false;
  config.n_overlap = 0;
  config.test_size = {256, 256};
  config.augment_type = {};

  auto start_time_c = std::chrono::high_resolution_clock::now();
  CompressionResult comp = compressor.compress(config, batch_size, error_bound);
  auto end_time_c = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> compression_time = end_time_c - start_time_c;

  if (show_timing || verbose) {
    std::cout << "\n⏱️  Compression time: " << compression_time.count()
              << " s\n";
  }

  std::string base_output = output_file;
  if (base_output.empty()) {
    base_output = input_file + ".cae";
  }

  std::string latents_file = base_output + ".latents";
  std::string hyper_file = base_output + ".hyper";

  if (!save_encoded_streams(comp.encoded_latents, latents_file)) {
    std::cerr << "Failed to save encoded_latents\n";
    return 1;
  }
  if (!save_encoded_streams(comp.encoded_hyper_latents, hyper_file)) {
    std::cerr << "Failed to save encoded_hyper_latents\n";
    return 1;
  }

  uint64_t compressed_bytes = 0;
  for (const auto& s : comp.encoded_latents) compressed_bytes += s.size();
  for (const auto& s : comp.encoded_hyper_latents) compressed_bytes += s.size();

  uint64_t num_elements = 1;
  for (auto d : shape) num_elements *= static_cast<uint64_t>(d);
  uint64_t uncompressed_bytes = num_elements * sizeof(float);

  size_t metadata_bytes = calculate_metadata_size(comp);

  double cr_without_meta = (compressed_bytes > 0)
                               ? static_cast<double>(uncompressed_bytes) /
                                     static_cast<double>(compressed_bytes)
                               : 0.0;

  double cr_with_meta = (compressed_bytes + metadata_bytes > 0)
                            ? static_cast<double>(uncompressed_bytes) /
                                  (static_cast<double>(compressed_bytes) +
                                   static_cast<double>(metadata_bytes))
                            : 0.0;

  if (!quiet) {
    std::cout << "\n Compression Statistics:\n";
    std::cout << "  Uncompressed size: " << uncompressed_bytes << " bytes ("
              << (uncompressed_bytes / 1024.0 / 1024.0) << " MB)\n";
    std::cout << "  Compressed size:   " << compressed_bytes << " bytes ("
              << (compressed_bytes / 1024.0 / 1024.0) << " MB)\n";
    std::cout << "  Metadata size:     " << metadata_bytes << " bytes ("
              << (metadata_bytes / 1024.0 / 1024.0) << " MB)\n";
    std::cout << "  CR (without metadata): " << std::fixed
              << std::setprecision(4) << cr_without_meta << ":1\n";
    std::cout << "  CR (with metadata):    " << cr_with_meta << ":1\n";
  }

  if (show_metadata) {
    print_metadata_stats(comp);
  }

  if (!metrics_csv.empty()) {
    save_metrics_to_csv(
        metrics_csv, input_file, shape, compression_time.count(), 0.0,
        uncompressed_bytes, compressed_bytes, metadata_bytes, cr_with_meta,
        cr_without_meta, 0.0, 0.0,  // NRMSE and PSNR not available yet
        error_bound, batch_size, n_frame, model_type,
        compress_device.is_cuda() ? "cuda" : "cpu", "N/A");
  }

  if (!quiet) {
    std::cout << "\n Compression complete!\n";
    std::cout << "Output files:\n";
    std::cout << "  - " << latents_file << "\n";
    std::cout << "  - " << hyper_file << "\n";
  }

  return 0;
}

int decompress_file(const std::string& input_base,
                    const std::string& output_file,
                    const std::vector<int64_t>& original_shape, int batch_size,
                    int n_frame, torch::Device decompress_device,
                    bool show_timing, bool verbose, bool quiet, bool verify,
                    const std::string& original_file,
                    const std::string& metrics_csv) {
  if (!quiet) {
    std::cout << "=== CAESAR DECOMPRESSION ===\n";
    std::cout << "Input base: " << input_base << "\n";
    std::cout << "Output file: " << output_file << "\n";
    std::cout << "Decompression device: " << decompress_device << "\n\n";
  }

  std::string latents_file = input_base + ".latents";
  std::string hyper_file = input_base + ".hyper";

  std::vector<std::string> loaded_latents = load_encoded_streams(latents_file);
  std::vector<std::string> loaded_hyper = load_encoded_streams(hyper_file);

  if (loaded_latents.empty() || loaded_hyper.empty()) {
    std::cerr << "Error: Failed to load compressed streams\n";
    return 1;
  }

  if (verbose) {
    std::cout << "Loaded " << loaded_latents.size() << " latent streams and "
              << loaded_hyper.size() << " hyper streams\n";
  }

  CompressionResult comp;
  comp.encoded_latents = loaded_latents;
  comp.encoded_hyper_latents = loaded_hyper;

  // Decompress
  auto start_time_d = std::chrono::high_resolution_clock::now();
  Decompressor decompressor(decompress_device);
  torch::Tensor recon = decompressor.decompress(batch_size, n_frame, comp);
  auto end_time_d = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> decompression_time = end_time_d - start_time_d;

  if (show_timing || verbose) {
    std::cout << "\n  Decompression time: " << decompression_time.count()
              << " s\n";
  }

  if (!recon.defined() || recon.numel() == 0) {
    std::cerr << "Error: Decompression failed - empty tensor\n";
    return 1;
  }

  if (verbose) {
    std::cout << "Reconstructed tensor shape: " << recon.sizes() << "\n";
  }

  PaddingInfo padding_info;
  torch::Tensor restored = restore_from_5d(recon, padding_info);

  save_tensor_to_bin(restored, output_file, verbose);

  // Verification
  if (verify && !original_file.empty()) {
    if (!quiet) std::cout << "\n Verifying reconstruction...\n";

    torch::Tensor original =
        load_raw_binary(original_file, original_shape, false);
    original = original.squeeze();

    torch::Tensor orig_cpu = original.to(torch::kCPU);
    torch::Tensor recon_cpu = restored.to(torch::kCPU);

    torch::Tensor diff = recon_cpu - orig_cpu;
    double mse = diff.pow(2).mean().item<double>();
    double rmse = std::sqrt(mse);
    double nrmse =
        rmse / (orig_cpu.max().item<double>() - orig_cpu.min().item<double>());
    double psnr = calculate_psnr(original, restored);

    if (!quiet) {
      std::cout << "\n Quality Metrics:\n";
      std::cout << "  NRMSE: " << std::scientific << std::setprecision(6)
                << nrmse << "\n";
      std::cout << "  PSNR:  " << std::fixed << std::setprecision(2) << psnr
                << " dB\n";
    }

    if (!metrics_csv.empty()) {
      uint64_t num_elements = 1;
      for (auto d : original_shape) num_elements *= static_cast<uint64_t>(d);
      uint64_t uncompressed_bytes = num_elements * sizeof(float);

      save_metrics_to_csv(metrics_csv, original_file, original_shape, 0.0,
                          decompression_time.count(), uncompressed_bytes, 0, 0,
                          0.0, 0.0, nrmse, psnr, 0.0, batch_size, n_frame, "V",
                          "N/A", decompress_device.is_cuda() ? "cuda" : "cpu");
    }
  }

  if (!quiet) {
    std::cout << "\n Decompression complete!\n";
    std::cout << "Output: " << output_file << "\n";
  }

  return 0;
}

int main(int argc, char* argv[]) {
  try {
    if (argc < 2) {
      print_usage(argv[0]);
      return 1;
    }

    std::string command = argv[1];
    if (command == "-h" || command == "--help") {
      print_usage(argv[0]);
      return 0;
    }

    if (command != "compress" && command != "decompress") {
      std::cerr << "Error: Unknown command '" << command << "'\n";
      print_usage(argv[0]);
      return 1;
    }

    if (argc < 3) {
      std::cerr << "Error: Missing input file\n";
      print_usage(argv[0]);
      return 1;
    }

    std::string input_file = argv[2];

    // Default parameters
    std::string output_file;
    std::vector<int64_t> shape;
    float error_bound = 0.001f;
    int batch_size = 128;
    int n_frame = 8;
    std::string model_type = "V";
    std::string compress_device_str;
    std::string decompress_device_str;
    bool show_timing = false;
    bool show_metadata = false;
    bool verbose = false;
    bool quiet = false;
    bool verify = false;
    bool force_padding = false;
    std::string metrics_csv;
    std::string original_file;

    // Parse arguments
    for (int i = 3; i < argc; ++i) {
      std::string arg = argv[i];

      if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
        output_file = argv[++i];
      } else if ((arg == "-s" || arg == "--shape") && i + 1 < argc) {
        shape = parse_shape(argv[++i]);
      } else if ((arg == "-e" || arg == "--error-bound") && i + 1 < argc) {
        error_bound = std::stof(argv[++i]);
      } else if ((arg == "-b" || arg == "--batch-size") && i + 1 < argc) {
        batch_size = std::stoi(argv[++i]);
      } else if ((arg == "-f" || arg == "--n-frame") && i + 1 < argc) {
        n_frame = std::stoi(argv[++i]);
      } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
        model_type = argv[++i];
        std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                       ::toupper);
        if (model_type != "V" && model_type != "D") {
          std::cerr << "Error: Model must be 'V' or 'D'\n";
          return 1;
        }
        if (model_type == "D") {
          std::cerr
              << "Warning: CAESAR-D is not yet implemented, using CAESAR-V\n";
          model_type = "V";
        }
      } else if (arg == "--compress-device" && i + 1 < argc) {
        compress_device_str = argv[++i];
      } else if (arg == "--decompress-device" && i + 1 < argc) {
        decompress_device_str = argv[++i];
      } else if (arg == "-t" || arg == "--timing") {
        show_timing = true;
      } else if (arg == "--metadata") {
        show_metadata = true;
      } else if (arg == "-v" || arg == "--verbose") {
        verbose = true;
      } else if (arg == "-q" || arg == "--quiet") {
        quiet = true;
      } else if (arg == "--verify") {
        verify = true;
      } else if (arg == "--force-padding") {
        force_padding = true;
      } else if (arg == "--metrics-csv" && i + 1 < argc) {
        metrics_csv = argv[++i];
      } else if (arg == "--preset" && i + 1 < argc) {
        std::string preset = argv[++i];
        std::transform(preset.begin(), preset.end(), preset.begin(), ::tolower);
        if (preset == "fast") {
          error_bound = 0.01f;
        } else if (preset == "balanced") {
          error_bound = 0.001f;
        } else if (preset == "best") {
          error_bound = 0.0001f;
        } else {
          std::cerr << "Warning: Unknown preset '" << preset
                    << "', using default error bound\n";
        }
      } else if (arg == "--original" && i + 1 < argc) {
        original_file = argv[++i];
      } else if (arg == "-h" || arg == "--help") {
        print_usage(argv[0]);
        return 0;
      } else {
        std::cerr << "Warning: Unknown argument '" << arg << "'\n";
      }
    }

    if (command == "compress") {
      if (shape.empty()) {
        std::cerr
            << "Error: Shape is required for compression (-s or --shape)\n";
        return 1;
      }

      torch::Device compress_device = compress_device_str.empty()
                                          ? auto_select_device()
                                          : parse_device(compress_device_str);

      if (output_file.empty()) {
        output_file = input_file + ".cae";
      }

      return compress_file(input_file, output_file, shape, error_bound,
                           batch_size, n_frame, model_type, compress_device,
                           show_timing, show_metadata, verbose, quiet,
                           force_padding, metrics_csv);

    } else if (command == "decompress") {
      torch::Device decompress_device =
          decompress_device_str.empty() ? auto_select_device()
                                        : parse_device(decompress_device_str);

      if (output_file.empty()) {
        std::string base = input_file;
        if (base.size() >= 4 && base.substr(base.size() - 4) == ".cae") {
          base = base.substr(0, base.size() - 4);
        }
        output_file = base + ".decompressed.bin";
      }

      if (verify && original_file.empty()) {
        std::cerr << "Warning: --verify requires --original <file>, skipping "
                     "verification\n";
        verify = false;
      }

      return decompress_file(input_file, output_file, shape, batch_size,
                             n_frame, decompress_device, show_timing, verbose,
                             quiet, verify, original_file, metrics_csv);
    }

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
