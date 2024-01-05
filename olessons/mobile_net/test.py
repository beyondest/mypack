import argparse
import numpy as np
import onnxruntime
import time
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument(
        "--calibrate_dataset", default="./test_images", help="calibration data set"
    )
    parser.add_argument(
        "--quant_format",
        default=QuantFormat.QDQ,
        type=QuantFormat.from_string,
        choices=list(QuantFormat),
    )
    parser.add_argument("--per_channel", default=False, type=bool)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_model_path = args.input_model
    output_model_path = args.output_model
    calibration_dataset_path = args.calibrate_dataset


    # Calibrate and quantize model
    # Turn off model optimization during quantization
    quantize_static(
        input_model_path,
        output_model_path,
        dr,
        quant_format=args.quant_format,
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
    )
    print("Calibrated and quantized model saved.")

    print("benchmarking fp32 model...")
    benchmark(input_model_path)

    print("benchmarking int8 model...")
    benchmark(output_model_path)


if __name__ == "__main__":
    main()