#!/usr/bin/env python3
"""
DICOM to PNG Converter
Converts DICOM medical image files to PNG format using PyDICOM library.
"""

import os
import sys
import glob
import numpy as np
import pydicom
from PIL import Image
from pathlib import Path


def normalize_pixel_array(pixel_array, window_center=None, window_width=None):
    """
    Normalize pixel array to 0-255 range for PNG export.

    Args:
        pixel_array: NumPy array of pixel data
        window_center: DICOM window center (optional)
        window_width: DICOM window width (optional)

    Returns:
        Normalized uint8 array
    """
    # Convert to float for processing
    pixel_array = pixel_array.astype(np.float64)

    # Apply windowing if parameters are provided
    if window_center is not None and window_width is not None:
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        pixel_array = np.clip(pixel_array, img_min, img_max)
    else:
        # Use full range if no windowing specified
        img_min = pixel_array.min()
        img_max = pixel_array.max()

    # Normalize to 0-255
    if img_max > img_min:
        pixel_array = ((pixel_array - img_min) / (img_max - img_min)) * 255.0
    else:
        pixel_array = np.zeros_like(pixel_array)

    return pixel_array.astype(np.uint8)


def convert_dicom_to_png(dicom_path, output_path=None, apply_windowing=True):
    """
    Convert a single DICOM file to PNG format.

    Args:
        dicom_path: Path to the DICOM file
        output_path: Path for the output PNG file (optional, auto-generated if None)
        apply_windowing: Whether to apply DICOM windowing parameters

    Returns:
        Path to the created PNG file
    """
    try:
        # Read DICOM file
        dicom = pydicom.dcmread(dicom_path)

        # Get pixel array
        pixel_array = dicom.pixel_array

        # Handle different photometric interpretations
        if hasattr(dicom, 'PhotometricInterpretation'):
            if dicom.PhotometricInterpretation == "MONOCHROME1":
                # Invert for MONOCHROME1 (lower values = brighter)
                pixel_array = np.max(pixel_array) - pixel_array

        # Get windowing parameters if available and requested
        window_center = None
        window_width = None
        if apply_windowing:
            if hasattr(dicom, 'WindowCenter') and hasattr(dicom, 'WindowWidth'):
                # Handle cases where these might be sequences
                window_center = dicom.WindowCenter
                window_width = dicom.WindowWidth
                if isinstance(window_center, (list, pydicom.multival.MultiValue)):
                    window_center = window_center[0]
                if isinstance(window_width, (list, pydicom.multival.MultiValue)):
                    window_width = window_width[0]

        # Normalize pixel array
        normalized_array = normalize_pixel_array(pixel_array, window_center, window_width)

        # Handle RGB images
        if len(normalized_array.shape) == 3:
            # Already RGB
            image = Image.fromarray(normalized_array)
        else:
            # Grayscale
            image = Image.fromarray(normalized_array, mode='L')

        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(dicom_path))[0]
            output_dir = os.path.dirname(dicom_path) or '.'
            output_path = os.path.join(output_dir, f"{base_name}.png")

        # Save as PNG
        image.save(output_path, 'PNG')
        print(f"✓ Converted: {os.path.basename(dicom_path)} → {os.path.basename(output_path)}")

        return output_path

    except Exception as e:
        print(f"✗ Error converting {dicom_path}: {str(e)}", file=sys.stderr)
        return None


def batch_convert_dicom_to_png(input_path='.', output_dir=None, pattern='*.dcm'):
    """
    Batch convert all DICOM files in a directory to PNG.

    Args:
        input_path: Directory containing DICOM files (default: current directory)
        output_dir: Directory for output PNG files (default: same as input)
        pattern: Glob pattern for DICOM files (default: *.dcm)

    Returns:
        List of successfully converted file paths
    """
    # Find all DICOM files
    search_pattern = os.path.join(input_path, pattern)
    dicom_files = glob.glob(search_pattern)

    if not dicom_files:
        print(f"No DICOM files found matching pattern: {search_pattern}")
        return []

    print(f"Found {len(dicom_files)} DICOM file(s)")

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Convert each file
    converted_files = []
    for dicom_path in dicom_files:
        if output_dir:
            base_name = os.path.splitext(os.path.basename(dicom_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.png")
        else:
            output_path = None

        result = convert_dicom_to_png(dicom_path, output_path)
        if result:
            converted_files.append(result)

    print(f"\nSuccessfully converted {len(converted_files)}/{len(dicom_files)} files")
    return converted_files


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert DICOM files to PNG format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single DICOM file
  python dicom_to_png.py input.dcm

  # Convert a single file with custom output name
  python dicom_to_png.py input.dcm -o output.png

  # Batch convert all DICOM files in current directory
  python dicom_to_png.py

  # Batch convert with custom pattern
  python dicom_to_png.py -p "*.dcm" -d output_folder
        """
    )

    parser.add_argument(
        'input',
        nargs='?',
        default='.',
        help='Input DICOM file or directory (default: current directory)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output PNG file path (for single file conversion)'
    )
    parser.add_argument(
        '-d', '--output-dir',
        help='Output directory for batch conversion'
    )
    parser.add_argument(
        '-p', '--pattern',
        default='*.dcm',
        help='File pattern for batch conversion (default: *.dcm)'
    )
    parser.add_argument(
        '--no-windowing',
        action='store_true',
        help='Disable DICOM windowing (use full pixel range)'
    )

    args = parser.parse_args()

    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Single file conversion
        convert_dicom_to_png(
            args.input,
            args.output,
            apply_windowing=not args.no_windowing
        )
    elif os.path.isdir(args.input):
        # Batch conversion
        batch_convert_dicom_to_png(
            args.input,
            args.output_dir,
            args.pattern
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
