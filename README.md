# Codebase Ultra-Compressor

A powerful utility for compressing codebases into ultra-compact representations that can be reconstructed by large language models like Claude 3.7 Sonnet.

## Overview

This tool takes a folder containing source code and processes its text-based files by either:

1. **Ultra-compressing** them via the OpenAI API (default)
2. **Combining** them without compression into a single document

The tool respects `.gitignore` rules, integrates with Git repositories, and creates a comprehensive output file with a tree-style directory listing followed by the compressed representations of each file.

## Features

- **Git Integration**: Uses Git commands when available to perfectly respect `.gitignore` rules
- **Smart Text File Detection**: Identifies text-based files using multiple methods
- **Robust `.gitignore` Support**: Properly handles complex gitignore patterns including negation and wildcards
- **Token Budget Management**: Allocates tokens based on file sizes to maximize information preservation
- **Timestamp Integration**: Adds EST timestamps to the output filename and content
- **Customizable Verbosity**: From quiet operation to detailed debug information
- **Flexible Output**: Configurable output path and multiple compression options
- **Directory Tree Visualization**: Shows the exact folder structure of your codebase

## Installation

### Prerequisites

- Python 3.6+
- OpenAI API key (for compression mode)
- Git (optional, for better integration)

### Setup

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install openai
```

3. Make the script executable (Unix/Linux/Mac):

```bash
chmod +x codebase_compressor.py
```

## Usage

### Basic Usage

```bash
python codebase_compressor.py /path/to/your/codebase
```

### Command Line Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--output`, | `-o` | Custom output file path (default: timestamped filename) |
| `--model` | `-m` | OpenAI model to use (default: "o3-mini") |
| `--gitignore` | `-g` | Custom .gitignore file path |
| `--preview` | `-p` | Preview ignored files before processing |
| `--api-key` | `-k` | OpenAI API key (alternatively use OPENAI_API_KEY env var) |
| `--verbosity` | `-v` | Verbosity level (0=quiet, 1=normal, 2=verbose, 3=debug) |
| `--use-git` | | Use Git commands if available (default: true) |
| `--no-git` | | Don't use Git commands even if Git is available |
| `--token-limit` | `-t` | Total token limit for compression (default: 25000) |
| `--min-tokens` | | Minimum tokens per file (default: 100) |
| `--max-tokens` | | Maximum tokens per file (default: 1000) |
| `--no-compression` | `-n` | Combine files without compression (no AI usage) |

### Examples

#### Basic compression with default settings:
```bash
python codebase_compressor.py /path/to/project
```

#### Custom output location and higher verbosity:
```bash
python codebase_compressor.py /path/to/project -o custom_output.txt -v 2
```

#### Preview ignored files and use a different model:
```bash
python codebase_compressor.py /path/to/project -p -m gpt-4o-mini
```

#### Combine files without compression:
```bash
python codebase_compressor.py /path/to/project --no-compression
```

#### Allocate more tokens in total:
```bash
python codebase_compressor.py /path/to/project --token-limit 50000
```

## Output Format

The tool generates a single text file containing:

1. **Header**: Script name, generation timestamp
2. **Codebase Structure**: Tree-style representation of the folder structure
3. **Compressed Content**: Each file's content in a compressed format with clear separators

Example of the output format:

```
# ULTRA-COMPRESSED CODEBASE
# Generated on: 2025-04-17 13:45:22 EST

# IMPORTANT NOTICE TO AI MODEL

This document contains an ultra-compressed representation of source code...

# Codebase Structure

```
project/
├── src/
│   ├── main.py
│   └── utils/
│       ├── helpers.py
│       └── config.py
├── tests/
│   └── test_main.py
└── README.md
```

# Ultra-Compressed File Contents

------------------------------ 📄 File: src/main.py (2,345 bytes, target 500 tokens) ------------------------------
[Ultra-compressed representation of main.py]

------------------------------ 📄 File: src/utils/helpers.py (1,234 bytes, target 250 tokens) ------------------------------
[Ultra-compressed representation of helpers.py]

...
```

## Token Budget Management

The tool intelligently allocates tokens to each file based on its size relative to the total codebase size. This ensures that:

- Larger, more complex files get more tokens for compression
- Every file gets at least a minimum number of tokens
- The total token budget stays within your specified limit

This approach maximizes the information preserved during compression while efficiently managing token usage.

## Verbosity Levels

- **0 (Quiet)**: Only errors and critical information
- **1 (Normal)**: Regular progress updates
- **2 (Verbose)**: Detailed metrics and processing information
- **3 (Debug)**: Maximum detail including internal operations

## Use Cases

- **Context Window Optimization**: Compress large codebases to fit within AI context windows
- **Code Summarization**: Create compact representations of codebases for analysis
- **Documentation**: Generate comprehensive file listings with content for documentation
- **Knowledge Transfer**: Share code context efficiently with team members or AI assistants

## Limitations

- Compression quality depends on the OpenAI model used
- Some fine details may be lost during compression
- Binary files are automatically excluded
- Large repositories with many files may take significant time to process

## License

MIT License

## Acknowledgments

Made with ❤️ by Thaer Almadhoun (talmadhoun@gmail.com) to help developers work more effectively with large language models by providing code context in an ultra-compressed format.
