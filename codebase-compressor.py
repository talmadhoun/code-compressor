#!/usr/bin/env python3
"""
Codebase Ultra-Compressor

This script takes a folder path as input and creates an ultra-compressed representation
of the codebase by sending each file's content to OpenAI's API for compression.
It respects .gitignore rules and works with text-based files.

Generated on: {datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S')} EST
"""

import os
import sys
import argparse
import fnmatch
import mimetypes
import pathlib
from typing import List, Dict, Set, Optional, Tuple
import json
import logging
import re
import openai
from datetime import datetime, timezone, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Default values
DEFAULT_MODEL = "o3-mini"
# Generate timestamped filename
TIMESTAMP = datetime.now(timezone(timedelta(hours=-5))).strftime('%Y%m%d_%H%M%S')
DEFAULT_OUTPUT_FILE = f"ultra-compressed-codebase_{TIMESTAMP}.txt"
DEFAULT_TEXT_EXTENSIONS = [
    ".py", ".js", ".ts", ".html", ".css", ".json", ".md", ".txt", 
    ".jsx", ".tsx", ".vue", ".yml", ".yaml", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".c", ".cpp", ".h", ".hpp", ".java", ".go", ".rb",
    ".php", ".swift", ".rs", ".scala", ".pl", ".pm", ".sql", ".xml",
]

# Token limit settings
DEFAULT_TOTAL_TOKEN_LIMIT = 25000  # Default total tokens (fits in about half of an AI session)
DEFAULT_MIN_TOKENS_PER_FILE = 100   # Minimum tokens to allocate for very small files
DEFAULT_MAX_TOKENS_PER_FILE = 1000  # Maximum tokens to allocate for very large files

# Verbosity levels
VERBOSITY_QUIET = 0    # Only errors and critical information
VERBOSITY_NORMAL = 1   # Default logging (INFO level)
VERBOSITY_VERBOSE = 2  # Detailed logging
VERBOSITY_DEBUG = 3    # Debug level (maximum verbosity)

def set_verbosity(verbosity_level: int) -> None:
    """
    Set logging level based on verbosity.
    """
    if verbosity_level == VERBOSITY_QUIET:
        logger.setLevel(logging.WARNING)
    elif verbosity_level == VERBOSITY_NORMAL:
        logger.setLevel(logging.INFO)
    elif verbosity_level == VERBOSITY_VERBOSE:
        logger.setLevel(logging.INFO)  # Still INFO but with more messages
    elif verbosity_level == VERBOSITY_DEBUG:
        logger.setLevel(logging.DEBUG)
    
    logger.debug(f"Verbosity level set to {verbosity_level}")

def get_file_size(file_path: str) -> int:
    """
    Get the size of a file in bytes.
    Returns 0 if the file cannot be accessed.
    """
    try:
        return os.path.getsize(file_path)
    except Exception:
        return 0

def calculate_token_limits(file_paths: List[str], total_token_limit: int, min_tokens_per_file: int, max_tokens_per_file: int, verbosity: int) -> Dict[str, int]:
    """
    Calculate token limits for each file based on their size relative to the total size.
    
    Args:
        file_paths: List of file paths
        total_token_limit: Total token budget for all files
        min_tokens_per_file: Minimum tokens to allocate per file
        max_tokens_per_file: Maximum tokens to allocate per file
        verbosity: Verbosity level
        
    Returns:
        Dictionary mapping file paths to their allocated token limits
    """
    if verbosity >= VERBOSITY_NORMAL:
        logger.info(f"Calculating token limits with total budget of {total_token_limit} tokens")
    
    # Get file sizes
    file_sizes = {}
    total_size = 0
    
    for file_path in file_paths:
        size = get_file_size(file_path)
        file_sizes[file_path] = size
        total_size += size
    
    if verbosity >= VERBOSITY_VERBOSE:
        logger.info(f"Total size of all files: {total_size} bytes")
    
    # Calculate initial token allocation based on file size proportion
    token_limits = {}
    
    if total_size == 0:
        # If total size is 0 (empty files), distribute tokens equally
        equal_share = max(min_tokens_per_file, total_token_limit // len(file_paths))
        for file_path in file_paths:
            token_limits[file_path] = min(equal_share, max_tokens_per_file)
    else:
        # Allocate tokens proportionally to file size
        for file_path, size in file_sizes.items():
            # Calculate proportion of total size
            proportion = size / total_size
            
            # Allocate tokens proportionally, respecting min and max limits
            allocated_tokens = max(min_tokens_per_file, int(proportion * total_token_limit))
            allocated_tokens = min(allocated_tokens, max_tokens_per_file)
            
            token_limits[file_path] = allocated_tokens
    
    # Check if we've overallocated and adjust if needed
    total_allocated = sum(token_limits.values())
    if total_allocated > total_token_limit:
        # Scale down all allocations proportionally
        scale_factor = total_token_limit / total_allocated
        for file_path in token_limits:
            token_limits[file_path] = max(min_tokens_per_file, int(token_limits[file_path] * scale_factor))
    
    if verbosity >= VERBOSITY_VERBOSE:
        logger.info(f"Allocated a total of {sum(token_limits.values())} tokens across {len(file_paths)} files")
        
        if verbosity >= VERBOSITY_DEBUG:
            # Log token allocation for each file
            logger.debug("Token allocation per file:")
            for file_path, tokens in token_limits.items():
                rel_path = os.path.basename(file_path)
                size = file_sizes[file_path]
                logger.debug(f"  - {rel_path}: {tokens} tokens for {size} bytes")
    
    return token_limits

def is_text_file(file_path: str, verbosity: int) -> bool:
    """
    Determine if a file is text-based using multiple methods.
    """
    if verbosity >= VERBOSITY_DEBUG:
        logger.debug(f"Checking if {file_path} is a text file")
    
    # Check by extension first (faster)
    ext = os.path.splitext(file_path)[1].lower()
    if ext in DEFAULT_TEXT_EXTENSIONS:
        if verbosity >= VERBOSITY_DEBUG:
            logger.debug(f"File {file_path} identified as text file by extension {ext}")
        return True
    
    # Fallback to mimetype checking
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('text/'):
        if verbosity >= VERBOSITY_DEBUG:
            logger.debug(f"File {file_path} identified as text file by MIME type {mime_type}")
        return True
    
    # Last resort: try to read as text
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(1024)  # Try to read a sample
        if verbosity >= VERBOSITY_DEBUG:
            logger.debug(f"File {file_path} identified as text file by successful text reading")
        return True
    except UnicodeDecodeError:
        if verbosity >= VERBOSITY_DEBUG:
            logger.debug(f"File {file_path} identified as binary file (failed text reading)")
        return False

def parse_gitignore(gitignore_path: str, verbosity: int) -> List[str]:
    """
    Parse a .gitignore file and return patterns.
    """
    patterns = []
    if not os.path.exists(gitignore_path):
        if verbosity >= VERBOSITY_NORMAL:
            logger.info(f"No .gitignore found at {gitignore_path}")
        return patterns
    
    if verbosity >= VERBOSITY_NORMAL:
        logger.info(f"Parsing .gitignore from {gitignore_path}")
    
    with open(gitignore_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Add the pattern
            patterns.append(line)
    
    if verbosity >= VERBOSITY_VERBOSE:
        logger.info(f"Found {len(patterns)} ignore patterns")
        if verbosity >= VERBOSITY_DEBUG:
            logger.debug(f"Ignore patterns: {patterns}")
    
    return patterns

def is_ignored(file_path: str, root_path: str, ignore_patterns: List[str], verbosity: int) -> bool:
    """
    Check if a file or directory should be ignored based on gitignore patterns.
    Implements the core functionality of gitignore pattern matching.
    """
    if not ignore_patterns:
        return False
    
    # Convert to relative path from the root
    rel_path = os.path.relpath(file_path, root_path)
    # Replace backslashes with forward slashes for consistent pattern matching
    rel_path = rel_path.replace('\\', '/')
    
    # For directories, we should match patterns with and without trailing slash
    is_dir = os.path.isdir(file_path)
    
    if verbosity >= VERBOSITY_DEBUG:
        logger.debug(f"Checking if {'directory' if is_dir else 'file'} {rel_path} should be ignored")
    
    # Keep track of all matches for accurate negation processing
    matches = []
    
    for pattern in ignore_patterns:
        # Skip empty patterns
        if not pattern:
            continue
            
        # Handle negation (patterns that start with !)
        negated = pattern.startswith('!')
        if negated:
            pattern = pattern[1:]
        
        # Handle directory-specific patterns (ending with /)
        is_dir_pattern = pattern.endswith('/')
        if is_dir_pattern:
            pattern = pattern[:-1]
            # Directory-specific patterns only match directories
            if not is_dir:
                continue
        
        # Handle patterns that start with / which means relative to the repository root
        if pattern.startswith('/'):
            pattern = pattern[1:]
            # Match only if the pattern matches from the start
            if not rel_path.startswith(pattern):
                continue
        
        # Support ** for directory matching (match zero or more directories)
        if '**' in pattern:
            # Convert ** to a regex that matches any number of directories
            regex_pattern = pattern.replace('**', '.*')
            # Use more flexible regex matching for ** patterns
            match = re.search(regex_pattern, rel_path) is not None
        else:
            # For standard patterns, use fnmatch which supports * and ? wildcards
            # These patterns can match anywhere in the path
            match = fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(rel_path, f"*/{pattern}")
            # Also match directories with patterns that don't have a trailing slash
            if is_dir and not is_dir_pattern:
                match = match or fnmatch.fnmatch(f"{rel_path}/", pattern) or fnmatch.fnmatch(f"{rel_path}/", f"*/{pattern}")
        
        if verbosity >= VERBOSITY_DEBUG:
            logger.debug(f"Pattern '{pattern}' {'matches' if match else 'does not match'} {rel_path}")
        
        if match:
            matches.append((pattern, negated))
    
    # Process matches with proper negation handling
    # Later rules override earlier ones
    if matches:
        # Get the last match which takes precedence
        last_pattern, last_negated = matches[-1]
        if verbosity >= VERBOSITY_VERBOSE:
            if not last_negated:
                logger.info(f"Ignoring {'directory' if is_dir else 'file'} {rel_path} due to pattern '{last_pattern}'")
        return not last_negated
    
    return False

def get_git_file_tree(directory: str, verbosity: int) -> str:
    """
    Generate a tree-style representation of the directory structure using Git commands.
    This ensures gitignore rules are followed exactly as Git would.
    """
    if verbosity >= VERBOSITY_NORMAL:
        logger.info(f"Generating file tree for {directory} using Git")
    
    # Check if directory is a git repository
    import subprocess
    
    try:
        # Save current directory to return to it later
        original_dir = os.getcwd()
        os.chdir(directory)
        
        # Check if this is a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            if verbosity >= VERBOSITY_NORMAL:
                logger.info("Not a git repository. Using manual tree generation.")
            os.chdir(original_dir)
            return get_manual_file_tree(directory, [], verbosity)
        
        # Get all tracked files using git ls-files
        if verbosity >= VERBOSITY_VERBOSE:
            logger.info("Getting tracked files from git")
        
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=True
        )
        
        git_files = [line for line in result.stdout.split('\n') if line]
        
        # Return to original directory
        os.chdir(original_dir)
        
        if verbosity >= VERBOSITY_VERBOSE:
            logger.info(f"Found {len(git_files)} tracked files from git")
        
        # Build the tree structure
        base_dir = os.path.basename(os.path.normpath(directory))
        tree = {base_dir: {}}
        
        for file_path in sorted(git_files):
            # Normalize path separators for Windows
            file_path = file_path.replace('/', os.sep)
            
            # Split into path components
            parts = file_path.split(os.sep)
            
            # Build tree structure
            current = tree[base_dir]
            for i, part in enumerate(parts):
                if i == len(parts) - 1:  # File (leaf node)
                    current[part] = None
                else:  # Directory
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        
        # Convert tree structure to string representation
        output = []
        output.append(f"{base_dir}/")
        
        def _build_tree_lines(node, prefix="", is_last=False, is_root=False):
            items = list(node.items())
            
            for i, (name, children) in enumerate(items):
                is_last_item = (i == len(items) - 1)
                
                if is_root:
                    new_prefix = ""
                    tree_prefix = ""
                else:
                    tree_prefix = "└── " if is_last_item else "├── "
                    new_prefix = prefix + ("    " if is_last_item else "│   ")
                
                if children is None:  # File
                    output.append(f"{prefix}{tree_prefix}{name}")
                else:  # Directory
                    output.append(f"{prefix}{tree_prefix}{name}/")
                    _build_tree_lines(children, new_prefix, is_last_item)
        
        _build_tree_lines(tree, is_root=True)
        
        return '\n'.join(output)
        
    except Exception as e:
        logger.error(f"Error generating git file tree: {str(e)}")
        if verbosity >= VERBOSITY_NORMAL:
            logger.info("Falling back to manual tree generation")
        return get_manual_file_tree(directory, [], verbosity)


def get_manual_file_tree(directory: str, ignore_patterns: List[str], verbosity: int) -> str:
    """
    Generate a tree-style representation of the directory structure.
    Properly excludes files and directories based on gitignore patterns.
    Used as a fallback when git is not available.
    """
    if verbosity >= VERBOSITY_NORMAL:
        logger.info(f"Generating file tree manually for {directory}")
    
    output = []
    base_dir = os.path.basename(os.path.normpath(directory))
    
    output.append(f"{base_dir}/")
    
    included_files_count = 0
    excluded_files_count = 0
    included_dirs_count = 0
    excluded_dirs_count = 0
    
    for root, dirs, files in os.walk(directory):
        # Calculate the level for proper indentation
        level = root.replace(directory, '').count(os.sep)
        indent = '│   ' * level
        
        # Check if the current directory itself should be ignored
        # If it is, we shouldn't process anything inside it
        if root != directory and is_ignored(root, directory, ignore_patterns, verbosity):
            if verbosity >= VERBOSITY_DEBUG:
                logger.debug(f"Skipping directory tree for ignored directory: {root}")
            # Remove all dirs to prevent further traversal
            dirs[:] = []
            continue
        
        # Filter out ignored directories to prevent traversal
        filtered_dirs = []
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not is_ignored(dir_path, directory, ignore_patterns, verbosity):
                filtered_dirs.append(d)
                included_dirs_count += 1
            else:
                excluded_dirs_count += 1
                if verbosity >= VERBOSITY_DEBUG:
                    logger.debug(f"Excluding directory from tree: {dir_path}")
        
        # Update dirs in-place to control os.walk traversal
        dirs[:] = filtered_dirs
        
        # Filter files that shouldn't be shown in the tree
        visible_files = []
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if not is_ignored(file_path, directory, ignore_patterns, verbosity):
                visible_files.append(file_name)
                included_files_count += 1
            else:
                excluded_files_count += 1
                if verbosity >= VERBOSITY_DEBUG:
                    logger.debug(f"Excluding file from tree: {file_path}")
        
        # Add subdirectories to the tree output
        for i, dir_name in enumerate(sorted(filtered_dirs)):
            is_last_dir = (i == len(filtered_dirs) - 1)
            if is_last_dir and not visible_files:
                output.append(f"{indent}└── {dir_name}/")
            else:
                output.append(f"{indent}├── {dir_name}/")
        
        # Add files to the tree output
        for i, file_name in enumerate(sorted(visible_files)):
            is_last_file = (i == len(visible_files) - 1)
            if is_last_file:
                output.append(f"{indent}└── {file_name}")
            else:
                output.append(f"{indent}├── {file_name}")
    
    if verbosity >= VERBOSITY_VERBOSE:
        logger.info(f"File tree includes {included_files_count} files and {included_dirs_count} directories")
        logger.info(f"File tree excludes {excluded_files_count} files and {excluded_dirs_count} directories")
    
    return '\n'.join(output)

def get_git_files(directory: str, verbosity: int) -> List[str]:
    """
    Get a list of all files tracked by Git (respecting .gitignore rules)
    Returns a list of files with a 'git:' prefix to distinguish from manually discovered files.
    Returns an empty list if Git is not available or the directory is not a Git repository.
    """
    try:
        import subprocess
        
        # Save current directory to return to it later
        original_dir = os.getcwd()
        os.chdir(directory)
        
        # Check if this is a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            if verbosity >= VERBOSITY_NORMAL:
                logger.info("Not a git repository. Using manual file discovery.")
            os.chdir(original_dir)
            return []
        
        # Get all tracked files using git ls-files
        if verbosity >= VERBOSITY_VERBOSE:
            logger.info("Getting tracked files from git")
        
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Add 'git:' prefix to distinguish from manually discovered files
        git_files = [f"git:{line}" for line in result.stdout.split('\n') if line]
        
        # Return to original directory
        os.chdir(original_dir)
        
        if verbosity >= VERBOSITY_VERBOSE:
            logger.info(f"Found {len(git_files)} tracked files from git")
            
        return git_files
    
    except Exception as e:
        if verbosity >= VERBOSITY_NORMAL:
            logger.info(f"Error getting git files: {str(e)}")
        
        # Return to original directory if changed
        try:
            os.chdir(original_dir)
        except:
            pass
            
        return []

def compress_file_content(file_path: str, api_key: str, model: str, token_limit: int, verbosity: int) -> str:
    """
    Send file content to OpenAI API for ultra-compression.
    Instructs the AI to compress the file content to fit within the specified token limit.
    """
    rel_path = os.path.basename(file_path)
    
    # Read the file content
    try:
        if verbosity >= VERBOSITY_VERBOSE:
            logger.info(f"Reading content from {rel_path}")
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        content_size = len(content)
        
        if verbosity >= VERBOSITY_VERBOSE:
            logger.info(f"File size: {content_size} characters, target token limit: {token_limit}")
        
    except Exception as e:
        logger.error(f"Error reading file {rel_path}: {str(e)}")
        return f"COMPRESSION ERROR: Unable to read file - {str(e)}"
    
    # Setup OpenAI client
    client = openai.OpenAI(api_key=api_key)
    
    # Prepare the prompt for compression
    prompt = f"""
    I have a source code file that I need to ultra-compress. I need you to create the most compact representation possible while ensuring Claude 3.7 Sonnet can reconstruct the original code with high fidelity.

    Focus on retaining:
    1. Core functionality and logic
    2. Essential comments (especially those explaining complex parts)
    3. Critical structure and organization
    4. Any unique or non-standard features

    Your compression should be optimized for later reconstruction by an LLM, not human readability.
    Use any representation format that will maximize compression while minimizing information loss.

    Here is the original code:
    ```
    {content}
    ```

    Provide ONLY the ultra-compressed representation, nothing else.
    """
    
    import time
    start_time = time.time()
    
    if verbosity >= VERBOSITY_NORMAL:
        logger.info(f"Sending {rel_path} to OpenAI API for compression (target token size: {token_limit}) using model {model}")
    
    try:
        # Call the API - for o3-mini, which doesn't support max_tokens or temperature
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert code compressor. Your task is to create ultra-compressed representations of source code that can be later reconstructed."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        # Extract and return the compressed content
        compressed = response.choices[0].message.content.strip()
        
        elapsed_time = time.time() - start_time
        compressed_size = len(compressed)
        
        if verbosity >= VERBOSITY_NORMAL:
            logger.info(f"Compression complete in {elapsed_time:.2f} seconds")
        
        if verbosity >= VERBOSITY_VERBOSE:
            compression_ratio = (1 - (compressed_size / content_size)) * 100 if content_size > 0 else 0
            logger.info(f"Original size: {content_size}, Compressed size: {compressed_size}, Compression ratio: {compression_ratio:.2f}%")
        
        return compressed
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error compressing {rel_path} after {elapsed_time:.2f} seconds: {str(e)}")
        return f"COMPRESSION ERROR: {str(e)}"

def get_file_content(file_path: str, verbosity: int) -> str:
    """
    Get the raw content of a file without compression.
    """
    rel_path = os.path.basename(file_path)
    
    try:
        if verbosity >= VERBOSITY_VERBOSE:
            logger.info(f"Reading content from {rel_path}")
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        content_size = len(content)
        
        if verbosity >= VERBOSITY_VERBOSE:
            logger.info(f"File size: {content_size} characters")
        
        return content
        
    except Exception as e:
        logger.error(f"Error reading file {rel_path}: {str(e)}")
        return f"ERROR: Unable to read file - {str(e)}"

def process_directory(
    directory: str, 
    output_file: str, 
    api_key: str,
    model: str,
    custom_gitignore: Optional[str] = None,
    preview_ignored: bool = False,
    verbosity: int = VERBOSITY_NORMAL,
    use_git: bool = True,
    total_token_limit: int = DEFAULT_TOTAL_TOKEN_LIMIT,
    min_tokens_per_file: int = DEFAULT_MIN_TOKENS_PER_FILE,
    max_tokens_per_file: int = DEFAULT_MAX_TOKENS_PER_FILE,
    no_compression: bool = False
) -> None:
    """
    Process a directory, compress its files, and write the results.
    
    If no_compression is True, files are combined without compression.
    """
    # Set verbosity level
    set_verbosity(verbosity)
    
    # Get current timestamp for output file
    timestamp = datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S')
    
    import time
    start_time = time.time()
    
    if verbosity >= VERBOSITY_NORMAL:
        if no_compression:
            logger.info(f"Starting codebase processing with verbosity level {verbosity} (no compression)")
        else:
            logger.info(f"Starting codebase compression with verbosity level {verbosity}")
            logger.info(f"Token budget: {total_token_limit} tokens total, {min_tokens_per_file}-{max_tokens_per_file} per file")
    
    # Validate and normalize input directory
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        sys.exit(1)
    
    if verbosity >= VERBOSITY_NORMAL:
        logger.info(f"Processing directory: {directory}")
    
    # Handle .gitignore
    ignore_patterns = []
    gitignore_path = os.path.join(directory, '.gitignore')
    
    if os.path.exists(gitignore_path):
        if verbosity >= VERBOSITY_NORMAL:
            logger.info(f"Using .gitignore from: {gitignore_path}")
        ignore_patterns = parse_gitignore(gitignore_path, verbosity)
    elif custom_gitignore:
        if os.path.exists(custom_gitignore):
            if verbosity >= VERBOSITY_NORMAL:
                logger.info(f"Using custom .gitignore from: {custom_gitignore}")
            ignore_patterns = parse_gitignore(custom_gitignore, verbosity)
        else:
            logger.error(f"Custom .gitignore not found: {custom_gitignore}")
            sys.exit(1)
    else:
        # Ask user what to do if no .gitignore found
        if verbosity >= VERBOSITY_NORMAL:
            logger.info("No .gitignore found in directory")
        
        choice = input(
            "No .gitignore found. What would you like to do?\n"
            "1. Provide a custom .gitignore path\n"
            "2. Proceed without ignoring any files\n"
            "3. Cancel operation\n"
            "Enter choice (1-3): "
        )
        
        if choice == "1":
            custom_path = input("Enter path to custom .gitignore: ").strip()
            if os.path.exists(custom_path):
                ignore_patterns = parse_gitignore(custom_path, verbosity)
            else:
                logger.error(f"Custom .gitignore not found: {custom_path}")
                sys.exit(1)
        elif choice == "2":
            if verbosity >= VERBOSITY_NORMAL:
                logger.info("Proceeding without ignoring any files")
        elif choice == "3":
            if verbosity >= VERBOSITY_NORMAL:
                logger.info("Operation cancelled by user")
            sys.exit(0)
        else:
            logger.error("Invalid choice")
            sys.exit(1)
    
    # Find all files - if using Git, we'll get files differently
    if verbosity >= VERBOSITY_NORMAL:
        logger.info("Scanning directory for files...")
    
    # Try to generate tree using Git if requested
    if use_git:
        file_tree = get_git_file_tree(directory, verbosity)
        # If using Git, also try to get files using Git
        all_files = get_git_files(directory, verbosity)
        if all_files:
            if verbosity >= VERBOSITY_VERBOSE:
                logger.info(f"Using Git to identify {len(all_files)} files")
        else:
            # Fall back to manual file scanning
            all_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
            
            if verbosity >= VERBOSITY_VERBOSE:
                logger.info(f"Found {len(all_files)} total files in directory through manual scanning")
    else:
        # Use manual tree and file discovery
        file_tree = get_manual_file_tree(directory, ignore_patterns, verbosity)
        all_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        
        if verbosity >= VERBOSITY_VERBOSE:
            logger.info(f"Found {len(all_files)} total files in directory through manual scanning")
    
    # Filter files based on ignore patterns and file type
    if verbosity >= VERBOSITY_NORMAL:
        logger.info("Filtering files based on ignore patterns and file types...")
    
    included_files = []
    ignored_files = []
    binary_files = []
    
    for file_path in all_files:
        # If we got files from Git, they're already filtered by gitignore
        # Just need to check if they're text files
        if use_git and all_files and isinstance(all_files[0], str) and all_files[0].startswith('git:'):
            clean_path = file_path[4:]  # Remove 'git:' prefix
            full_path = os.path.join(directory, clean_path)
            if is_text_file(full_path, verbosity):
                included_files.append(full_path)
                if verbosity >= VERBOSITY_DEBUG:
                    logger.debug(f"Including file: {clean_path}")
            else:
                binary_files.append(full_path)
                if verbosity >= VERBOSITY_DEBUG:
                    logger.debug(f"Ignoring file (binary): {clean_path}")
        else:
            # Normal path - check both gitignore and if it's a text file
            if is_ignored(file_path, directory, ignore_patterns, verbosity):
                ignored_files.append(file_path)
                if verbosity >= VERBOSITY_DEBUG:
                    logger.debug(f"Ignoring file (matched ignore pattern): {os.path.relpath(file_path, directory)}")
            elif is_text_file(file_path, verbosity):
                included_files.append(file_path)
                if verbosity >= VERBOSITY_DEBUG:
                    logger.debug(f"Including file: {os.path.relpath(file_path, directory)}")
            else:
                binary_files.append(file_path)
                if verbosity >= VERBOSITY_DEBUG:
                    logger.debug(f"Ignoring file (binary): {os.path.relpath(file_path, directory)}")
    
    if verbosity >= VERBOSITY_VERBOSE:
        logger.info(f"Files summary: {len(included_files)} included, {len(ignored_files)} ignored by patterns, {len(binary_files)} binary")
    
    # Preview ignored files if requested
    if preview_ignored and (ignored_files or binary_files):
        logger.info("The following files will be ignored:")
        
        if verbosity >= VERBOSITY_VERBOSE:
            logger.info("Ignored by patterns:")
        
        for file in ignored_files:
            rel_path = os.path.relpath(file, directory)
            logger.info(f"  - {rel_path} (matched ignore pattern)")
        
        if verbosity >= VERBOSITY_VERBOSE:
            logger.info("Binary files:")
            
            for file in binary_files:
                rel_path = os.path.relpath(file, directory)
                logger.info(f"  - {rel_path} (binary file)")
        
        confirm = input("Continue with these exclusions? (y/n): ").lower()
        if confirm != 'y':
            logger.info("Operation cancelled by user")
            sys.exit(0)
    
    # Calculate total size statistics
    total_size = sum(get_file_size(file_path) for file_path in included_files)
    if verbosity >= VERBOSITY_NORMAL:
        logger.info(f"Total size of included files: {total_size:,} bytes")
    
    # Calculate token limits if doing compression
    token_limits = {}
    if not no_compression:
        token_limits = calculate_token_limits(
            included_files, 
            total_token_limit,
            min_tokens_per_file,
            max_tokens_per_file,
            verbosity
        )
    
    # Process included files
    if verbosity >= VERBOSITY_NORMAL:
        logger.info(f"Writing output to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # Write a header at the top
        if no_compression:
            out_f.write(f"# CODEBASE FILE CONTENTS\n")
            out_f.write(f"# Generated on: {timestamp} EST\n\n")
            out_f.write("This document contains the combined contents of source code files from the codebase.\n")
            out_f.write("Each file is presented with its original contents, uncompressed and unmodified.\n\n")
        else:
            out_f.write(f"# ULTRA-COMPRESSED CODEBASE\n")
            out_f.write(f"# Generated on: {timestamp} EST\n\n")
            out_f.write("# IMPORTANT NOTICE TO AI MODEL\n\n")
            out_f.write("This document contains an ultra-compressed representation of source code, not the original code itself. ")
            out_f.write("The content has been specifically compressed to provide context while conserving tokens. ")
            out_f.write("When working with this data, please understand that you're viewing a densely packed abstraction designed for AI consumption. ")
            out_f.write("The original code syntax, formatting, and some non-critical elements may have been altered during compression. ")
            out_f.write("Use this as a contextual reference to understand the codebase's structure and functionality, ")
            out_f.write("but be aware that direct copying of this compressed representation would not produce working code.\n\n")
        
        # Write the file tree
        out_f.write("# Codebase Structure\n\n")
        out_f.write("```\n")
        out_f.write(file_tree)
        out_f.write("\n```\n\n")
        
        if no_compression:
            out_f.write("# File Contents\n\n")
        else:
            out_f.write("# Ultra-Compressed File Contents\n\n")
        
        total_files = len(included_files)
        
        if verbosity >= VERBOSITY_NORMAL:
            if no_compression:
                logger.info(f"Beginning processing of {total_files} files (no compression)...")
            else:
                logger.info(f"Beginning compression of {total_files} files...")
        
        for i, file_path in enumerate(included_files, 1):
            rel_path = os.path.relpath(file_path, directory)
            file_size = get_file_size(file_path)
            
            if verbosity >= VERBOSITY_NORMAL:
                if no_compression:
                    logger.info(f"Processing file {i}/{total_files}: {rel_path} ({file_size:,} bytes)")
                else:
                    token_limit = token_limits[file_path]
                    logger.info(f"Processing file {i}/{total_files}: {rel_path} ({file_size:,} bytes, target {token_limit} tokens)")
            
            # Get file content - either compressed or raw
            if no_compression:
                file_content = get_file_content(file_path, verbosity)
                out_f.write(f"------------------------------ 📄 File: {rel_path} ({file_size:,} bytes) ------------------------------\n")
                out_f.write("```\n")
                out_f.write(file_content)
                out_f.write("\n```\n\n")
            else:
                token_limit = token_limits[file_path]
                compressed_content = compress_file_content(file_path, api_key, model, token_limit, verbosity)
                out_f.write(f"------------------------------ 📄 File: {rel_path} ({file_size:,} bytes, target {token_limit} tokens) ------------------------------\n")
                out_f.write(compressed_content)
                out_f.write("\n\n")
            
            if verbosity >= VERBOSITY_VERBOSE:
                logger.info(f"Completed file {i}/{total_files}: {rel_path}")
    
    total_time = time.time() - start_time
    
    if verbosity >= VERBOSITY_NORMAL:
        if no_compression:
            logger.info(f"Processed {len(included_files)} files ({total_size:,} bytes) in {total_time:.2f} seconds")
        else:
            logger.info(f"Compressed {len(included_files)} files ({total_size:,} bytes) in {total_time:.2f} seconds")
        logger.info(f"Output written to: {output_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a codebase - either compress it using AI or simply combine files")
    parser.add_argument("directory", help="Directory containing the codebase to process")
    parser.add_argument("--output", "-o", 
                       default=DEFAULT_OUTPUT_FILE, 
                       help=f"Output file path (default: timestamped file)")
    parser.add_argument("--model", "-m", 
                       default=DEFAULT_MODEL,
                       help=f"OpenAI model to use for compression (default: {DEFAULT_MODEL})")
    parser.add_argument("--gitignore", "-g",
                       help="Custom .gitignore file path to use instead of looking in the directory")
    parser.add_argument("--preview", "-p", 
                       action="store_true",
                       help="Preview ignored files before processing")
    parser.add_argument("--api-key", "-k",
                       help="OpenAI API key (will use OPENAI_API_KEY environment variable if not provided)")
    parser.add_argument("--verbosity", "-v", type=int, default=VERBOSITY_NORMAL, choices=range(4),
                       help="Verbosity level: 0=quiet, 1=normal, 2=verbose, 3=debug (default: 1)")
    parser.add_argument("--use-git", action="store_true", default=True,
                       help="Use Git to determine which files to include (respects .gitignore perfectly)")
    parser.add_argument("--no-git", action="store_false", dest="use_git",
                       help="Don't use Git commands even if Git is available")
    parser.add_argument("--token-limit", "-t", type=int, default=DEFAULT_TOTAL_TOKEN_LIMIT,
                       help=f"Total token limit for all compressed files (default: {DEFAULT_TOTAL_TOKEN_LIMIT})")
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS_PER_FILE,
                       help=f"Minimum tokens per file (default: {DEFAULT_MIN_TOKENS_PER_FILE})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS_PER_FILE,
                       help=f"Maximum tokens per file (default: {DEFAULT_MAX_TOKENS_PER_FILE})")
    parser.add_argument("--no-compression", "-n", action="store_true",
                       help="Combine files without compression (no AI usage)")
    
    args = parser.parse_args()
    
    # Get OpenAI API key if we're compressing
    api_key = None
    if not args.no_compression:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = input("Enter your OpenAI API key: ").strip()
            if not api_key:
                logger.error("No OpenAI API key provided")
                sys.exit(1)
    
    # Process the directory
    process_directory(
        directory=args.directory,
        output_file=args.output,
        api_key=api_key,
        model=args.model,
        custom_gitignore=args.gitignore,
        preview_ignored=args.preview,
        verbosity=args.verbosity,
        use_git=args.use_git,
        total_token_limit=args.token_limit,
        min_tokens_per_file=args.min_tokens,
        max_tokens_per_file=args.max_tokens,
        no_compression=args.no_compression
    )

if __name__ == "__main__":
    main()