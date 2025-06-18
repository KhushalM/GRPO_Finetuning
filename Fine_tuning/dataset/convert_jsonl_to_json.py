#!/usr/bin/env python3
"""
Multi-line JSON objects to JSON array converter
Handles JSON objects that span multiple lines
"""
import json
import sys
import re

def convert_multiline_json_to_array(input_file, output_file=None):
    if output_file is None:
        output_file = input_file.replace('.json', '_fixed.json')
    
    print(f"Converting {input_file} from multi-line JSON objects to JSON array...")
    
    # Read the entire file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Split by }\n{ pattern to separate individual JSON objects
    # This regex looks for } followed by optional whitespace, newline, optional whitespace, then {
    json_objects = re.split(r'}\s*\n\s*{', content)
    
    data = []
    valid_count = 0
    error_count = 0
    
    for i, obj_str in enumerate(json_objects):
        # Reconstruct the JSON object by adding back the braces
        if i == 0:
            # First object - add closing brace
            if not obj_str.strip().endswith('}'):
                obj_str = obj_str + '}'
        elif i == len(json_objects) - 1:
            # Last object - add opening brace
            if not obj_str.strip().startswith('{'):
                obj_str = '{' + obj_str
        else:
            # Middle objects - add both braces
            if not obj_str.strip().startswith('{'):
                obj_str = '{' + obj_str
            if not obj_str.strip().endswith('}'):
                obj_str = obj_str + '}'
        
        try:
            obj = json.loads(obj_str.strip())
            if isinstance(obj, dict) and 'prompt' in obj and 'answer' in obj:
                data.append(obj)
                valid_count += 1
            else:
                print(f"Warning: Object {i+1} missing required keys")
                error_count += 1
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse object {i+1}: {e}")
            error_count += 1
    
    # Write as JSON array
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully converted {valid_count} entries")
    if error_count > 0:
        print(f"⚠️  Skipped {error_count} invalid objects")
    print(f"💾 Saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python multiline_converter.py <input_file>")
        sys.exit(1)
    
    convert_multiline_json_to_array(sys.argv[1])