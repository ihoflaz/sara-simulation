# Quick GUI Fix Script
# Fixes the broken main_window.py file

import os
import shutil

def fix_main_window():
    """Fix the broken main_window.py file"""
    main_window_path = 'gui/main_window.py'
    
    print("Fixing main_window.py file...")
    
    # Read the broken file
    with open(main_window_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the broken try-except structure around line 327
    # The issue is that the jammer detection code got moved outside the try block
    
    # Find the problematic section and fix indentation
    lines = content.split('\n')
    fixed_lines = []
    in_try_block = False
    try_depth = 0
    
    for i, line in enumerate(lines):
        # Track try blocks
        if 'try:' in line:
            in_try_block = True
            try_depth = len(line) - len(line.lstrip())
            
        # Fix the specific problematic section around jammer detection
        if 'jammer_detected = channel_state.get' in line and in_try_block:
            # This line should be indented to match the try block
            if not line.startswith('                '):  # Should have 16 spaces
                line = '                ' + line.strip()
        
        # Fix other lines that should be in the try block
        if any(keyword in line for keyword in [
            'jammer_type =', 'self.jammer_detection_updated.emit',
            'channel_quality =', 'original_errors =', 'coding_scheme =',
            'correction_rate =', 'coding_effectiveness =', 'corrected_errors =',
            'remaining_errors =', 'security_message =', 'self.security_event_logged.emit',
            'countermeasure_message =', 'all_band_snrs =', 'for band_idx in range',
            'self.subband_snr_updated.emit', 'self.security_status_updated.emit'
        ]) and in_try_block and not line.strip().startswith('#'):
            if not line.startswith('                '):  # Should have 16 spaces
                line = '                ' + line.strip()
        
        # Handle except blocks
        if 'except Exception as e:' in line:
            in_try_block = False
            
        fixed_lines.append(line)
    
    # Write the fixed content
    fixed_content = '\n'.join(fixed_lines)
    
    # Backup the broken file
    backup_path = 'gui/main_window_broken.py'
    shutil.copy2(main_window_path, backup_path)
    print(f"Backed up broken file to: {backup_path}")
    
    # Write the fixed version
    with open(main_window_path, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Fixed main_window.py file")
    
    # Test the fix
    try:
        import ast
        with open(main_window_path, 'r', encoding='utf-8') as f:
            test_content = f.read()
        ast.parse(test_content)
        print("✅ Syntax check passed!")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error still present: {e}")
        return False

if __name__ == "__main__":
    fix_main_window()
