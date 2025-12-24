#!/usr/bin/env python3
import os

# Path to the buggy file
file_path = os.path.join(
    'venv',
    'Lib',
    'site-packages',
    'crewai',
    'events',
    'types',
    'system_events.py'
)

# Read the file
with open(file_path, 'r') as f:
    content = f.read()

# Find and replace the buggy line
old_line = 'SIGHUP = signal.SIGHUP'
new_line = 'SIGHUP = getattr(signal, "SIGHUP", None)'

if old_line in content:
    content = content.replace(old_line, new_line)
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ SUCCESS! Fixed the SIGHUP bug!")
else:
    print("❌ Could not find buggy line")
