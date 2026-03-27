"""
Quick test script for JupyterSessionTool functionality.
Tests: kernel lifecycle, variable persistence, HTML/SVG capture, image capture.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.tools import JupyterSessionTool
import json

def test_kernel():
    print("=" * 60)
    print("JUPYTER KERNEL FUNCTIONALITY TEST")
    print("=" * 60)

    tool = JupyterSessionTool(output_dir="./results/test_charts")
    results = {"passed": 0, "failed": 0, "tests": []}

    def check(name, condition, details=""):
        status = "PASS" if condition else "FAIL"
        results["passed" if condition else "failed"] += 1
        results["tests"].append({"name": name, "status": status, "details": details})
        print(f"[{status}] {name}" + (f" - {details}" if details else ""))

    try:
        # Test 1: Kernel Start
        print("\n1. Testing kernel start...")
        tool.start_kernel()
        check("Kernel Start", tool._km is not None and tool._kc is not None)

        # Test 2: Variable Persistence
        print("\n2. Testing variable persistence...")
        result1 = tool._run("x = 42; print(f'Set x = {x}')", agent_name="test")
        result2 = tool._run("y = x * 2; print(f'y = x * 2 = {y}')", agent_name="test")
        data2 = json.loads(result2)
        check("Variable Persistence", "84" in data2["stdout"], data2["stdout"].strip())

        # Test 3: Pre-loaded Libraries
        print("\n3. Testing pre-loaded libraries...")
        result3 = tool._run("print(f'numpy version: {np.__version__}')", agent_name="test")
        data3 = json.loads(result3)
        check("Pre-loaded NumPy", data3["success"] and "numpy version:" in data3["stdout"])

        # Test 4: DataFrame HTML Output
        print("\n4. Testing DataFrame HTML capture...")
        df_code = """
import pandas as pd
df_test = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [85.5, 90.2, 78.8]
})
df_test  # Display DataFrame (triggers HTML output)
"""
        result4 = tool._run(df_code, agent_name="test")
        data4 = json.loads(result4)
        has_html = bool(data4.get("html", "").strip())
        check("DataFrame HTML Output", has_html, f"HTML length: {len(data4.get('html', ''))}")

        # Test 5: Matplotlib PNG Image
        print("\n5. Testing matplotlib image capture...")
        plot_code = """
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'bo-')
plt.title('Test Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.show()
"""
        result5 = tool._run(plot_code, agent_name="test")
        data5 = json.loads(result5)
        has_image = len(data5.get("images", [])) > 0
        check("Matplotlib Image Capture", has_image, f"Images: {data5.get('images', [])}")

        # Test 6: Error Capture (Clean ANSI)
        print("\n6. Testing error capture (clean ANSI)...")
        result6 = tool._run("undefined_variable_xyz", agent_name="test")
        data6 = json.loads(result6)
        has_error = "NameError" in data6.get("stderr", "")
        no_ansi = "\x1b[" not in data6.get("stderr", "")
        check("Error Capture", has_error and not data6["success"],
              f"NameError captured, ANSI stripped: {no_ansi}")

        # Test 7: Cell Tracking
        print("\n7. Testing cell tracking...")
        cells = tool.get_cells()
        check("Cell Tracking", len(cells) >= 6, f"Total cells: {len(cells)}")

        # Test 8: HTML/SVG Fields in Cells
        print("\n8. Testing HTML/SVG fields in cell records...")
        html_cells = [c for c in cells if c.get("html")]
        check("HTML in Cell Records", len(html_cells) > 0,
              f"Cells with HTML: {len(html_cells)}")

        # Test 9: CRUD - Create Cell
        print("\n9. Testing CRUD operations...")
        new_cell_id = tool.create_cell("# New cell created via CRUD", agent_name="user")
        check("Create Cell", new_cell_id is not None, f"Cell ID: {new_cell_id}")

        # Test 10: CRUD - Rerun Cell
        rerun_result = tool.rerun_cell(new_cell_id)
        check("Rerun Cell", rerun_result is not None and "html" in rerun_result)

        # Test 11: Export Notebook
        print("\n10. Testing notebook export...")
        export_path = tool.export_notebook("./results/test_export.ipynb")
        check("Export Notebook", os.path.exists(export_path), f"Path: {export_path}")

    except Exception as e:
        check("Exception", False, str(e))

    finally:
        # Test: Kernel Shutdown
        print("\n11. Testing kernel shutdown...")
        tool.shutdown_kernel()
        check("Kernel Shutdown", tool._km is None and tool._kc is None)

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: {results['passed']} passed, {results['failed']} failed")
    print("=" * 60)

    return results["failed"] == 0

if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)
