"""
Streamlit Community Cloud Root Application Entrypoint.
Redirects execution cleanly to Week 4/dashboard.py.
"""
from pathlib import Path
import sys

# Setting root path in sys.path.
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Executing the Week 4 dashboard module.
dashboard_script = project_root / "Week 4" / "dashboard.py"
with open(dashboard_script, "r", encoding="utf-8") as f:
    code = f.read()

# Executing dashboard script in current global context.
exec(code, globals())
