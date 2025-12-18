import json
import pandas as pd
from pathlib import Path

# IMPORT CORRETTO
from dovalens.utils import safe_json


HTML_TEMPLATE = """
<html>
<head>
<title>DovaLens Automated Report</title>
<style>
body {{
    background:#0D0D0D; color:white; font-family: Arial;
}}
h1, h2 {{
    color:#5AB0FF;
}}
pre {{
    background:#111; padding:10px; border-radius:8px;
}}
</style>
</head>
<body>
<h1>DovaLens Automated Report</h1>

<h2>Dataset Preview</h2>
<pre>{preview}</pre>

<h2>Analysis Output</h2>
<pre>{analysis}</pre>

</body>
</html>
"""


def generate_report(df: pd.DataFrame, analysis: dict, output_path="report.html"):
    preview = df.head(10).to_string()

    # JSON encoding using our safe encoder
    analysis_json = json.dumps(analysis, indent=4, default=safe_json)

    html = HTML_TEMPLATE.format(
        preview=preview,
        analysis=analysis_json
    )

    Path(output_path).write_text(html, encoding="utf-8")
    return output_path
