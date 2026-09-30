#!/bin/sh
set -eu

cd "$(dirname "$0")"

if command -v conda >/dev/null 2>&1 && conda env list | grep -qE '(^|[[:space:]])xgboostbest-mac[[:space:]]'; then
  exec conda run --no-capture-output -n xgboostbest-mac python app.py
fi

if command -v python3 >/dev/null 2>&1 && python3 -c 'import flask, sklearn, shap' >/dev/null 2>&1; then
  exec python3 app.py
fi

echo "Required Python environment was not found."
echo "Create it with: conda env create -f environment.yml"
echo "Then run: conda run --no-capture-output -n pregnancy-appendicitis-web python app.py"
exit 1
