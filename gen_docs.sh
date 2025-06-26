echo "Make sure to be in the correct Python environment"
read -p "Press Enter to continue..."
pip install --upgrade -r requirements.txt > /dev/null 
pip install --upgrade -r requirements-dev.txt > /dev/null
lazydocs \
    --output-path="./docs/api-docs" \
    --overview-file="README.md" \
    --src-base-url="https://github.com/leoandeol/cods/blob/main/" \
    cods/
mkdocs build
