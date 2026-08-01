import re
from pathlib import Path

def bundle_latex():
    base_dir = Path(__file__).resolve().parent
    main_path = base_dir / "thesis.tex"
    output_path = base_dir / "thesis_bundled.tex"
    
    if not main_path.exists():
        print(f"Error: {main_path} not found.")
        return
        
    print(f"Reading main LaTeX driver: {main_path}")
    with open(main_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Regex to find \input{chapters/filename}
    pattern = re.compile(r'\\input\{chapters/([a-zA-Z0-9_-]+)\}')
    
    def replace_input(match):
        chapter_name = match.group(1)
        chapter_path = base_dir / "chapters" / f"{chapter_name}.tex"
        if chapter_path.exists():
            print(f"  Inlining chapter: {chapter_name}")
            with open(chapter_path, "r", encoding="utf-8") as cf:
                return f"\n% --- BEGIN INPUT: {chapter_name}.tex ---\n" + cf.read() + f"\n% --- END INPUT: {chapter_name}.tex ---\n"
        else:
            print(f"  Warning: Chapter file not found: {chapter_path}")
            return match.group(0)
            
    bundled_content = pattern.sub(replace_input, content)
    
    # Save bundled file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(bundled_content)
        
    print(f"Successfully generated bundled LaTeX document: {output_path}")

if __name__ == "__main__":
    bundle_latex()
