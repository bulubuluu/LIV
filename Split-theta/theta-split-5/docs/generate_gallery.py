import argparse
import html
import os
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


def find_images(root: Path):
    groups = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel_path = path.relative_to(root)
        group = str(rel_path.parent)
        groups.setdefault(group, []).append(rel_path)
    return groups


def build_hierarchy(groups):
    hierarchy = {}
    for group, images in groups.items():
        parts = Path(group).parts if group != "." else ()
        level1 = parts[0] if len(parts) >= 1 else "root"
        level2 = parts[1] if len(parts) >= 2 else "misc"
        level3 = "/".join(parts[2:]) if len(parts) >= 3 else ""

        hierarchy.setdefault(level1, {})
        hierarchy[level1].setdefault(level2, [])
        hierarchy[level1][level2].append((level3, images))
    return hierarchy


def build_html(root: Path, title: str):
    groups = find_images(root)
    hierarchy = build_hierarchy(groups)
    sections = []
    nav_items = []

    level1_index = 0
    for level1, level2_map in hierarchy.items():
        level1_id = f"group-{level1_index}"
        level1_index += 1
        level1_title = html.escape(level1)
        sub_nav = []
        level2_sections = []

        level2_index = 0
        for level2, image_groups in level2_map.items():
            level2_id = f"{level1_id}-{level2_index}"
            level2_index += 1
            level2_title = html.escape(level2)
            sub_nav.append(f'<a class="subnav-link" href="#{level2_id}">{level2_title}</a>')

            folder_blocks = []
            for level3, images in image_groups:
                cards = []
                for rel_path in images:
                    rel_str = rel_path.as_posix()
                    label = html.escape(rel_path.name)
                    cards.append(
                        f"""
                        <a class="card" href="{html.escape(rel_str)}" target="_blank">
                          <img src="{html.escape(rel_str)}" alt="{label}" loading="lazy">
                          <div class="caption">{label}</div>
                        </a>
                        """
                    )

                folder_title = (
                    f'<div class="folder-label">{html.escape(level3)}</div>' if level3 else ""
                )
                folder_blocks.append(
                    f"""
                    <div class="folder-block">
                      {folder_title}
                      <div class="grid">
                        {''.join(cards)}
                      </div>
                    </div>
                    """
                )

            level2_sections.append(
                f"""
                <section class="subgroup" id="{level2_id}">
                  <h3>{level2_title}</h3>
                  {''.join(folder_blocks)}
                </section>
                """
            )

        nav_items.append(
            f"""
            <div class="nav-group">
              <a class="nav-link" href="#{level1_id}">{level1_title}</a>
              <div class="subnav">
                {''.join(sub_nav)}
              </div>
            </div>
            """
        )
        sections.append(
            f"""
            <section class="group" id="{level1_id}">
              <h2>{level1_title}</h2>
              {''.join(level2_sections)}
            </section>
            """
        )

    body = "".join(sections) if sections else "<p>No images found.</p>"
    nav = "".join(nav_items) if nav_items else '<span class="nav-empty">No folders</span>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f3ee;
      --panel: #fffdf9;
      --text: #1e1b18;
      --muted: #6b6259;
      --line: #d7cec2;
      --accent: #9a3412;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, #fff7ed 0, transparent 28%),
        linear-gradient(180deg, #f8f5ef 0%, var(--bg) 100%);
      color: var(--text);
    }}
    .layout {{
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      min-height: 100vh;
    }}
    .sidebar {{
      position: sticky;
      top: 0;
      align-self: start;
      height: 100vh;
      padding: 28px 20px 32px;
      background: rgba(255, 253, 249, 0.92);
      border-right: 1px solid var(--line);
      overflow-y: auto;
      backdrop-filter: blur(6px);
    }}
    .sidebar h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      line-height: 1.05;
      letter-spacing: -0.03em;
    }}
    .intro {{
      margin: 0 0 28px;
      color: var(--muted);
      font-size: 16px;
    }}
    .nav {{
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .nav-group {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .nav-link {{
      display: block;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      color: var(--text);
      text-decoration: none;
      font-size: 14px;
      word-break: break-word;
      transition: background 160ms ease, transform 160ms ease;
    }}
    .nav-link:hover {{
      background: #fff7ed;
      transform: translateX(2px);
    }}
    .subnav {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      padding-left: 12px;
      border-left: 2px solid var(--line);
    }}
    .subnav-link {{
      display: block;
      padding: 6px 10px;
      border-radius: 10px;
      text-decoration: none;
      color: var(--muted);
      font-size: 13px;
      word-break: break-word;
    }}
    .subnav-link:hover {{
      background: #fff7ed;
      color: var(--text);
    }}
    .nav-empty {{
      color: var(--muted);
      font-size: 14px;
    }}
    .content {{
      max-width: 1400px;
      padding: 40px 24px 80px;
    }}
    .content-header {{
      margin-bottom: 22px;
    }}
    .content-header h1 {{
      margin: 0;
      font-size: clamp(32px, 4vw, 56px);
    }}
    .group {{
      margin-top: 34px;
      padding-top: 18px;
      border-top: 2px solid var(--line);
    }}
    h2 {{
      margin: 0 0 18px;
      font-size: 24px;
      color: var(--accent);
      word-break: break-word;
    }}
    h3 {{
      margin: 0 0 14px;
      font-size: 20px;
      color: var(--text);
    }}
    .subgroup {{
      margin-top: 20px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255, 253, 249, 0.7);
    }}
    .folder-block + .folder-block {{
      margin-top: 18px;
    }}
    .folder-label {{
      margin: 0 0 10px;
      font-size: 13px;
      color: var(--muted);
      word-break: break-word;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 18px;
    }}
    .card {{
      display: block;
      text-decoration: none;
      color: inherit;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 12px 30px rgba(62, 39, 23, 0.08);
      transition: transform 180ms ease, box-shadow 180ms ease;
    }}
    .card:hover {{
      transform: translateY(-3px);
      box-shadow: 0 18px 40px rgba(62, 39, 23, 0.14);
    }}
    img {{
      display: block;
      width: 100%;
      height: 220px;
      object-fit: contain;
      background: #fff;
      padding: 10px;
    }}
    .caption {{
      padding: 12px 14px 14px;
      border-top: 1px solid var(--line);
      font-size: 14px;
      color: var(--muted);
      word-break: break-word;
    }}
    @media (max-width: 900px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        position: static;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}
      .content {{
        padding-top: 24px;
      }}
    }}
  </style>
</head>
<body>
  <main class="layout">
    <aside class="sidebar">
      <h1>{html.escape(title)}</h1>
      <p class="intro">Folders</p>
      <nav class="nav">
        {nav}
      </nav>
    </aside>
    <section class="content">
      <div class="content-header">
        <h1>{html.escape(title)}</h1>
      </div>
      {body}
    </section>
  </main>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Generate a static HTML image gallery for a folder tree."
    )
    parser.add_argument("root", help="Folder containing images and subfolders")
    parser.add_argument(
        "--title",
        default="Figure Gallery",
        help="Page title for the generated gallery",
    )
    parser.add_argument(
        "--output",
        default="index.html",
        help="Output HTML file path",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {root}")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(root, args.title), encoding="utf-8")

    print(f"Gallery written to: {output}")
    print(f"Image root: {root}")


if __name__ == "__main__":
    main()
