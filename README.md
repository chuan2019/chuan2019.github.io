# Chuan Zhang's Technical Blog

Personal technical blog built with [Jekyll](https://jekyllrb.com) and the [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) theme, hosted on [GitHub Pages](https://pages.github.com).

**Live site:** https://chuan-zhang.github.io

---

## Prerequisites

These only need to be installed once.

- **rbenv** — installed at `~/.rbenv` (manages the Ruby version)
- **Ruby 3.3.8** — installed via rbenv
- **Bundler + gems** — installed into `vendor/bundle`

If you are setting up on a new machine:

```bash
# 1. Install system dependencies
sudo apt-get install -y ruby-dev libyaml-dev build-essential

# 2. Install rbenv
curl -fsSL https://github.com/rbenv/rbenv-installer/raw/HEAD/bin/rbenv-installer | bash

# 3. Reload your shell, then install Ruby
export PATH="$HOME/.rbenv/bin:$PATH" && eval "$(rbenv init -)"
rbenv install 3.3.8

# 4. Install gems
bundle config set --local path 'vendor/bundle'
bundle install
```

---

## Daily workflow

### Preview locally

Start a live-reload server and open the site in your browser:

```bash
./scripts/preview.sh
```

The site is available at `http://localhost:4000`. The page reloads automatically whenever you save a file.

### Deploy to GitHub Pages

Build the site and push everything to GitHub in one command:

```bash
./scripts/deploy.sh
```

To use a custom commit message:

```bash
./scripts/deploy.sh "add post: your post title here"
```

The live site updates at https://chuan-zhang.github.io within about a minute of pushing.

---

## Writing a new post

1. Create a file in `_posts/` named with the format:

   ```
   _posts/YYYY-MM-DD-your-post-title.md
   ```

2. Add the front matter at the top of the file:

   ```yaml
   ---
   title: "Your Post Title"
   date: YYYY-MM-DD HH:MM:SS +0000
   categories: [Category]
   tags: [tag1, tag2]
   ---
   ```

3. Write your content in Markdown below the front matter.

4. Preview with `./scripts/preview.sh`, then deploy with `./scripts/deploy.sh`.

### Front matter options

| Field | Required | Description |
|---|:-:|---|
| `title` | Yes | Post title shown in the header and post list |
| `date` | Yes | Publication date and time |
| `categories` | No | Up to two levels, e.g. `[Machine Learning, NLP]` |
| `tags` | No | Any number of lowercase tags |
| `math` | No | Set to `true` to enable LaTeX math rendering |
| `image` | No | Cover image, e.g. `{ path: /assets/img/cover.png }` |
| `pin` | No | Set to `true` to pin the post to the top of the home page |

### Math equations

Enable math in the front matter (`math: true`), then use standard LaTeX syntax:

```markdown
Inline: $E = mc^2$

Block:
$$
\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}
$$
```

### Code blocks

Fenced code blocks with syntax highlighting:

````markdown
```python
def hello(name: str) -> str:
    return f"Hello, {name}"
```
````

Supported languages include `python`, `bash`, `javascript`, `go`, `sql`, `yaml`, and [many more](https://rouge-ruby.github.io/docs/Rouge/Lexers.html).

---

## Site configuration

All global settings are in `_config.yml`. Common fields to update:

| Field | Description |
|---|---|
| `title` | Site title shown in the sidebar |
| `tagline` | Subtitle below the site title |
| `avatar` | Path to your profile picture (e.g. `/assets/img/avatar.jpg`) |
| `timezone` | Your local timezone |
| `github.username` | Your GitHub username for the sidebar link |
| `social.email` | Your contact email |

After changing `_config.yml`, restart the preview server for changes to take effect.

---

## Adding a profile picture

1. Copy your image to `assets/img/avatar.jpg` (PNG also works).
2. The path is already set in `_config.yml`. If you use a different filename, update the `avatar:` field accordingly.

---

## Project structure

```
.
├── _config.yml          # Site-wide configuration
├── _posts/              # Blog posts (one Markdown file per post)
├── _tabs/               # Sidebar pages: About, Archives, Categories, Tags
├── _data/               # Contact links and share button config
├── _sass/addon/
│   └── custom.scss      # Futuristic theme overrides (colors, fonts, effects)
├── assets/
│   └── img/             # Images (avatar, post covers, etc.)
├── docs/                # Built site — committed and served by GitHub Pages
├── scripts/
│   ├── deploy.sh        # Build + commit + push
│   └── preview.sh       # Local live-reload server
├── Gemfile              # Ruby gem dependencies
└── Gemfile.lock         # Locked gem versions for reproducible builds
```

---

## GitHub Pages setup

This site is deployed by committing the built output to the `docs/` folder and pushing to `main`. GitHub Pages serves the contents of that folder directly — no CI pipeline required.

**One-time configuration** (already done if the site is live):

1. Go to the repository on GitHub.
2. Navigate to **Settings → Pages**.
3. Under **Source**, select **Deploy from a branch**.
4. Set branch to `main` and folder to `/docs`.
5. Click **Save**.
