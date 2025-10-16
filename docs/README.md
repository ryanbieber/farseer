# Farseer Documentation

This directory contains the GitHub Pages site for Farseer.

## Structure

- `index.html` - Homepage
- `quickstart.html` - Quick start guide
- `examples.html` - Examples and tutorials
- `migration.html` - Prophet to Farseer migration guide
- `styles.css` - Global stylesheet
- `script.js` - Interactive JavaScript
- `_config.yml` - Jekyll configuration for GitHub Pages

## Local Development

To preview the site locally, you can use any static file server:

```bash
# Using Python
cd docs
python -m http.server 8000

# Using Node.js
npx http-server docs

# Then visit http://localhost:8000
```

## Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the `master` branch.

Visit: https://ryanbieber.github.io/seer/

## Building

No build step required! The site uses vanilla HTML, CSS, and JavaScript for simplicity and performance.
