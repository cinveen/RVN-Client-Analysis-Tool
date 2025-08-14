# Assets Directory

This directory contains assets for the MkDocs documentation site.

## Required Assets

The following assets are required for the MkDocs documentation site:

- `logo.png` - The Reuters logo used in the header
- `favicon.ico` - The favicon used in the browser tab

## Adding Assets

When adding assets to this directory, please ensure they follow the Reuters brand guidelines.

### Logo Requirements

- The logo should be the official Reuters logo
- The logo should be in PNG format with a transparent background
- The logo should be at least 200px wide
- The logo should have the correct aspect ratio

### Favicon Requirements

- The favicon should be the official Reuters favicon
- The favicon should be in ICO format
- The favicon should be 32x32 pixels

## Usage

These assets are referenced in the `mkdocs.yml` file:

```yaml
theme:
  name: material
  logo: assets/logo.png
  favicon: assets/favicon.ico
