# Ben Eze's Portfolio

This repository contains the source code for my personal website, hosted at [ben-eze.github.io](https://ben-eze.github.io).

## Development

The site is built with [Eleventy](https://www.11ty.dev/). Page shell (nav, `<head>`) lives once in `src/_includes/`; each page's content is its own file under `src/`.

```
npm install
npm run serve   # local dev server with live reload
npm run build   # outputs static site to _site/
```

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds the site and deploys it via GitHub Pages.