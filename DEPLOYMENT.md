# Restaurant Dashboard Deployment Guide

## Overview
This restaurant analytics dashboard has been transformed into a modern presentation-style web application, similar to the PolicyEngine OBR presentation format. It features slide-based navigation with arrow key support and is ready for deployment to GitHub Pages.

## Features
- Slide-based navigation with arrow keys
- Animated transitions between slides
- Responsive design
- Interactive data visualizations
- Automatic GitHub Pages deployment

## Local Development
```bash
cd restaurant-dashboard
npm install --legacy-peer-deps
npm start
```

Visit `http://localhost:3000` to view the dashboard locally.

## Manual Deployment to GitHub Pages

To deploy your dashboard to GitHub Pages:

```bash
cd restaurant-dashboard
npm run deploy
```

This will:
1. Build the production version
2. Deploy to GitHub Pages at: https://vahid-ahmadi.github.io/stint/restaurant-dashboard

## Automatic Deployment

The repository includes a GitHub Actions workflow that automatically deploys the dashboard when you push changes to the main branch. The workflow is located at `.github/workflows/deploy-dashboard.yml`.

## Repository Setup for GitHub Pages

1. Go to your repository settings on GitHub
2. Navigate to "Pages" section
3. Set Source to "Deploy from a branch"
4. Select "gh-pages" branch and "/ (root)" folder
5. Save the settings

## Navigation

Use the following to navigate through the presentation:
- **Arrow Keys**: Press ← and → to navigate between slides
- **Click Navigation**: Click on the navigation tabs at the top
- **Available Slides**:
  - Introduction
  - Key Findings
  - Demand Patterns
  - Recommendations
  - Full Dashboard (original dashboard view)

## Project Structure
```
restaurant-dashboard/
├── src/
│   ├── slides/          # Presentation slides
│   ├── components/      # Reusable components
│   ├── hooks/          # Custom React hooks
│   └── types/          # TypeScript definitions
├── public/
│   └── analysis_results.json  # Data file
└── package.json
```

## Customization

To add new slides:
1. Create a new slide component in `src/slides/`
2. Import it in `src/App.tsx`
3. Add it to the `slides` array

## Tech Stack
- React 19 with TypeScript
- Styled Components for styling
- Framer Motion for animations
- React Router for navigation
- Recharts for data visualization
- GitHub Pages for hosting

## Troubleshooting

If the build fails with peer dependency issues:
```bash
npm install --legacy-peer-deps
```

If GitHub Pages shows 404:
- Ensure the `homepage` field in `package.json` matches your GitHub username and repository
- Wait 5-10 minutes after deployment for changes to propagate

## Live URL
Once deployed, your dashboard will be available at:
https://vahid-ahmadi.github.io/stint/restaurant-dashboard