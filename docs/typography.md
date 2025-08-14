# Reuters Teletrax Analysis Tool - Typography Guide

This document provides detailed information about the typography implementation in the Reuters Teletrax Analysis Tool, following the official Thomson Reuters font style guide.

## Font Implementation

The application uses the official Thomson Reuters Clario font family throughout the web interface, with Arial as a fallback and for PowerPoint presentations.

### Clario Font Family

Clario is the exclusive typeface created for Thomson Reuters, designed for balance, clarity, and modernity. It's a versatile sans-serif font that works well across digital and print media.

#### Font Files

The following Clario font files are included in the application:

- **Web Fonts** (in `static/fonts/`):
  - Clario-Air.woff and Clario-Air.woff2
  - Clario-AirItalic.woff and Clario-AirItalic.woff2
  - Clario-Thin.woff and Clario-Thin.woff2
  - Clario-ThinItalic.woff and Clario-ThinItalic.woff2
  - Clario-Light.woff and Clario-Light.woff2
  - Clario-LightItalic.woff and Clario-LightItalic.woff2
  - Clario-Regular.woff and Clario-Regular.woff2
  - Clario-RegularItalic.woff and Clario-RegularItalic.woff2
  - Clario-Medium.woff and Clario-Medium.woff2
  - Clario-MediumItalic.woff and Clario-MediumItalic.woff2
  - Clario-Bold.woff and Clario-Bold.woff2
  - Clario-BoldItalic.woff and Clario-BoldItalic.woff2
  - Clario-Black.woff and Clario-Black.woff2
  - Clario-BlackItalic.woff and Clario-BlackItalic.woff2

#### Font Weights

The application uses specific font weights according to the Reuters style guide:

- **Medium (500)**: Used for headlines, large titles, and navigation elements
- **Regular (400)**: Used for body copy, captions, and most UI text
- **Bold (700)**: Used sparingly for emphasis within body text
- **Light (300)**: Used for certain UI elements where a lighter weight is appropriate

### Font Implementation in CSS

The Clario font is implemented in the CSS using `@font-face` declarations for each weight and style. The font-family is defined as 'Clario' with Arial and sans-serif as fallbacks.

```css
@font-face {
    font-family: 'Clario';
    src: url('../fonts/Clario-Regular.woff2') format('woff2'),
         url('../fonts/Clario-Regular.woff') format('woff');
    font-weight: 400;
    font-style: normal;
    font-display: swap;
}

@font-face {
    font-family: 'Clario';
    src: url('../fonts/Clario-Medium.woff2') format('woff2'),
         url('../fonts/Clario-Medium.woff') format('woff');
    font-weight: 500;
    font-style: normal;
    font-display: swap;
}

/* Additional @font-face declarations for other weights and styles */

body {
    font-family: 'Clario', Arial, sans-serif;
    font-weight: 400;
    /* Other properties */
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Clario', Arial, sans-serif;
    font-weight: 500;
    /* Other properties */
}
```

## Typography Hierarchy

The application implements a clear typography hierarchy according to the Reuters style guide:

### Web Interface Typography

- **XL Headline (h1)**:
  - Font: Clario Medium
  - Size: 2.5rem (40px)
  - Line height: 1.2
  - Used for main page titles

- **Headline 1 (h2)**:
  - Font: Clario Medium
  - Size: 2rem (32px)
  - Line height: 1.25
  - Used for major section headings

- **Headline 2 (h3)**:
  - Font: Clario Medium
  - Size: 1.5rem (24px)
  - Line height: 1.3
  - Used for card headers and subsection titles

- **Subhead (h4, h5)**:
  - Font: Clario Medium
  - Size: 1.25rem (20px)
  - Line height: 1.4
  - Used for minor section headings

- **Body Text**:
  - Font: Clario Regular
  - Size: 1rem (16px)
  - Line height: 1.5
  - Used for main content text

- **Small Text / Captions**:
  - Font: Clario Regular
  - Size: 0.875rem (14px)
  - Line height: 1.4
  - Used for captions, footnotes, and secondary information

### PowerPoint Typography

PowerPoint presentations use Arial font as recommended by the Reuters style guide for downloadable and shareable content:

- **Slide Title**:
  - Font: Arial Bold
  - Size: 28pt
  - Used for slide titles

- **Slide Subtitle**:
  - Font: Arial Regular
  - Size: 20pt
  - Used for slide subtitles

- **Heading**:
  - Font: Arial Bold
  - Size: 18pt
  - Used for section headings within slides

- **Body Text**:
  - Font: Arial Regular
  - Size: 14pt
  - Used for main content text

- **Caption / Footer**:
  - Font: Arial Regular
  - Size: 10pt
  - Used for captions, footnotes, and slide footers

## Chart and Visualization Typography

For charts and visualizations, the application uses the following typography settings:

- **Chart Title**:
  - Font: Clario Medium (web) / Arial Bold (PowerPoint)
  - Size: 16px (web) / 16pt (PowerPoint)

- **Axis Labels**:
  - Font: Clario Regular (web) / Arial Regular (PowerPoint)
  - Size: 12px (web) / 12pt (PowerPoint)

- **Legend**:
  - Font: Clario Regular (web) / Arial Regular (PowerPoint)
  - Size: 10px (web) / 10pt (PowerPoint)

- **Data Labels**:
  - Font: Clario Regular (web) / Arial Regular (PowerPoint)
  - Size: 10px (web) / 10pt (PowerPoint)

## Implementation Details

### Web Interface

The typography is implemented in the web interface through the `tr_styles.css` file, which defines the font-family, font-weight, font-size, and line-height properties for all text elements.

### PowerPoint Generation

The PowerPoint generation code in `teletrax_analysis.py` sets the font family to Arial for all text elements in the PowerPoint presentations, following the Reuters style guide recommendation for downloadable and shareable content.

### Matplotlib Charts

For charts generated with Matplotlib, the font settings are configured in the `teletrax_analysis.py` file:

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Try to use Clario font if available, otherwise fall back to Arial
try:
    plt.rcParams['font.family'] = ['Clario', 'Arial', 'sans-serif']
except:
    plt.rcParams['font.family'] = ['Arial', 'sans-serif']

# Set font sizes
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
```

## Accessibility Considerations

The typography implementation includes considerations for accessibility:

- **Minimum Font Size**: Body text is never smaller than 16px (1rem) to ensure readability
- **Contrast Ratio**: Text colors maintain a minimum contrast ratio of 4.5:1 against their backgrounds
- **Line Height**: Generous line height (1.5 for body text) improves readability
- **Font Weights**: Clear distinction between different font weights for visual hierarchy

## Future Improvements

Potential future improvements to the typography implementation:

- Add support for variable font technology when Clario variable fonts become available
- Implement responsive typography that adjusts based on viewport size
- Add more extensive internationalization support for non-Latin scripts
- Implement a dark mode with appropriate typography adjustments

## References

- Thomson Reuters Brand Guidelines (2025)
- Reuters Typography Style Guide (2025)
- Web Content Accessibility Guidelines (WCAG) 2.1
