# Reuters Teletrax Analysis Tool - Branding Guide

This document provides detailed information about the branding implementation in the Reuters Teletrax Analysis Tool, following the official Thomson Reuters brand guidelines.

## Color Palette

The application implements the official Thomson Reuters brand colors throughout the interface and visualizations.

### Primary Colors

- **TR Orange (#D64000)**
  - Used for primary buttons, headers, and key UI elements
  - Represents the Reuters brand identity
  - Hex: #D64000
  - RGB: 214, 64, 0
  - CMYK: 0, 70, 100, 16

- **TR Racing Green (#123015)**
  - Used for secondary elements and footer
  - Provides a professional, authoritative contrast to TR Orange
  - Hex: #123015
  - RGB: 18, 48, 21
  - CMYK: 63, 0, 56, 81

- **TR White (#FFFFFF)**
  - Used as the main background color
  - Provides clean, clear space for content
  - Hex: #FFFFFF
  - RGB: 255, 255, 255
  - CMYK: 0, 0, 0, 0

### Secondary Colors

- **Sky Pair**
  - Light: #E3F1FD
  - Dark: #0874E3
  - Used for card backgrounds, content areas, and data visualization
  - Represents clarity and insight

- **Grass Pair**
  - Light: #E5F4E7
  - Dark: #1D8C29
  - Used for positive trends and growth indicators in visualizations
  - Represents growth and positive performance

- **Sand Pair**
  - Light: #FFF5E5
  - Dark: #FF9D28
  - Used for warnings and cautionary indicators
  - Represents attention and awareness

- **Clay Pair**
  - Light: #FFEAE5
  - Dark: #D64000 (same as TR Orange)
  - Used for errors and critical information
  - Represents urgency and importance

## Color Implementation in CSS

The color palette is implemented in the CSS using custom properties (CSS variables) for consistency and easy updates:

```css
:root {
    /* Primary Colors */
    --tr-orange: #D64000;
    --tr-racing-green: #123015;
    --tr-white: #FFFFFF;
    
    /* Secondary Colors - Sky Pair */
    --tr-sky-light: #E3F1FD;
    --tr-sky-dark: #0874E3;
    
    /* Secondary Colors - Grass Pair */
    --tr-grass-light: #E5F4E7;
    --tr-grass-dark: #1D8C29;
    
    /* Secondary Colors - Sand Pair */
    --tr-sand-light: #FFF5E5;
    --tr-sand-dark: #FF9D28;
    
    /* Secondary Colors - Clay Pair */
    --tr-clay-light: #FFEAE5;
    --tr-clay-dark: #D64000;
    
    /* Text Colors */
    --text-primary: #333333;
    --text-secondary: #666666;
    --text-tertiary: #999999;
}
```

## Logo Usage

The Reuters logo is used in the application according to the Thomson Reuters brand guidelines:

### Primary Logo
- **Placement**: Top-left corner of the header
- **Clear Space**: Maintained around the logo (minimum of 1x the height of the logo)
- **Minimum Size**: Never smaller than 100px wide to ensure legibility
- **Background**: Always placed on a solid background (white or TR Racing Green)
- **Variations**: Only the approved logo variations are used

### Symbol Mark (Favicon)
- **Placement**: Used in the footer for a cleaner, more aesthetically pleasing look
- **Purpose**: Provides brand recognition in a compact form without visual clutter
- **Size**: 24px height to ensure visibility while maintaining a subtle presence
- **Implementation**: Used the circular orange symbol mark from the Thomson Reuters favicon package

## Typography

Please refer to the [Typography Guide](typography.md) for detailed information about the typography implementation.

## UI Elements

### Buttons

- **Primary Buttons**:
  - Background: TR Orange (#D64000)
  - Text: White (#FFFFFF)
  - Hover: Darker shade of TR Orange
  - Border: None
  - Border Radius: 4px

- **Secondary Buttons**:
  - Background: White (#FFFFFF)
  - Text: TR Orange (#D64000)
  - Border: 1px solid TR Orange
  - Hover: Light orange background
  - Border Radius: 4px

- **Tertiary Buttons**:
  - Background: Transparent
  - Text: TR Orange (#D64000)
  - Border: None
  - Hover: Light orange background
  - Border Radius: 4px

### Cards

- **Standard Card**:
  - Background: White (#FFFFFF)
  - Border: 1px solid #E0E0E0
  - Border Radius: 8px
  - Shadow: 0 2px 4px rgba(0, 0, 0, 0.1)

- **Highlighted Card**:
  - Background: TR Sky Light (#E3F1FD)
  - Border: 1px solid #E0E0E0
  - Border Radius: 8px
  - Shadow: 0 2px 4px rgba(0, 0, 0, 0.1)

### Form Elements

- **Input Fields**:
  - Border: 1px solid #E0E0E0
  - Border Radius: 4px
  - Focus: Border color changes to TR Sky Dark (#0874E3)
  - Padding: 8px 12px

- **Dropdowns**:
  - Border: 1px solid #E0E0E0
  - Border Radius: 4px
  - Arrow: Custom dropdown arrow in TR Orange

- **Checkboxes and Radio Buttons**:
  - Selected: TR Orange (#D64000)
  - Border: 1px solid #E0E0E0
  - Border Radius: 2px (checkbox), 50% (radio)

### Alerts and Notifications

- **Success**:
  - Background: TR Grass Light (#E5F4E7)
  - Border: 1px solid TR Grass Dark (#1D8C29)
  - Icon: Checkmark in TR Grass Dark

- **Warning**:
  - Background: TR Sand Light (#FFF5E5)
  - Border: 1px solid TR Sand Dark (#FF9D28)
  - Icon: Exclamation triangle in TR Sand Dark

- **Error**:
  - Background: TR Clay Light (#FFEAE5)
  - Border: 1px solid TR Clay Dark (#D64000)
  - Icon: Exclamation circle in TR Clay Dark

- **Info**:
  - Background: TR Sky Light (#E3F1FD)
  - Border: 1px solid TR Sky Dark (#0874E3)
  - Icon: Information circle in TR Sky Dark

## Data Visualization

### Charts and Graphs

- **Color Scheme**:
  - Primary: TR Orange (#D64000)
  - Secondary: TR Sky Dark (#0874E3)
  - Tertiary: TR Grass Dark (#1D8C29)
  - Quaternary: TR Sand Dark (#FF9D28)
  - Additional colors from the Reuters extended palette as needed

- **Chart Types**:
  - Bar Charts: Used for comparing values across categories
  - Line Charts: Used for showing trends over time
  - Pie/Donut Charts: Used for showing composition
  - Heatmaps: Used for showing patterns across two dimensions

- **Chart Elements**:
  - Grid Lines: Light gray (#E0E0E0)
  - Axis Lines: Dark gray (#666666)
  - Labels: TR Racing Green (#123015)
  - Tooltips: White background with TR Racing Green text

### PowerPoint Presentations

PowerPoint presentations follow the Reuters brand guidelines:

- **Slide Background**: White (#FFFFFF)
- **Slide Title**: TR Racing Green (#123015)
- **Headings**: TR Racing Green (#123015)
- **Body Text**: Dark gray (#333333)
- **Charts and Graphs**: Reuters color palette as described above
- **Footer**: Contains Reuters logo and copyright information

## Implementation Details

### Web Interface

The branding is implemented in the web interface through the `tr_styles.css` file, which defines the color variables, typography, and UI element styles.

### PowerPoint Generation

The PowerPoint generation code in `teletrax_analysis.py` applies the Reuters branding to all slides, including:

- Setting the slide background to white
- Using the Reuters color palette for charts and graphs
- Applying the correct typography (Arial for PowerPoint)
- Including the Reuters logo and copyright information in the footer

### Matplotlib Charts

For charts generated with Matplotlib, the branding is applied in the `teletrax_analysis.py` file:

```python
import matplotlib.pyplot as plt

# Set Reuters brand colors
tr_orange = '#D64000'
tr_racing_green = '#123015'
tr_sky_dark = '#0874E3'
tr_grass_dark = '#1D8C29'
tr_sand_dark = '#FF9D28'

# Apply colors to chart
plt.figure(figsize=(12, 6))
plt.plot(x_data, y_data, color=tr_orange, linewidth=2)
plt.title('Chart Title', color=tr_racing_green)
plt.xlabel('X Axis Label', color=tr_racing_green)
plt.ylabel('Y Axis Label', color=tr_racing_green)
plt.grid(True, alpha=0.3)
```

## Accessibility Considerations

The branding implementation includes considerations for accessibility:

- **Color Contrast**: All text colors maintain a minimum contrast ratio of 4.5:1 against their backgrounds
- **Color Independence**: Information is never conveyed by color alone; patterns, labels, or icons are used in addition to color
- **Focus Indicators**: Clear focus indicators for keyboard navigation
- **Alternative Text**: All images, including the Reuters logo, have appropriate alternative text

## Future Improvements

Potential future improvements to the branding implementation:

- Implement a dark mode with appropriate color adjustments
- Create additional chart templates with Reuters branding
- Develop a component library with pre-styled UI elements
- Implement more interactive data visualizations with consistent branding

## References

- Thomson Reuters Brand Guidelines (2025)
- Reuters Visual Identity System (2025)
- Web Content Accessibility Guidelines (WCAG) 2.1
