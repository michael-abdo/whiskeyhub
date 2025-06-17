# Whiskey Hub ML Demo - Design System

## 1. Global Navigation Structure

### Main Navigation Items
- **Home** - index.html
- **Recommendations** - recommendations.html
- **Flavor Profile** - flavor-profile.html
- **Rating Predictor** - rating-prediction.html
- **Taste Insights** - sensitivity-analysis.html
- **Gift Guide** - gift-guide.html

### Navigation Behavior
- Fixed header navigation with transparent background
- Background becomes solid on scroll
- Active page highlighted with accent color
- Smooth transitions on hover

## 2. Shared CSS Variables & Styling

### Color Palette
```css
/* Primary Colors */
--primary-color: #B8860B; /* Goldenrod - whiskey color */
--primary-dark: #8B6508;
--primary-light: #DAA520;

/* Secondary Colors */
--secondary-color: #2C1810; /* Dark brown */
--secondary-light: #4A2C1C;

/* Accent Colors */
--accent-color: #FF6B35; /* Burnt orange */
--accent-light: #FF8C42;

/* Neutral Colors */
--text-primary: #1A1A1A;
--text-secondary: #666666;
--text-light: #999999;
--background-primary: #FAFAFA;
--background-secondary: #F5F5F5;
--white: #FFFFFF;
--black: #000000;

/* Semantic Colors */
--success: #4CAF50;
--warning: #FFC107;
--error: #F44336;
--info: #2196F3;

/* Shadows */
--shadow-sm: 0 2px 4px rgba(0,0,0,0.1);
--shadow-md: 0 4px 8px rgba(0,0,0,0.15);
--shadow-lg: 0 8px 16px rgba(0,0,0,0.2);
```

### Typography
```css
/* Font Families */
--font-primary: 'Playfair Display', serif;
--font-secondary: 'Raleway', sans-serif;
--font-mono: 'Courier New', monospace;

/* Font Sizes */
--text-xs: 0.75rem;
--text-sm: 0.875rem;
--text-base: 1rem;
--text-lg: 1.125rem;
--text-xl: 1.25rem;
--text-2xl: 1.5rem;
--text-3xl: 1.875rem;
--text-4xl: 2.25rem;
--text-5xl: 3rem;

/* Font Weights */
--font-light: 300;
--font-normal: 400;
--font-medium: 500;
--font-semibold: 600;
--font-bold: 700;

/* Line Heights */
--leading-tight: 1.25;
--leading-normal: 1.5;
--leading-relaxed: 1.75;
```

### Spacing
```css
/* Spacing Scale */
--space-xs: 0.25rem;
--space-sm: 0.5rem;
--space-md: 1rem;
--space-lg: 1.5rem;
--space-xl: 2rem;
--space-2xl: 3rem;
--space-3xl: 4rem;

/* Border Radius */
--radius-sm: 0.25rem;
--radius-md: 0.5rem;
--radius-lg: 1rem;
--radius-full: 9999px;
```

### Button Styles
```css
/* Base Button */
.btn {
    font-family: var(--font-secondary);
    font-weight: var(--font-semibold);
    padding: var(--space-sm) var(--space-lg);
    border-radius: var(--radius-md);
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    text-decoration: none;
    display: inline-block;
    text-align: center;
}

/* Primary Button */
.btn-primary {
    background-color: var(--primary-color);
    color: var(--white);
}

.btn-primary:hover {
    background-color: var(--primary-dark);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

/* Secondary Button */
.btn-secondary {
    background-color: transparent;
    color: var(--primary-color);
    border: 2px solid var(--primary-color);
}

.btn-secondary:hover {
    background-color: var(--primary-color);
    color: var(--white);
}

/* Accent Button */
.btn-accent {
    background-color: var(--accent-color);
    color: var(--white);
}

.btn-accent:hover {
    background-color: var(--accent-light);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}
```

### Card Styles
```css
.card {
    background-color: var(--white);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    padding: var(--space-lg);
    transition: all 0.3s ease;
}

.card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-4px);
}

.card-header {
    font-family: var(--font-primary);
    font-size: var(--text-2xl);
    color: var(--text-primary);
    margin-bottom: var(--space-md);
}

.card-content {
    font-family: var(--font-secondary);
    color: var(--text-secondary);
    line-height: var(--leading-relaxed);
}
```

### Form Styles
```css
.form-group {
    margin-bottom: var(--space-lg);
}

.form-label {
    display: block;
    font-family: var(--font-secondary);
    font-weight: var(--font-semibold);
    color: var(--text-primary);
    margin-bottom: var(--space-sm);
}

.form-input {
    width: 100%;
    padding: var(--space-sm) var(--space-md);
    font-family: var(--font-secondary);
    font-size: var(--text-base);
    border: 2px solid #E0E0E0;
    border-radius: var(--radius-md);
    transition: all 0.3s ease;
}

.form-input:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(184, 134, 11, 0.1);
}

/* Slider Styles */
.slider {
    -webkit-appearance: none;
    width: 100%;
    height: 8px;
    border-radius: var(--radius-full);
    background: #E0E0E0;
    outline: none;
    transition: all 0.3s ease;
}

.slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 24px;
    height: 24px;
    border-radius: var(--radius-full);
    background: var(--primary-color);
    cursor: pointer;
    transition: all 0.3s ease;
}

.slider::-webkit-slider-thumb:hover {
    background: var(--primary-dark);
    transform: scale(1.1);
}
```

## 3. Component Templates

### Navigation HTML
```html
<nav class="navbar">
    <div class="nav-container">
        <div class="nav-logo">
            <h1>Whiskey Hub ML</h1>
            <span class="nav-tagline">AI-Powered Recommendations</span>
        </div>
        <ul class="nav-menu">
            <li class="nav-item"><a href="index.html" class="nav-link active">Home</a></li>
            <li class="nav-item"><a href="recommendations.html" class="nav-link">Recommendations</a></li>
            <li class="nav-item"><a href="flavor-profile.html" class="nav-link">Flavor Profile</a></li>
            <li class="nav-item"><a href="rating-prediction.html" class="nav-link">Rating Predictor</a></li>
            <li class="nav-item"><a href="sensitivity-analysis.html" class="nav-link">Taste Insights</a></li>
            <li class="nav-item"><a href="gift-guide.html" class="nav-link">Gift Guide</a></li>
        </ul>
    </div>
</nav>
```

### Footer HTML
```html
<footer class="footer">
    <div class="footer-container">
        <div class="footer-content">
            <div class="footer-section">
                <h3>Whiskey Hub ML</h3>
                <p>Advanced machine learning for personalized whiskey recommendations</p>
            </div>
            <div class="footer-section">
                <h4>Features</h4>
                <ul>
                    <li>Personalized Recommendations</li>
                    <li>Rating Predictions</li>
                    <li>Flavor Analysis</li>
                    <li>Gift Suggestions</li>
                </ul>
            </div>
            <div class="footer-section">
                <h4>Technology</h4>
                <ul>
                    <li>Machine Learning Models</li>
                    <li>Collaborative Filtering</li>
                    <li>Content-Based Analysis</li>
                    <li>SHAP Explanations</li>
                </ul>
            </div>
        </div>
        <div class="footer-bottom">
            <p>&copy; 2025 Whiskey Hub ML Demo. Powered by advanced ML algorithms.</p>
        </div>
    </div>
</footer>
```

### Card Component HTML
```html
<div class="card">
    <div class="card-header">
        <h3>Card Title</h3>
    </div>
    <div class="card-content">
        <p>Card content goes here...</p>
    </div>
    <div class="card-footer">
        <button class="btn btn-primary">Action</button>
    </div>
</div>
```

### Whiskey Card HTML
```html
<div class="whiskey-card">
    <div class="whiskey-image">
        <img src="whiskey-placeholder.jpg" alt="Whiskey Name">
        <div class="whiskey-rating">
            <span class="rating-score">4.5</span>
            <span class="rating-stars">★★★★☆</span>
        </div>
    </div>
    <div class="whiskey-details">
        <h3 class="whiskey-name">Whiskey Name</h3>
        <p class="whiskey-type">Single Malt Scotch</p>
        <div class="whiskey-attributes">
            <span class="attribute">Proof: 86</span>
            <span class="attribute">Price: $$$</span>
        </div>
        <div class="flavor-preview">
            <span class="flavor-tag">Sweet</span>
            <span class="flavor-tag">Fruity</span>
            <span class="flavor-tag">Spicy</span>
        </div>
    </div>
</div>
```

### Form Elements HTML
```html
<!-- Text Input -->
<div class="form-group">
    <label class="form-label" for="input-name">Label</label>
    <input type="text" id="input-name" class="form-input" placeholder="Enter value...">
</div>

<!-- Slider Input -->
<div class="form-group">
    <label class="form-label" for="slider-sweet">Sweet</label>
    <input type="range" id="slider-sweet" class="slider" min="0" max="10" value="5">
    <div class="slider-value">5</div>
</div>

<!-- Select Dropdown -->
<div class="form-group">
    <label class="form-label" for="select-type">Whiskey Type</label>
    <select id="select-type" class="form-input">
        <option value="">Select type...</option>
        <option value="bourbon">Bourbon</option>
        <option value="scotch">Scotch</option>
        <option value="irish">Irish</option>
        <option value="rye">Rye</option>
    </select>
</div>
```

## 4. Page Layout Standards

### Base Page Structure
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Title - Whiskey Hub ML</title>
    <style>
        /* Include all CSS variables and styles here */
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar">
        <!-- Navigation content -->
    </nav>

    <!-- Main Content -->
    <main class="main-container">
        <section class="hero-section">
            <!-- Hero content -->
        </section>
        
        <section class="content-section">
            <!-- Page-specific content -->
        </section>
    </main>

    <!-- Footer -->
    <footer class="footer">
        <!-- Footer content -->
    </footer>

    <script>
        /* Include JavaScript here */
    </script>
</body>
</html>
```

### Container Specifications
```css
.main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 var(--space-lg);
}

.hero-section {
    padding: var(--space-3xl) 0;
    text-align: center;
}

.content-section {
    padding: var(--space-2xl) 0;
}

.grid {
    display: grid;
    gap: var(--space-lg);
}

.grid-2 {
    grid-template-columns: repeat(2, 1fr);
}

.grid-3 {
    grid-template-columns: repeat(3, 1fr);
}

.grid-4 {
    grid-template-columns: repeat(4, 1fr);
}
```

### Responsive Breakpoints (Desktop-Only Focus)
```css
/* Minimum desktop width */
@media (min-width: 1024px) {
    body {
        min-width: 1024px;
    }
}

/* Large desktop */
@media (min-width: 1440px) {
    .main-container {
        max-width: 1400px;
    }
}
```

## 5. Animation & Interaction Standards

### Transitions
```css
/* Default transition */
.transition {
    transition: all 0.3s ease;
}

/* Fade in animation */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in {
    animation: fadeIn 0.6s ease-out;
}
```

### Loading States
```css
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(184, 134, 11, 0.3);
    border-radius: 50%;
    border-top-color: var(--primary-color);
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}
```

## 6. Data Visualization Standards

### Chart Colors
```javascript
const chartColors = {
    primary: '#B8860B',
    secondary: '#2C1810',
    accent: '#FF6B35',
    scale: [
        '#B8860B', '#FF6B35', '#2C1810', '#DAA520',
        '#8B6508', '#4A2C1C', '#FF8C42', '#666666'
    ]
};
```

### Chart Defaults
- Font: Raleway, sans-serif
- Grid lines: Light gray (#E0E0E0)
- Tooltips: Dark background with white text
- Animations: 1000ms duration, easeOutQuart timing