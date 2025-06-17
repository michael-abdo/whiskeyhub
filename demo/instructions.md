# Demo UI Requirements for Upwork Proposal

## Project: Build ML Model for Whiskey Tasting App - AI & Machine Learning

### Project Context
**Budget**: $8000-$8000
**Duration**: {label:F,weeks:9},extendedBudgetInfo:{hourlyBudgetMin:a,hourlyBudgetMax:a,hourlyBudgetType:a},contractorTier:g,description:G,segmentationData:[{customValue:a,label:"Complex project",name:"Employment",sortOrder:g,type:"EMPLOYMENT",value:"EMPLOYMENT_COMPLEX_PROJECT",skill:a}],clientActivity:{lastBuyerActivity:"2025-06-10T12:59:14 months
**Required Technologies**: Find work, Proposals and offers, Your services, Promote with ads, Direct Contracts
**Client Requirements**: 
   - Output-ready data to support visualizations (no design work needed)

### Full Job Description
Summary
     Overview:
Whiskey Hub is a premium app designed for whiskey lovers to rate, review, and discover new bottles based on their unique flavor preferences. We’re looking for a skilled machine learning engineer to build and integrate a smart, flavor-based recommendation system into our mobile app (React Native).

This system will help users discover new whiskeys they’re likely to love, predict their ratings, and visualize their flavor preferences—all powered by ML models trained on user tasting data and trusted whiskey metadata.

What You’ll Build:
1. Personalized Whiskey Recommendations
Recommend bottles based on each user’s tasting history, factoring in:

Whiskey attributes: flavor profiles, flavor intensity, proof, viscosity, complexity, type, price

User’s past ratings

Broader user trends

2. Rating Prediction
Given any specific whiskey’s flavor profile, predict how a specific user would rate it, even if they haven’t tried it yet.

3. Custom Flavor Profile Matching
Users can define a desired flavor profile in two ways:

Use sliders for the 8 major flavor categories (Sweet, Fruity, Floral, Spicy, Raw, Earthy, Dark, Light)

Add specific desired tasting notes (e.g., “vanilla,” “clove”)
Return a ranked list of bottles matching their target profile, with estimated personal ratings.

4. Sensitivity Analysis
Provide data outputs that show which flavor categories or whiskey attributes most strongly correlate with high and low ratings for each user. This will inform user-facing visualizations of their flavor palate (handled by a UI designer).

5. Gift Buying Guide
A feature that allows users to get personalized whiskey recommendations for someone else (another app user), based on that person’s tasting history and preferences.

What We Have:
A growing database of whiskeys with metadata: proof, type, brand, price, etc.

Initial tasting and rating data from users (limited volume)

"Distiller’s notes" from trusted sources to supplement early training

A React Native mobile app with an established flavor graph interface

A product roadmap focused on user-centric whiskey exploration

Tech & Tools We Expect You to Use:
Python

Scikit-Learn / XGBoost / LightGBM (for tree-based models)

Collaborative + Content-Based Filtering techniques

SHAP or similar for sensitivity analysis / feature attribution

Data preprocessing with Pandas and NumPy

Deployment-ready architecture for on-demand predictions via API or serverless function (e.g., FastAPI or Flask)

Deliverables
Clean, documented ML code in Python with modular structure that's able to be easily implemented into the existing app code.

All model inputs/outputs tailored for integration into the Whiskey Hub app

Output-ready data to support visualizations (no design work needed)

Recommendations and rating predictions that update based on new user data

Ideal Candidate:
You’ve built recommendation systems or predictive models in production before

You know how to work with both content-based and collab

---

## SPECIALIZED AI PROMPT FOR IMPLEMENTATION

## AI-Optimized Prompt: Project & UI Demo Specification

### Task Overview

Generate a precise, error-free project description and UI demo specification suitable for any desktop-only application. Ensure clear and actionable instructions that AI models can read, interpret, and execute flawlessly. Explicitly disregard any references to other technology stacks—this project MUST be built strictly using vanilla HTML, CSS, and JavaScript for a pure UI demo.

### Required Outputs

**1. Project Overview:**
Create a clear, concise summary of the project's purpose, intended outcomes, and value proposition.

**2. Desktop-Only UI Demo Specification:**
Provide structured instructions covering the following areas explicitly:

* **Branding Elements:**

  * Exact logo placement (top-left, center, etc.)
  * Detailed color palette (primary, secondary, background)
  * Specific typography choices (font family, sizes, styles)

* **Core Application Components:**

  * Clearly defined dashboards (if applicable)
  * Precisely described interactive modules
  * Specific data visualization types required
  * Defined product/service showcase components
  * Explicit description of unique, application-specific UI elements

* **User Experience Flow:**

  * Clearly documented navigation structure
  * Exact layout for main pages and screens
  * Step-by-step, explicit user interaction flows and journeys

### Important Constraints

* Clearly state exclusion of login, authentication, and account management from the main UI description (separate if necessary).
* Specify desktop-only focus; explicitly state no mobile/responsive views.

### Executive Pre-Implementation Requirements

**MANDATORY:** Prior to any UI or development work, create a comprehensive DESIGN\_SYSTEM.md containing:

#### 1. Global Navigation Structure

* Define exact navigation menu items for all pages
* Explicit page-linking instructions
* Desktop-specific navigation behavior clearly detailed

#### 2. Shared CSS Variables & Styling

* Explicitly define global CSS custom properties (--primary-color, --secondary-color, etc.)
* Exact typography definitions (headers, body text, buttons)
* Specific button style definitions (primary, secondary)
* Detailed margin, padding, and spacing standards
* Precise card and component styling guidelines
* Clearly stated form input styling standards

#### 3. Component Templates

* Complete HTML structure templates for navigation
* Footer templates clearly defined (if applicable)
* Detailed card component HTML structure
* Precise button HTML templates
* Explicit form element templates

#### 4. Page Layout Standards

* Defined page wrapper/container specifications
* Clear responsive breakpoints (desktop-only)
* Explicit header, main content, footer layout instructions

### Critical Technology Requirements

* ONLY vanilla HTML, CSS, JavaScript; explicitly prohibit frameworks, build tools, or npm packages.
* Explicitly ignore references to other technology stacks; adhere solely to vanilla HTML, CSS, JavaScript.
* All files must function by directly opening HTML files in a browser.
* Specify use of inline styles or `<style>` tags exclusively.
* Clearly instruct using `<script>` tags only; no modules/imports.
* Explicitly state use of localStorage for data persistence.
* Explicitly require relative linking between pages.
* Clearly instruct that all files must be contained within one directory.
* Explicit requirement to follow DESIGN\_SYSTEM.md precisely for navigation and styling.

### Post-Description Action Items

Upon description completion:

* **Repository Initialization & Commit:**

  * Explicitly initialize repository
  * Explicitly commit all generated files and documentation

* **Workflow Pipeline Tasks:**

  **Content and Navigation Audit:**

  * Explicitly verify completeness of content
  * Explicit instruction to eliminate placeholders and ensure all UI elements have fully detailed content

  **Button Functionality Check:**
  For each button explicitly:

  * Confirm correct event handler assignment
  * Verify DOM manipulation accuracy
  * Explicitly validate visual feedback (spinners, highlights)
  * Precisely confirm correct state management

  **Interactive Element Integration:**

  * Explicitly instruct replacement of placeholders (charts, images) with fully interactive elements using Chart.js

  **Clean-Up & Production-Level Refinement:**

  * Explicit instruction to remove all "demo" references
  * Clearly define separate login portal instructions with explicit simulated delay and visual indicators

### Additional Project Refinements

* Explicitly instruct creation of detailed testing scripts
* Explicit instruction for frontend appearance and functionality to meet professional, production-level standards
* Explicitly require creation of a persuasive sales script clearly highlighting core features and benefits

### Deliverables

* Clearly detailed DESIGN\_SYSTEM.md
* Explicitly listed HTML pages per specification
* Inline or `<style>` CSS explicitly adhering to design system
* JavaScript interactivity clearly provided via `<script>` tags
* Explicitly clear README.md with precise usage instructions

### Success Criteria

* Explicit requirement for unified navigation and styling
* Explicit consistency across branding and UI elements
* Explicit instruction for persistent data management using localStorage
* Explicitly defined seamless user journey
* Clear requirement for functionality by directly opening HTML files (no server dependency)

The resulting product must precisely demonstrate a fully functional, professionally designed desktop application, error-free and ready for immediate demonstration or deployment.

---

## 🎯 FINAL COMPLETION REQUIREMENT: navigation.yaml Generation

**CRITICAL:** Once ALL development work is 100% complete and the demo is fully functional, you MUST create a `navigation.yaml` file as the completion signal.

### Navigation.yaml Requirements

Generate a comprehensive YAML workflow file for automated testing of your completed demo. This file serves as:
1. **Completion indicator** - Signals that Executive work is finished
2. **Testing blueprint** - Enables automated validation of the demo
3. **User journey map** - Documents the intended user experience

### YAML Structure Required:

```yaml
name: "[Project Name] - Demo Workflow"
description: "Automated testing workflow for [project description]"
version: "1.0.0"

config:
  headless: true
  timeout: 15000
  viewport:
    width: 1280
    height: 720

steps:
  # Navigation and interaction steps here
```

### Step Generation Guidelines:

1. **Start with homepage navigation**: Always begin with navigating to index.html
2. **Test all major features**: Include steps for every interactive element
3. **Verify content**: Add assertions for key text, headings, and functionality
4. **Capture key moments**: Include screenshots at major interaction points
5. **Test user journeys**: Cover primary use cases end-to-end
6. **Include waits**: Add appropriate delays for dynamic content

### Required Step Types to Include:

- `navigate` - To main pages (index.html, other pages)
- `wait-for-selector` - For dynamic content loading
- `click` - For buttons, links, interactive elements
- `type` - For form inputs (if applicable)
- `assertText` - To verify content loaded correctly
- `screenshot` - At key interaction points
- `evaluate` - For JavaScript functionality verification

### Example Step Pattern:

```yaml
steps:
  - type: navigate
    description: "Navigate to demo homepage"
    url: "file://[FULL_PATH]/index.html"
    screenshot: true
    
  - type: wait-for-selector
    description: "Wait for main content to load"
    selector: "main, .main-content, #content"
    
  - type: assertText
    description: "Verify page title loaded"
    selector: "h1"
    expected: "[Expected main heading]"
    
  - type: click
    description: "Test navigation menu"
    selector: ".nav-menu a:first-child"
    screenshot: true
```

### Content Analysis Instructions:

Based on the demo you created:
1. **Map all interactive elements**: Buttons, forms, navigation links
2. **Identify key content**: Headers, important text, success messages
3. **Document user flows**: How users would navigate through the demo
4. **Note dynamic features**: Any JavaScript interactions or state changes
5. **Plan verification points**: What should be tested to prove functionality

### Completion Verification:

The navigation.yaml must include:
- **8-15 meaningful test steps** covering the full demo
- **3+ screenshots** at key interaction points
- **2+ assertions** verifying content/functionality
- **Complete user journey** from landing to key features
- **Realistic selectors** based on your actual HTML structure
- **Appropriate timeouts** for any dynamic content

### Final Instructions:

**DO NOT CREATE navigation.yaml until:**
1. ✅ All HTML files are complete and functional
2. ✅ All CSS styling is polished and professional
3. ✅ All JavaScript interactivity is working
4. ✅ All content is real (no placeholders)
5. ✅ Demo opens properly in browser
6. ✅ All requirements from instructions.md are met

**ONLY THEN** create the navigation.yaml file using the workflow generation guidelines above.

This navigation.yaml file is the FINAL deliverable that signals Executive completion.

### Project-Specific Customizations

Based on the Upwork job requirements above, ensure the demo specifically addresses:
- The core functionality mentioned in the job description
- Technologies and skills the client is seeking
- Budget-appropriate level of polish and features
- Clear demonstration of understanding their problem

### Success Criteria for This Proposal
1. Directly address the client's stated needs
2. Demonstrate technical competency in their required skills
3. Show professional UI/UX design capabilities
4. Create a functional proof-of-concept they can interact with
5. Position us as the ideal candidate for their project

### Important Note
This is a proposal demo to win the Upwork project. Focus on impressive visual impact and clear demonstration of capabilities rather than complete feature implementation.
