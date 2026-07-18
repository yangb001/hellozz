/**
 * ThinkingComponent - Thinking process display component
 *
 * Displays the LLM thinking process with support for:
 * - Collapsible display
 * - Multi-step display
 * - Duration statistics
 * - Real-time updates
 */
class ThinkingComponent {
  /**
   * Create thinking component
   * @param {string} containerId - Container element ID
   */
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.steps = [];
    this.currentStep = null;
    this.isVisible = false;
    this.startTime = null;

    if (!this.container) {
      console.error(`Thinking container not found: ${containerId}`);
    }
  }

  /**
   * Start a new thinking step
   * @param {string} label - Step label
   * @returns {Object} Thinking step object
   */
  startThinking(label = '') {
    // Save previous step
    if (this.currentStep && this.currentStep.content) {
      this.currentStep.endTime = Date.now();
      this.currentStep.duration = this.currentStep.endTime - this.currentStep.startTime;
      this.steps.push(this.currentStep);
    }

    this.currentStep = {
      step: this.steps.length + 1,
      label: label,
      content: '',
      startTime: Date.now(),
      endTime: null,
      duration: null
    };

    if (!this.startTime) {
      this.startTime = Date.now();
    }

    this.render();
    return this.currentStep;
  }

  /**
   * Add thinking content
   * @param {string} content - Content fragment
   * @returns {Object} Current thinking step
   */
  addContent(content) {
    if (this.currentStep) {
      this.currentStep.content += content;
      this.render();
    }
    return this.currentStep;
  }

  /**
   * End current thinking step
   * @returns {Object} Completed thinking step
   */
  endThinking() {
    if (this.currentStep) {
      this.currentStep.endTime = Date.now();
      this.currentStep.duration = this.currentStep.endTime - this.currentStep.startTime;
      this.steps.push(this.currentStep);
      this.currentStep = null;
      this.render();
    }
    return this.steps[this.steps.length - 1];
  }

  /**
   * Get total duration
   * @returns {number} Total duration in milliseconds
   */
  getTotalDuration() {
    const stepsDuration = this.steps.reduce((sum, step) => {
      return sum + (step.duration || 0);
    }, 0);

    const currentDuration = this.currentStep
      ? (Date.now() - this.currentStep.startTime)
      : 0;

    return stepsDuration + currentDuration;
  }

  /**
   * Get step count
   * @returns {number} Number of steps
   */
  getStepCount() {
    return this.steps.length + (this.currentStep ? 1 : 0);
  }

  /**
   * Toggle visibility
   */
  toggle() {
    this.isVisible = !this.isVisible;
    this.render();
  }

  /**
   * Show thinking content
   */
  show() {
    this.isVisible = true;
    this.render();
  }

  /**
   * Hide thinking content
   */
  hide() {
    this.isVisible = false;
    this.render();
  }

  /**
   * Clear all records
   */
  clear() {
    this.steps = [];
    this.currentStep = null;
    this.startTime = null;
    this.isVisible = false;
    this.render();
  }

  /**
   * Render component
   */
  render() {
    if (!this.container) return;

    if (this.steps.length === 0 && !this.currentStep) {
      this.container.innerHTML = '';
      return;
    }

    const allSteps = this.currentStep
      ? [...this.steps, this.currentStep]
      : this.steps;

    const totalDuration = this.getTotalDuration();
    const stepCount = this.getStepCount();

    const html = `
      <div class="thinking-container">
        <div class="thinking-header" onclick="thinkingComponent.toggle()">
          <span class="thinking-icon">💭</span>
          <span class="thinking-title">Thinking Process (${stepCount} steps)</span>
          <span class="thinking-duration">${(totalDuration / 1000).toFixed(1)}s</span>
          <span class="thinking-toggle ${this.isVisible ? 'expanded' : ''}">&#9654;</span>
        </div>
        <div class="thinking-content ${this.isVisible ? 'visible' : ''}">
          ${allSteps.map(step => `
            <div class="thinking-step ${step === this.currentStep ? 'active' : ''}">
              <div class="step-header">
                <span class="step-number">Step ${step.step}</span>
                <span class="step-label">${step.label || 'Thinking...'}</span>
                ${step.duration
                  ? `<span class="step-duration">${(step.duration / 1000).toFixed(1)}s</span>`
                  : '<span class="thinking-loading"><span class="dot"></span><span class="dot"></span><span class="dot"></span></span>'
                }
              </div>
              <div class="step-content">${step.content || '...'}</div>
            </div>
          `).join('')}
        </div>
      </div>
    `;

    this.container.innerHTML = html;
  }

  /**
   * Handle thinking event
   * @param {Object} event - Event object
   */
  handleEvent(event) {
    if (!event || !event.type) return;

    switch (event.type) {
      case 'thinking_start':
        this.startThinking(event.thinking?.label || '');
        break;

      case 'thinking_content':
        this.addContent(event.content || '');
        break;

      case 'thinking_end':
        this.endThinking();
        break;
    }
  }

  /**
   * Get all step data
   * @returns {Array} Steps array
   */
  getSteps() {
    return [...this.steps];
  }

  /**
   * Get summary
   * @returns {string} Summary text
   */
  getSummary() {
    if (this.steps.length === 0) return '';

    return this.steps.map(step => {
      const contentPreview = step.content.length > 100
        ? step.content.substring(0, 100) + '...'
        : step.content;
      return `Step ${step.step}: ${contentPreview}`;
    }).join('\n');
  }
}

// Global instance
let thinkingComponent = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  thinkingComponent = new ThinkingComponent('thinking-container');
});

/**
 * Handle thinking event (global function)
 * @param {Object} event - Event object
 */
function handleThinkingEvent(event) {
  if (!thinkingComponent) {
    thinkingComponent = new ThinkingComponent('thinking-container');
  }
  thinkingComponent.handleEvent(event);
}

/**
 * Toggle thinking content display (global function)
 * @param {HTMLElement} header - Header element
 */
function toggleThinking(header) {
  if (thinkingComponent) {
    thinkingComponent.toggle();
  }
}
