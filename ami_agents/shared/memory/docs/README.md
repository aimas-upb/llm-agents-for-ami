# Claude Code Python Project Setup

This is a simple, concrete setup for Python projects using Claude Code with effective project tracking.

## Project Structure

```
your-project/
├── CLAUDE.md                    # Main configuration for Claude Code
├── .claude/
│   ├── commands/               # Custom slash commands
│   │   ├── review-status.md   # Review project state
│   │   ├── update-status.md   # Update current status
│   │   ├── log-progress.md    # Log completed work
│   │   └── update-roadmap.md  # Update roadmap tasks
│   └── settings.json          # Project settings (optional)
├── docs/
│   ├── STATUS.md              # Current project state
│   ├── ROADMAP.md             # Features and todos  
│   └── IMPLEMENTATION_LOG.md  # Implementation history
├── src/                       # Your Python source code
├── tests/                     # Your test files
└── requirements.txt           # Python dependencies
```

## Key Features

1. **NO EMOJIS** - Strict rule enforced across all project files
2. **Python Best Practices** - PEP 8, type hints, docstrings, testing
3. **Simple Tracking** - Markdown-based status tracking
4. **Custom Commands** - Easy-to-use slash commands for project management

## Quick Start

### 1. Copy Files to Your Project

Copy these files to your project root:
- `CLAUDE.md`
- `.claude/` directory (with all commands)
- `docs/` directory (with all tracking files)

### 2. Initialize Your Project

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Claude Code
claude
```

### 3. Start Using Commands

Inside Claude Code, you can now use:

```bash
/review-status      # Get overview of project state
/update-status      # Update current status
/log-progress       # Log completed work
/update-roadmap     # Update roadmap tasks
```

## Tracking Files Explained

### STATUS.md
**Purpose:** Current snapshot of the project

**Contains:**
- Current state and active work
- Any blockers
- Recent changes (last 5)
- Next immediate steps

**When to update:** Whenever you start new work or complete a task

### ROADMAP.md
**Purpose:** High-level feature planning and todos

**Contains:**
- Features organized by priority (High/Medium/Low)
- Task status: [ ] todo, [IN PROGRESS] active, [DONE] completed
- Dates for started and completed tasks

**When to update:** When planning new features or completing tasks

### IMPLEMENTATION_LOG.md
**Purpose:** Detailed history of what was built

**Contains:**
- Date of implementation
- Files modified
- Description of changes
- Important notes and decisions

**When to update:** After completing significant work

## Workflow Example

### Starting Your Day

1. Open Claude Code in your project
2. Run `/review-status` to see what needs to be done
3. Pick a task from the roadmap
4. Run `/update-roadmap` to mark it as [IN PROGRESS]

### During Development

1. Write code following Python best practices (see CLAUDE.md)
2. Write tests for your code
3. Run tests: `pytest tests/`
4. Commit changes: `git commit -m "add: feature description"`

### Finishing a Task

1. Run `/log-progress` to document what you built
2. Run `/update-roadmap` to mark task as [DONE]
3. Run `/update-status` to update current state
4. Push your changes: `git push`

## Custom Commands

### /review-status
**What it does:** Reads all tracking files and gives you a summary

**Use when:** 
- Starting a coding session
- Need to understand project state
- Want to see what's next

**Example:**
```
> /review-status

PROJECT STATUS REVIEW
====================

CURRENT STATE:
Working on SVM classifier implementation

ACTIVE WORK:
- Implementing cross-validation
- Adding performance metrics

NEXT PRIORITIES:
1. Complete SVM implementation
2. Add hyperparameter tuning
3. Create visualization tools
```

### /update-status
**What it does:** Updates STATUS.md with current information

**Use when:**
- Starting new work
- Encountering blockers
- Completing tasks

**Example:**
```
> /update-status

What updates should I make to STATUS.md?
- What work is currently active?
> Working on gradient boosting implementation

- Any new blockers?  
> None

- What should the next steps be?
> 1. Test with real dataset
> 2. Compare with baseline
> 3. Document results
```

### /log-progress
**What it does:** Adds entry to IMPLEMENTATION_LOG.md

**Use when:**
- Completed a feature
- Made significant changes
- Want to document decisions

**Example:**
```
> /log-progress

What work was completed?

1. Feature/task name:
> SVM classifier with RBF kernel

2. Files created or modified:
> src/models/svm_classifier.py
> tests/test_svm_classifier.py

3. Brief description:
> Implemented SVM with RBF kernel, added cross-validation,
> included grid search for C and gamma parameters

4. Any important notes:
> Used scikit-learn's SVC. Set random_state=42 for 
> reproducibility. Need to add feature scaling in future.
```

### /update-roadmap
**What it does:** Updates task status in ROADMAP.md

**Use when:**
- Starting a task
- Completing a task
- Adding new tasks

**Example:**
```
> /update-roadmap

What would you like to update?
1. Mark task as started
2. Mark task as completed
3. Add new task

> 2

Which task was completed?
> Implement SVM classifier

[Updates roadmap with completion date]
```

## Python Coding Guidelines

### Code Style
- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Max line length: 88 characters
- No emojis anywhere!

### Testing
- Use pytest
- Test file naming: `test_<module>.py`
- Write tests for new functionality
- Include edge cases

### Documentation
Every function should have a docstring:

```python
def train_model(X: np.ndarray, y: np.ndarray) -> SVMClassifier:
    """Train SVM classifier on provided data.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target labels of shape (n_samples,)
        
    Returns:
        Trained SVM classifier instance
        
    Raises:
        ValueError: If X and y have incompatible shapes
    """
    # Implementation
```

### Error Handling
```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise
```

## Configuration Details

### CLAUDE.md
Main configuration file that Claude Code automatically reads. Contains:
- Python coding standards
- Project structure
- Development workflow
- What NOT to do (including NO EMOJIS rule)

### settings.json (Optional)
Project-specific settings:
- Model selection
- Permission controls
- Tool restrictions

## Tips

1. **Use /review-status frequently** - Start each session with it
2. **Keep STATUS.md current** - Update it when things change
3. **Log significant work** - Use /log-progress for important implementations
4. **Update roadmap regularly** - Keep it as your source of truth
5. **Follow Python best practices** - They're in CLAUDE.md for a reason
6. **Write tests** - They'll save you time debugging
7. **Commit often** - Small, focused commits are easier to understand

## Customization

### Adding New Commands
Create a new `.md` file in `.claude/commands/`:

```markdown
# /my-command

Description of what the command does.

## Usage
How to use it

## Process
Steps to execute
```

### Modifying Tracking Files
Feel free to adjust the format of STATUS.md, ROADMAP.md, or IMPLEMENTATION_LOG.md to fit your needs. Just update the commands accordingly.

### Python Configuration
Update CLAUDE.md to match your:
- Python version
- Package manager (pip/uv/poetry)
- Testing framework
- Code style preferences

## Troubleshooting

### Commands not showing up?
- Make sure files are in `.claude/commands/` directory
- Files must end in `.md`
- Restart Claude Code

### Claude not following Python rules?
- Check CLAUDE.md is in project root
- Verify the guidelines are clear
- Remind Claude to check CLAUDE.md

### Tracking files getting messy?
- Keep STATUS.md to last 5 recent changes
- Archive completed roadmap items
- Use clear, consistent formatting

## What Makes This Setup Good

1. **Simple** - Just markdown files, no complex tooling
2. **Concrete** - Clear structure and examples
3. **Not Over-engineered** - No unnecessary complexity
4. **Easy to Maintain** - Plain text files you can edit
5. **Effective Tracking** - Know what was done and what's next
6. **Python-focused** - Tailored for Python ML projects
7. **NO EMOJIS** - Professional and clean

## Credits

This setup is based on best practices from:
- Claude Code official documentation
- awesome-claude-code GitHub repository
- Real-world Python project structures
- Community feedback on effective tracking systems

## License

Use this setup however you want. No attribution needed.

---

**Remember: NO EMOJIS ANYWHERE!**
