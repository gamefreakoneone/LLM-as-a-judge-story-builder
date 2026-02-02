# LLM-as-a-Judge Story Builder 

A multi-agent storytelling system that creates **safe, engaging bedtime stories** for children ages 5-10. Built with GPT-3.5-turbo, this project demonstrates advanced LLM orchestration through specialized agents, best-of-N selection, iterative refinement, and safety-first content filtering.

> **Built for Hippocratic AI** - Showcasing production-ready AI safety and quality control patterns

---

##Key Features

### Safety-First Architecture
- **Content Transformation Engine**: Automatically converts inappropriate requests into safe, cozy alternatives
- **Multi-Layer Safety Checks**: 
  - Deterministic preflight validation (word count, taboo words, sleep cues)
  - LLM judge evaluation across 5 quality dimensions
  - Iterative refinement until safety thresholds are met
- **Hard-Coded Safety Guardrails**: Enforced blocklists for violence, scary content, and inappropriate themes

### Quality Optimization
- **Best-of-N Selection**: Generates multiple story candidates and selects the highest-quality version
- **Specialized Agent Pipeline**:
  - **Brief Builder**: Normalizes and safety-transforms user requests
  - **Story Planner**: Creates structured 3-act outlines
  - **Story Generator**: Writes engaging, age-appropriate narratives
  - **LLM Judge**: Evaluates quality across multiple dimensions
  - **Refiner**: Iteratively improves stories based on feedback

### Production-Ready Design
- **Robust JSON Parsing**: Handles markdown code blocks, trailing commas, and malformed responses
- **Fallback Mechanisms**: Graceful degradation when parsing fails
- **Temperature Variation**: Optimized for each pipeline stage (0.3 for extraction → 0.8 for creative writing)
- **Token Efficiency**: Quick scoring for candidate ranking, full evaluation only for final selection

---

## 📐 System Architecture

The system follows a **multi-agent pipeline** with quality gates at each stage:

![System Architecture](Resources/system%20architecture.png)

### Pipeline Stages

1. **USER INPUT** → Brief Builder
2. **BRIEF BUILDER** → Transforms requests into safety-constrained specifications
3. **STORY PLANNER** → Creates 3-act narrative structure
4. **BEST-OF-N STORY GENERATOR** → Generates multiple candidates with temperature variation
5. **PREFLIGHT CHECK** → Deterministic validation (word count, sleep cues, taboo words)
   - **Pass** → Judge evaluation
   - **Fail** → Quick Fix → Retry preflight (max 2 attempts)
6. **JUDGE** → LLM evaluates across 5 dimensions
   - **Pass** (overall ≥ 7.5 AND bedtime ≥ 7) → Final Story
   - **Fail** → Refiner → Judge (max 2-3 iterations)
7. **FINAL STORY** → Ready for bedtime reading!

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gamefreakoneone/LLM-as-a-judge-story-builder.git
   cd LLM-as-a-judge-story-builder
   ```

2. **Install dependencies**
   ```bash
   pip install openai python-dotenv
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

### Usage

**Interactive Mode** (recommended)
```bash
python main.py
```

**Programmatic Usage**
```python
from main import generate_bedtime_story, print_story

# Generate a story
story, evaluation = generate_bedtime_story(
    "A brave little robot who learns about friendship",
    max_iterations=3,
    verbose=True
)

# Display results
print_story(story, evaluation)
```

### Example Prompts
- "A girl named Alice and her cat friend Bob"
- "A magical garden where vegetables come alive"
- "A curious fox who discovers a hidden library"

---

## 🧪 Testing & Validation

Run the comprehensive test suite:
```bash
python test.py
```

The test file (`test.py`) includes the full pipeline implementation with **Best-of-N selection** enabled by default.

---

## 📊 Quality Evaluation Criteria

Stories are judged across **5 dimensions** (1-10 scale):

| Dimension | Description |
|-----------|-------------|
| **Age Appropriateness** | Simple vocabulary, safe content, 5-year-old comprehension |
| **Engagement** | Likable characters, vivid details, maintains attention |
| **Story Structure** | Clear beginning/middle/end, satisfying arc |
| **Bedtime Suitability** | Calming ending, sleep cues (yawning, stars, moon, cozy bed) |
| **Moral Clarity** | Gentle lesson, natural integration (not preachy) |

**Passing Criteria**: `overall_score ≥ 7.5` AND `bedtime_suitability ≥ 7`

---

## 🛡️ Safety Mechanisms

### 1. Request Transformation
```
User: "Scary monster story" 
→ Brief: "Friendly creature who looks different"
```

### 2. Preflight Validation
- **Word Count**: ±25% of target (400-600 words)
- **Sleep Cues**: Minimum 2 in last 100 words (`yawn`, `stars`, `moon`, `cozy`, etc.)
- **Taboo Words**: Zero-tolerance blocklist (`kill`, `weapon`, `nightmare`, etc.)

### 3. Hard-Coded Constraints
Enforced in `StoryBrief.from_dict()`:
```python
must_avoid = ["violence", "gore", "scary monsters", "death", 
              "profanity", "weapons", "kidnapping", "abuse"]
```

---

## ⚙️ Configuration

### Model Settings
```python
MODEL_NAME = "gpt-3.5-turbo"  # Required per assignment
N_CANDIDATES = 2              # Best-of-N candidates (2-3 recommended)
```

### Temperature Strategy
- **Brief Builder**: 0.3 (consistent extraction)
- **Story Planner**: 0.6 (moderate creativity)
- **Story Generator**: 0.75-0.95 (high creativity, varied across candidates)
- **Judge**: 0.2 (consistent scoring)
- **Refiner**: 0.7 (balanced improvement)

---

## 🔍 Advanced Features

### Best-of-N Selection
Generates multiple candidates with temperature variation, ranks using:
1. **Preflight penalties**: -1.5 points per issue
2. **Quick judge score**: Abbreviated evaluation (100 tokens vs 600)
3. **Selection**: Highest-scoring candidate proceeds to full judge

### Iterative Refinement Loop
```
Generate → Preflight (max 2 fixes) → Judge → [if fail] Refine → Judge (max 2-3 iterations)
```
Tracks best version across iterations to prevent quality regression.

---

## 📝 Future Enhancements

If I had **more time**, I would add:

1. **Interactive feedback loop**: Let users say "make it funnier" or "add a dragon" post-generation
2. **Style templates**: "Dr. Seuss rhyme", "Aesop fable" with few-shot examples
3. **Text-to-speech integration**: Add SSML markers for read-aloud pacing
4. **Illustrations**: Use Google's Nanobanana to generate illustrations for the story
5. **Conversation memory**: Story series with recurring characters

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

