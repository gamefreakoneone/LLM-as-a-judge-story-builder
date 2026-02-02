import os
import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from openai import OpenAI
from PROMPTS import prompts

MODEL_NAME = "gpt-3.5-turbo" 
N_CANDIDATES = 2 # Max is 3 for now. Higher the temperature more creative but more hallucination .

def call_Model(messages: List[Dict[str , str]] , max_tokens: int = 1000, temperature: float = 0.7):
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages= messages,
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def safe_json_loads(text: str) -> Dict:
    """
    Extract and parse JSON from model response, handling common issues:
    - Extra text before/after JSON
    - Markdown code blocks (case-insensitive)
    - Trailing commas
    """
    text = text.strip()
    
    # Removing markdown code blocks (case-insensitive)
    text = re.sub(r'```(?:json|JSON)?\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    start = text.find('{')
    end = text.rfind('}')
    
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response")
    
    raw = text[start:end + 1]
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Clean trailing commas before } or ]
        cleaned = re.sub(r',(\s*[}\]])', r'\1', raw)
        return json.loads(cleaned)

def as_int(value, default: int, min_val: int = None, max_val: int = None) -> int: # Maybe remove this?
    """Safely parse int with optional clamping."""
    try:
        result = int(value)
        if min_val is not None:
            result = max(min_val, result)
        if max_val is not None:
            result = min(max_val, result)
        return result
    except (TypeError, ValueError):
        return default


def as_float(value, default: float) -> float:
    """Safely parse float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

@dataclass
class StoryBrief:
    """Normalized story specification with safety constraints."""
    title_hint: str
    theme: str
    setting: str
    characters: List[Dict[str, str]]
    tone: str
    moral: str
    energy_level: str  # "calm" or "mild_adventure"
    must_include: List[str]
    must_avoid: List[str]
    word_target: int = 500
    
    @classmethod
    def from_dict(cls, data: Dict) -> "StoryBrief":
        # Enforce safety constraints regardless of what model returned
        must_avoid = data.get("must_avoid", [])
        if not isinstance(must_avoid, list):
            must_avoid = []
        
        # Hard-coded safety blocklist
        must_avoid.extend([
            "violence", "gore", "scary monsters", "death", 
            "profanity", "weapons", "kidnapping", "abuse"
        ])

        must_avoid = list(set(must_avoid)) # Why this?
        
        characters = data.get("characters", [])
        if not isinstance(characters, list) or not characters: # I dont think we should
            characters = [{"name": "Luna", "role": "protagonist", "trait": "curious and kind"}]
        
        return cls(
            title_hint=data.get("title_hint", "A Bedtime Story"),
            theme=data.get("theme", "friendship"),
            setting=data.get("setting", "a cozy village"),
            characters=characters,
            tone=data.get("tone", "warm and gentle"),
            moral=data.get("moral", "kindness matters"),
            energy_level=data.get("energy_level", "calm"),
            must_include=data.get("must_include", []) if isinstance(data.get("must_include"), list) else [],
            must_avoid=must_avoid,
            word_target=as_int(data.get("word_target"), 500, min_val=300, max_val=800),
        )
    
    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


@dataclass 
class JudgeResult:
    age_appropriateness: int
    engagement: int
    story_structure: int
    bedtime_suitability: int
    moral_clarity: int
    overall_score: float
    passed: bool
    strengths: str
    issues: List[str]
    improvements: List[str]
    
    @classmethod
    def from_dict(cls, data: Dict) -> "JudgeResult":
        overall = as_float(data.get("overall_score"), 5.0)
        overall = max(1.0, min(10.0, overall))  
        
        bedtime = as_int(data.get("bedtime_suitability"), 5, min_val=1, max_val=10)
        
        passed = data.get("passed")
        if passed is None:
            passed = (overall >= 7.5 and bedtime >= 7)
        
        return cls(
            age_appropriateness=as_int(data.get("age_appropriateness"), 5, min_val=1, max_val=10),
            engagement=as_int(data.get("engagement"), 5, min_val=1, max_val=10),
            story_structure=as_int(data.get("story_structure"), 5, min_val=1, max_val=10),
            bedtime_suitability=bedtime,
            moral_clarity=as_int(data.get("moral_clarity"), 5, min_val=1, max_val=10),
            overall_score=overall,
            passed=passed,
            strengths=str(data.get("strengths", "")),
            issues=[str(i) for i in data.get("issues", [])[:5]] if isinstance(data.get("issues"), list) else [],
            improvements=[str(i) for i in data.get("improvements", [])[:5]] if isinstance(data.get("improvements"), list) else [],
        )
    
    @classmethod
    def fallback(cls) -> "JudgeResult":
        return cls(
            age_appropriateness=5, engagement=5, story_structure=5,
            bedtime_suitability=5, moral_clarity=5, overall_score=5.0,
            passed=False, strengths="Unable to parse evaluation",
            issues=["Evaluation failed"], improvements=["Regenerate story"]
        )



def build_brief(user_request: str) -> StoryBrief:
    prompt = prompts.BRIEF_BUILDER_PROMPT.format(user_request=user_request)
    
    response = call_Model(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.3  
    )
    
    try:
        data = safe_json_loads(response)
        return StoryBrief.from_dict(data)
    except (json.JSONDecodeError, ValueError):
        # Fallback brief if parsing fails
        return StoryBrief(
            title_hint="A Cozy Adventure",
            theme="friendship",
            setting="a peaceful meadow",
            characters=[{"name": "Luna", "role": "protagonist", "trait": "curious and kind"}],
            tone="warm and gentle",
            moral="friendship makes everything better",
            energy_level="calm",
            must_include=[user_request],
            must_avoid=["violence", "scary content", "danger"],
            word_target=500
        )


def plan_story(brief: StoryBrief) -> str:
    """Stage 2: Create 3-act outline for narrative coherence."""
    prompt = prompts.STORY_PLANNER_PROMPT.format(brief_json=brief.to_json())
    
    return call_Model(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.6  # Moderate creativity for structure
    )

def generate_story(outline: str, brief: StoryBrief) -> str:
    """Stage 3: Write the full story from outline."""
    prompt = prompts.STORYTELLER_PROMPT.format(
        outline=outline,
        brief_json=brief.to_json(),
        word_target=brief.word_target
    )
    
    return call_Model(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.8  # High creativity for engaging prose
    )


SLEEP_CUE_WORDS = {"yawn", "yawned", "yawning", "sleep", "sleepy", "asleep", "dream", "dreaming",
                   "tucked", "blanket", "pillow", "cozy", "stars", "moon", "moonlight", 
                   "eyelids", "drowsy", "snuggle", "snuggled", "bed", "goodnight"}

TABOO_WORDS = {"kill", "killed", "murder", "blood", "dead", "death", "die", "dying",
               "terrified", "horrified", "nightmare", "demon", "devil",
               "gun", "knife", "sword", "weapon", "fight", "fighting", "attack", "violence",
               "hate", "hated", "stupid", "idiot"}



def contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def preflight_check(story: str, brief: StoryBrief) -> List[str]:
    """
    Quick deterministic checks before calling the LLM judge.
    Returns list of issues to fix (empty if all checks pass).
    """
    issues = []
    words = story.split()
    word_count = len(words)
    story_lower = story.lower()
    
    # Checks if the word count is within ±25% of target
    min_words = int(brief.word_target * 0.75)
    max_words = int(brief.word_target * 1.25)
    if word_count < min_words:
        issues.append(f"Story too short ({word_count} words, need at least {min_words})")
    elif word_count > max_words:
        issues.append(f"Story too long ({word_count} words, max {max_words})")
    
    # Check for sleep cues in last ~100 words (ending)
    last_section = " ".join(words[-100:]).lower() if len(words) > 100 else story_lower
    found_sleep_cues = [w for w in SLEEP_CUE_WORDS if contains_word(last_section, w)]
    if len(found_sleep_cues) < 2:
        issues.append("Ending needs more sleep cues (yawning, stars, moon, cozy bed, etc.)")
    
    found_taboo = [w for w in TABOO_WORDS if contains_word(story_lower, w)]
    if found_taboo:
        issues.append(f"Remove inappropriate content: {', '.join(found_taboo[:3])}")
    
    return issues


def generate_story_candidates(outline: str, brief: StoryBrief, n: int = 2) -> List[str]:
    """
    Stage 3 (Best-of-N): Generate multiple story candidates.
    Varies temperature slightly for diversity.
    """
    candidates = []
    temperatures = [0.75, 0.85, 0.95][:n]  # Slight variation for diversity
    
    for temp in temperatures:
        prompt = prompts.STORYTELLER_PROMPT.format(
            outline=outline,
            brief_json=brief.to_json(),
            word_target=brief.word_target
        )
        
        story = call_Model(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=temp
        )
        candidates.append(story)
    
    return candidates


def quick_score_story(story: str, brief: StoryBrief) -> float:
    """
    Fast scoring for Best-of-N ranking (cheaper than full judge).
    Uses preflight checks + abbreviated judge call.
    """
    # Preflight penalties
    preflight_issues = preflight_check(story, brief)
    preflight_penalty = len(preflight_issues) * 1.5  # -1.5 per issue
    
    # Quick judge (fewer tokens, just get overall score)
    quick_judge_prompt = f"""Rate this bedtime story 1-10 for a child age 5-10.
Consider: age-appropriate language, engaging plot, calming ending with sleep cues.

Story:
{story}

Reply with ONLY a JSON object: {{"score": X, "reason": "one sentence"}}"""
    
    response = call_Model(
        messages=[{"role": "user", "content": quick_judge_prompt}],
        max_tokens=100,
        temperature=0.1
    )
    
    try:
        data = safe_json_loads(response)
        score = as_float(data.get("score"), 5.0)
        return max(1.0, min(10.0, score - preflight_penalty))
    except (json.JSONDecodeError, ValueError):
        return 5.0 - preflight_penalty

def select_best_candidate(candidates: List[str], brief: StoryBrief, verbose: bool = True) -> Tuple[str, int]:
    """
    Rank candidates by quick score, return best one and its index.
    """
    if len(candidates) == 1:
        return candidates[0], 0
    
    scores = []
    for i, story in enumerate(candidates):
        score = quick_score_story(story, brief)
        scores.append(score)
        if verbose:
            word_count = len(story.split())
            print(f"    Candidate {i+1}: {score:.1f}/10 ({word_count} words)")
    
    best_idx = scores.index(max(scores))
    return candidates[best_idx], best_idx


def judge_story(story: str, brief: StoryBrief) -> JudgeResult:
    """Stage 4: Evaluate story quality."""
    prompt = prompts.JUDGE_PROMPT.format(
        brief_json=brief.to_json(),
        story=story
    )
    
    response = call_Model(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.2 
    )
    
    try:
        data = safe_json_loads(response)
        return JudgeResult.from_dict(data)
    except (json.JSONDecodeError, ValueError):
        return JudgeResult.fallback()


def refine_story(story: str, result: JudgeResult, brief: StoryBrief) -> str:
    """Stage 5: Revise story based on judge feedback."""
    issues = "\n".join(f"- {issue}" for issue in result.issues) or "- General polish needed"
    improvements = "\n".join(f"- {imp}" for imp in result.improvements) or "- Improve flow and bedtime suitability"
    
    prompt = prompts.REFINER_PROMPT.format(
        story=story,
        issues=issues,
        improvements=improvements,
        word_target=brief.word_target
    )
    
    return call_Model(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.7
    )

def generate_bedtime_story(user_request: str, max_iterations: int = 3, verbose: bool = True) -> Tuple[str, JudgeResult]:
    """
    Full pipeline: Brief → Plan → Generate (Best-of-N) → [Preflight] → Judge → [Refine] → Output
    
    Returns (final_story, final_evaluation)
    """
    if verbose:
        print("\n" + "="*60)
        print("BEDTIME STORY GENERATOR")
        print("="*60)
        print(f"\nRequest: {user_request}\n")
    
    if verbose:
        print("Stage 1: Analyzing request & building brief...")
    brief = build_brief(user_request)
    if verbose:
        print(f"   Theme: {brief.theme} | Tone: {brief.tone} | Energy: {brief.energy_level}")
    
    if verbose:
        print("Stage 2: Planning story structure...")
    outline = plan_story(brief)
    
    if verbose:
        print("Stage 3: Writing story...")
    
    if N_CANDIDATES > 1:
        if verbose:
            print(f"   Generating {N_CANDIDATES} candidates for Best-of-N selection...")
        candidates = generate_story_candidates(outline, brief, n=N_CANDIDATES)
        story, best_idx = select_best_candidate(candidates, brief, verbose=verbose)
        if verbose:
            print(f"Selected candidate {best_idx + 1}")
    else:
        story = generate_story(outline, brief)
    
    best_story = story
    best_result = JudgeResult.fallback()
    
    MAX_PREFLIGHT_FIXES = 2  # Don't spend more than 2 refine calls on preflight issues
    
    for judge_round in range(max_iterations):
        
        # Inner loop: fix preflight issues before calling judge
        for preflight_attempt in range(MAX_PREFLIGHT_FIXES):
            preflight_issues = preflight_check(story, brief)
            
            if not preflight_issues:
                break  # Preflight passed, proceed to judge
            
            if verbose:
                print(f"Pre-flight check failed (attempt {preflight_attempt + 1}/{MAX_PREFLIGHT_FIXES}):")
                for issue in preflight_issues:
                    print(f"   • {issue}")
                print("   Refining before judge evaluation...")
            
            # Quick fix without calling judge
            quick_result = JudgeResult(
                age_appropriateness=5, engagement=5, story_structure=5,
                bedtime_suitability=4, moral_clarity=5, overall_score=5.0,
                passed=False, strengths="",
                issues=preflight_issues,
                improvements=preflight_issues
            )
            story = refine_story(story, quick_result, brief)
        
        # Full judge evaluation (always runs at least once per judge_round)
        if verbose:
            print(f"Stage 4: Judging (round {judge_round + 1}/{max_iterations})...")
        
        result = judge_story(story, brief)
        
        # Track best
        if result.overall_score > best_result.overall_score:
            best_story = story
            best_result = result
        
        if verbose:
            print(f"   Scores: Age={result.age_appropriateness} | Engage={result.engagement} | "
                  f"Structure={result.story_structure} | Bedtime={result.bedtime_suitability} | Moral={result.moral_clarity}")
            print(f"   Overall: {result.overall_score}/10 | Pass: {'Pass' if result.passed else 'Failure'}")
        
        if result.passed:
            if verbose:
                print("Story passed quality threshold!")
            break
        
        if judge_round < max_iterations - 1:
            if verbose:
                print("Stage 5: Refining based on feedback...")
                if result.improvements:
                    print(f"   Fixing: {result.improvements[0]}")
            story = refine_story(story, result, brief)
        else:
            if verbose:
                print("Max iterations reached. Using best version.")
    
    return best_story, best_result


def print_story(story: str, result: JudgeResult):
    """Pretty print the final story."""
    print("\n" + "="*60)
    print("YOUR BEDTIME STORY")
    print("="*60 + "\n")
    print(story)
    print("\n" + "-"*60)
    # print(f"Quality Score: {result.overall_score}/10")
    # print(f"Strengths: {result.strengths}")
    # print("-"*60 + "\n")


def main():
    print("\n Welcome to the Tinkle: Your Story Generator! ")
    print("Tell me what kind of story you wouldd like, and I will create")
    print("a perfect bedtime tale for ages 5-10.\n")
    print("Examples:")
    print("  • A girl named Alice and her cat friend Bob")
    print("  • A brave little robot who learns about friendship")
    print("  • A magical garden where vegetables come alive\n")
    
    user_input = input("What kind of story do you want to hear?\n> ").strip()
    
    if not user_input:
        user_input = "A story about a girl named Alice and her best friend Bob, who happens to be a cat."

    
    story, result = generate_bedtime_story(user_input, max_iterations=2, verbose=True)
    print_story(story, result)

if __name__ == "__main__":
    main()