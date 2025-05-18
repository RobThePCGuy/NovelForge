import os
import argparse
import yaml
import logging
import time
import random
import re  # For parsing (though minimized)
import traceback
import json  # Added for load_novel_data

# Pydantic for structured output/validation
from pydantic import BaseModel, Field, ValidationError

# Langchain & Langgraph Core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

# from langchain_core.output_parsers.json import SimpleJsonOutputParser # Not explicitly used, PydanticOutputParser is used for JSON
from langchain.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Ollama Integration
from langchain_ollama import ChatOllama

# Vector Store Integration (Optional but recommended for long context)
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    VECTOR_STORE_ENABLED = True
except ImportError:
    print(
        "Warning: Vector Store libraries (chromadb, sentence-transformers) not found."
    )
    print(
        "Context management will be limited. Run 'pip install chromadb sentence-transformers tiktoken'"
    )
    VECTOR_STORE_ENABLED = False

from typing import TypedDict, List, Optional, Dict, Any  # Keep this, it's used

# --- Configuration Loading ---
DEFAULT_CONFIG_PATH = "config.yaml"


def load_config(config_path=DEFAULT_CONFIG_PATH):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded successfully from {config_path}")
        # Basic validation (can be expanded)
        if not all(k in config for k in ["llm_settings", "workflow_settings"]):
            raise ValueError("Config file missing required top-level keys.")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}. Exiting.")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        exit(1)


# --- Logging Setup ---
# Configure logging to write to file and console
log_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("NovelWriter")
logger.setLevel(logging.INFO)  # Set base level

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# File Handler (will be added after config load to get output dir)
file_handler = None


def setup_file_logging(log_dir):
    global file_handler
    if file_handler:  # Remove existing handler if re-configuring
        logger.removeHandler(file_handler)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "novel_generation.log")
    file_handler = logging.FileHandler(log_file, mode="a")  # Append mode
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info(f"File logging setup at: {log_file}")


# --- Default Parameter Data (Hardcoded Fallbacks) ---
# These lists are used if novel_data.json is not found or incomplete.
# I. GENRES
GENRES = [
    "Literary Fiction",
    "Contemporary Fiction",
    "Autofiction",
    "Women's Fiction",
    "Action/Adventure",
    "Heroic",
    "Lost World",
    "Sword-and-Sandal",
    "Spy/Espionage",
    "Military",
    "Survival",
    "Treasure Hunting/Archaeological",
    "Pirate",
    "Quest",
    "Superhero",
    "High/Epic Fantasy",
    "Low Fantasy",
    "Urban Fantasy",
    "Dark Fantasy/Grimdark",
    "Cozy Fantasy",
    "Historical Fantasy",
    "Magical Realism",
    "Contemporary Fantasy",
    "Sword & Sorcery",
    "Portal Fantasy",
    "Mythic Fantasy",
    "Fairy Tale Retellings",
    "Wuxia",
    "Anthropomorphic Fantasy",
    "Romantic Fantasy (Romantasy)",
    "Science Fantasy",
    "Hard SF",
    "Soft SF",
    "Space Opera",
    "Cyberpunk",
    "Steampunk",
    "Biopunk",
    "Nanopunk",
    "Dystopian",
    "Utopian",
    "Post-Apocalyptic/Apocalyptic",
    "Alternate History",
    "Time Travel",
    "Military SF",
    "Alien Contact/Invasion",
    "Climate Fiction (Cli-Fi)",
    "Social SF",
    "Space Western",
    "Tech Noir",
    "Psychological Horror",
    "Supernatural/Paranormal Horror",
    "Gothic Horror",
    "Body Horror",
    "Cosmic/Lovecraftian Horror",
    "Folk Horror",
    "Monster Literature",
    "Slasher",
    "Splatterpunk",
    "Quiet Horror",
    "Occult Horror",
    "Domestic Horror",
    "Found Footage",
    "J-Horror/K-Horror",
    "Techno-Horror",
    "Cozy Mystery",
    "Police Procedural",
    "Hard-Boiled",
    "Historical Mystery",
    "Amateur Sleuth",
    "Locked-Room Mystery",
    "Howdunnit (Inverted Mystery)",
    "Noir",
    "Supernatural Mystery",
    "Caper",
    "Detective Fiction",
    "Psychological Thriller",
    "Legal Thriller",
    "Political Thriller",
    "Espionage/Spy Thriller",
    "Medical Thriller",
    "Techno-Thriller",
    "Conspiracy Thriller",
    "Supernatural Thriller",
    "Historical Thriller",
    "Domestic Thriller",
    "Serial Killer Thriller",
    "Military Thriller",
    "Heist Thriller",
    "Financial Thriller",
    "Environmental Thriller",
    "Survival Thriller",
    "Disaster Thriller",
    "Action Thriller",
    "Religious Thriller",
    "Forensic Thriller",
    "Dark Academia",
    "Contemporary Romance",
    "Historical Romance",
    "Paranormal Romance",
    "Fantasy Romance (Romantasy)",
    "Sci-Fi Romance",
    "Erotic Romance",
    "Romantic Suspense",
    "Gothic Romance",
    "Romantic Comedy (Rom-Com)",
    "Christian Romance",
    "LGBTQ+ Romance",
    "New Adult Romance",
    "Young Adult Romance",
    "Sports Romance",
    "Small-Town Romance",
    "Holiday Romance",
    "Mail-Order Bride Romance",
    "Billionaire Romance",
    "Western",
    "Historical Fiction",
    "Chick Lit",
    "Christian Fiction",
    "Satire",
    "Graphic Novel",
    "Young Adult (YA)",
    "New Adult (NA)",
    "Middle Grade (MG)",
    "Children's Fiction",
]
# II. SETTINGS
SETTINGS = [
    "Remote mountain observatory",
    "Dense, unexplored rainforest",
    "Sun-scorched desert ruins",
    "Arctic tundra research station",
    "Volcanic island chain",
    "Misty, haunted swamp/bayou",
    "Floating islands in the sky",
    "Subterranean cave network",
    "Ancient, whispering forest",
    "Neon-drenched cyberpunk metropolis",
    "Cobbled streets of a gaslit Victorian city",
    "Bustling medieval market town",
    "Sprawling futuristic spaceport",
    "Post-apocalyptic city ruins",
    "Decaying industrial district",
    "Opulent high-society district",
    "Underground resistance tunnels",
    "University campus with hidden libraries",
    "Quiet, eerie suburban neighborhood",
    "Isolated farmhouse miles from anywhere",
    "Self-sufficient ranch on the frontier",
    "Sleepy fishing village with dark secrets",
    "Forgotten mining town",
    "Secluded monastery/abbey",
    "Homestead in a newly colonized territory",
    "Crumbling gothic castle",
    "High-tech underground bunker",
    "Abandoned asylum with a troubled past",
    "Grand library containing forbidden knowledge",
    "Mysterious lighthouse on a stormy coast",
    "Ornate theater hiding secret passages",
    "Overgrown greenhouse with strange flora",
    "Carnival funhouse after dark",
    "Ancient temple complex",
    "Haunted Victorian mansion",
    "Generation ship centuries into its journey",
    "Luxury cruise liner in troubled waters",
    "Long-haul freighter spaceship",
    "Steampunk airship exploring new continents",
    "Transcontinental express train",
    "Deep-sea exploration submarine",
    "Alien planet with bizarre biology",
    "Orbiting space station overlooking Earth",
    "City built inside a giant crystal",
    "Magical realm accessible only through portals",
    "Virtual reality simulation",
    "Dyson sphere surrounding a star",
    "Underwater city facing collapse",
    "Alternate history timeline (e.g., Roman Empire with magic)",
    "World where dreams manifest physically",
    "War-torn battlefield",
    "Technologically advanced utopia (with cracks showing)",
    "Oppressive surveillance state",
    "Magically saturated environment",
    "Zone devastated by ecological disaster",
    "Festive, chaotic celebration",
    "Place where time flows differently",
    "Zone of perpetual storm/fog",
]
# III. TARGET AUDIENCES
TARGET_AUDIENCES = [
    "Young Adults (YA) (12-18)",
    "New Adults (NA) (18-30)",
    "Adults (30+)",
    "Middle Grade (MG) (8-12)",
    "Readers of Epic Fantasy",
    "Readers of Hard Sci-Fi",
    "Readers of Cozy Mysteries",
    "Readers of Psychological Thrillers",
    "Readers of Historical Romance",
    "Readers of Urban Fantasy",
    "Readers of Dark Academia",
    "History enthusiasts",
    "Science and Technology buffs",
    "Gamers",
    "True Crime followers",
    "Readers seeking pure escapism",
    "Readers desiring intellectual challenges",
    "Readers looking for emotional catharsis",
    "Thrill-seekers wanting adrenaline rushes",
    "LGBTQ+ readers seeking representation",
    "Readers from specific cultural backgrounds",
    "Readers interested in social justice themes",
]
# IV. CONCEPTS
CONCEPTS = [
    "What if dinosaurs could be brought back to life in a theme park?",
    "What if a hidden magical school existed alongside the modern world?",
    "What if you could enter and manipulate people's dreams?",
    "What if consciousness could be transferred between bodies?",
    "What if memories were a tradable commodity?",
    "What if animals gained human-level intelligence and organized a revolution?",
    "What if a single day repeated endlessly for one person?",
    "An incompetent detective must solve a major crime, revealing hidden depths.",
    "A seemingly ordinary person discovers they possess extraordinary abilities.",
    "A character seeks revenge against those who wronged their family.",
    "An aging hero confronts their legacy on a final mission.",
    "Estranged siblings must reunite to fulfill a parent's dying wish.",
    "A character must choose between personal ambition and the greater good.",
    "Someone trying to escape their past finds it catching up.",
    "A race against time to find a cure for a deadly plague.",
    "The quest to retrieve a powerful magical artifact before it falls into enemy hands.",
    "Unraveling a complex conspiracy involving powerful figures.",
    "Surviving a zombie apocalypse or other societal collapse.",
    "Solving an 'impossible' locked-room murder mystery.",
    "A team plans and executes an elaborate heist.",
    "Characters must navigate a perilous journey through dangerous territory.",
    "The first human expedition to explore a bizarre, sentient alien planet.",
    "Daily life and political intrigue aboard a multi-generational starship.",
    "Students navigate rivalries and dark secrets at a magical university.",
    "Communities struggle to survive and rebuild in a world ravaged by climate change.",
    "The discovery of a portal to a parallel universe with different physical laws.",
    "Examining fate vs. free will in a world governed by prophecy.",
    "An allegorical tale exploring the corrupting influence of power.",
    "Questioning the definition of consciousness with sentient AI.",
    "Depicting the clash between tradition and necessary change.",
    "An exploration of grief and memory following a devastating loss.",
]
# V. WORLD_ELEMENTS (Dictionary)
WORLD_ELEMENTS = {
    "Hard Magic System": (
        "Magic with clearly defined rules, limitations, costs, and predictable effects (e.g., specific components needed, energy cost)."
    ),
    "Soft Magic System": (
        "Magic is mysterious, unexplained, or flexible; focuses on wonder and awe, less predictable."
    ),
    "Elemental Magic": (
        "Magic drawn from natural elements (Fire, Water, Earth, Air, Spirit, etc.) via specific rituals or bloodlines."
    ),
    "Divine Magic": (
        "Power granted by deities, spirits, or celestial beings, often involving prayer or service."
    ),
    "Psychic/Psionics": (
        "Mental powers like telekinesis, telepathy, precognition, derived from mental energy."
    ),
    "Necromancy": (
        "Magic involving the manipulation of death, souls, or the undead, often with high costs."
    ),
    "Blood Magic/Sacrifice": (
        "Magic powered by life force, blood, or sacrifice, often considered forbidden or dangerous."
    ),
    "Runic/Symbol Magic": (
        "Magic activated through inscribed runes, glyphs, or sigils on objects or surfaces."
    ),
    "Alchemy/Potion Crafting": (
        "Creating magical effects through mixing substances, brewing potions, or transmutation."
    ),
    "Magitech/Technomancy": (
        "A blend where magic and technology coexist, interact, or are used to power each other."
    ),
    "Dream Weaving": (
        "Individuals can enter, manipulate, or harvest power from the dreams of others."
    ),
    "Pre-Industrial Tech": (
        "Society operates at a level similar to Earth's Stone Age, Bronze Age, Iron Age, or Medieval period."
    ),
    "Steampunk Tech": (
        "Victorian-era aesthetics combined with advanced steam-powered machinery, clockwork devices, and airships."
    ),
    "Cyberpunk Tech": (
        "Near-future setting with advanced cybernetics, AI, ubiquitous internet, often alongside societal decay."
    ),
    "Futuristic/Space Opera Tech": (
        "Far future with interstellar travel (FTL), advanced robotics, energy weapons, terraforming."
    ),
    "Post-Apocalyptic Tech": (
        "Society uses scavenged, repurposed, or decaying technology from a more advanced past era."
    ),
    "Biopunk Tech": (
        "Focuses on advanced genetic engineering, biological manipulation, and synthetic biology."
    ),
    "Nanopunk Tech": (
        "Technology revolves around manipulation at the molecular level using nanotechnology."
    ),
    "Feudal Monarchy": (
        "Society ruled by a hereditary king/queen with power distributed through a hierarchy of nobles and vassals."
    ),
    "Corporate State": (
        "Government is controlled or heavily influenced by powerful corporations; profit is the main driver."
    ),
    "Theocracy": "Society governed by religious leaders or according to religious law.",
    "Dystopian Oligarchy": (
        "Oppressive rule by a small, elite group controlling resources and information."
    ),
    "Anarcho-Syndicalism": (
        "Society organized through decentralized, voluntary worker collectives or syndicates."
    ),
    "Caste System": (
        "Rigid social hierarchy where status is determined by birth and mobility is restricted."
    ),
    "Meritocracy (Flawed)": (
        "Society claims advancement based on merit, but systemic biases or corruption exist."
    ),
    "Unique Flora/Fauna": (
        "The world features distinctive, perhaps dangerous or symbiotic, plant and animal life."
    ),
    "Sentient AI Network": (
        "A powerful, interconnected artificial intelligence plays a major role in society (helper, ruler, threat)."
    ),
    "Ancient Alien Ruins": (
        "Remnants of a precursor alien civilization with advanced, mysterious technology dot the landscape."
    ),
    "Memory Currency": (
        "Personal memories can be extracted, stored, and traded as a form of valuable currency or information."
    ),
    "FTL Travel Gates": (
        "Ancient gates allow for faster-than-light travel between star systems, but their origins and full function are unknown."
    ),
    "Extreme Climate Conditions": (
        "The world is defined by constant extreme weather (e.g., perpetual storms, extreme heat/cold)."
    ),
}
# VI. THEMES
THEMES = [
    "Good vs. Evil",
    "Order vs. Chaos",
    "Fate vs. Free Will",
    "Nature vs. Technology",
    "Life vs. Death",
    "Knowledge vs. Ignorance",
    "Change vs. Tradition",
    "Appearance vs. Reality",
    "Love (Romantic, Familial, Platonic)",
    "Hate / Prejudice",
    "Loyalty / Fidelity",
    "Betrayal / Treachery",
    "Revenge / Vengeance",
    "Justice / Injustice",
    "Power (Desire for, Corruption by)",
    "Ambition",
    "Greed / Materialism",
    "Sacrifice / Selflessness",
    "Courage / Perseverance",
    "Cowardice / Fear",
    "Hope / Optimism",
    "Despair / Hopelessness",
    "Jealousy / Envy",
    "Forgiveness / Mercy",
    "Guilt / Atonement / Redemption",
    "Truth / Deception",
    "Human Weakness / Flawed Nature",
    "Hubris / Overconfidence",
    "Compassion / Empathy",
    "Coming of Age / Loss of Innocence",
    "Childhood Trauma / Nostalgia",
    "Aging / Mortality",
    "Memory / The Past",
    "Dreams vs. Reality",
    "Suffering / Resilience",
    "Search for Meaning / Purpose",
    "Search for Identity / Self-Discovery",
    "Alienation / Loneliness / Isolation",
    "Conformity vs. Individuality",
    "Belonging / Community / Found Family",
    "Family Dynamics / Legacy",
    "Friendship (Bonds, Betrayal)",
    "Concept of Home",
    "Social Class / Inequality / Poverty",
    "War / Conflict / Violence",
    "Technology / Progress (Benefits and Dangers)",
    "Religion / Faith / Doubt",
    "Environmentalism / Climate Change",
    "Beauty / Art / Aesthetics",
    "Survival",
    "Freedom / Oppression / Control",
    "Culture Clash / Xenophobia",
    "Media / Propaganda",
]
# VII. TONES
TONES = [
    "Formal",
    "Informal",
    "Colloquial",
    "Direct",
    "Objective",
    "Subjective",
    "Serious",
    "Humorous",
    "Lighthearted",
    "Playful",
    "Witty",
    "Sarcastic",
    "Ironic",
    "Satirical",
    "Cynical",
    "Sardonic",
    "Whimsical",
    "Didactic",
    "Grave",
    "Solemn",
    "Optimistic",
    "Hopeful",
    "Cheerful",
    "Joyful",
    "Elated",
    "Loving",
    "Affectionate",
    "Admiring",
    "Reverent",
    "Enthusiastic",
    "Passionate",
    "Lively",
    "Encouraging",
    "Inspirational",
    "Confident",
    "Calm",
    "Serene",
    "Peaceful",
    "Pessimistic",
    "Hopeless",
    "Gloomy",
    "Somber",
    "Melancholic",
    "Mourning",
    "Sad",
    "Regretful",
    "Wistful",
    "Resigned",
    "Bitter",
    "Resentful",
    "Angry",
    "Indignant",
    "Outraged",
    "Hostile",
    "Malicious",
    "Scathing",
    "Caustic",
    "Contemptuous",
    "Scornful",
    "Condescending",
    "Arrogant",
    "Worried",
    "Anxious",
    "Apprehensive",
    "Fearful",
    "Horrified",
    "Panicked",
    "Tense",
    "Nervous",
    "Suspicious",
    "Skeptical",
    "Uncertain",
    "Ambiguous",
    "Confused",
    "Ominous",
    "Foreboding",
    "Sinister",
    "Threatening",
    "Paranoid",
    "Detached",
    "Aloof",
    "Indifferent",
    "Apathetic",
    "Nonchalant",
    "Impassive",
    "Clinical",
    "Matter-of-fact",
    "Restrained",
    "Subdued",
    "Curious",
    "Inquisitive",
    "Pensive",
    "Contemplative",
    "Reflective",
    "Nostalgic",
    "Sentimental",
    "Romantic",
    "Dreamy",
    "Mysterious",
    "Candid",
    "Terse",
    "Urgent",
    "Pleading",
    "Sympathetic",
    "Empathetic",
    "Benevolent",
    "Gentle",
    "Humble",
    "Apologetic",
    "Macabre",
    "Ghoulish",
    "Tragic",
    "Dramatic",
    "Melodramatic",
]
# VIII. PROTAGONISTS (List of [name, description] tuples)
PROTAGONISTS = [
    (
        "Commander Valerius",
        "A classic Hero: leads the fleet against overwhelming odds, defined by courage but must overcome recklessness.",
    ),
    (
        "Anya Sharma",
        "A reluctant Hero: chosen by prophecy, must find the courage to face a dark lord she never believed existed.",
    ),
    (
        "Sam 'Widget' Jones",
        "An Everyman/Orphan: a spaceship mechanic caught in an interstellar war, just trying to survive and find his lost sister.",
    ),
    (
        "Elara Meadowbrook",
        "An Everyman: a village baker who discovers a hidden map and is thrust into an unexpected adventure.",
    ),
    (
        "Pip",
        "An Innocent: a wide-eyed automaton exploring a complex human world, seeking understanding and friendship, vulnerable to manipulation.",
    ),
    (
        "Sister Agnes",
        "An Innocent: a young nun whose unwavering faith is tested when her isolated convent faces a supernatural threat.",
    ),
    (
        "Jax 'Shadow' Kade",
        "A Rebel/Outlaw: fights against a corrupt corporate government from the city's underbelly, values freedom above all.",
    ),
    (
        "Lyra 'Spark' Volkov",
        "A Revolutionary: leads an uprising against a technologically superior empire, challenging the status quo.",
    ),
    (
        "Lysander Thorne",
        "A Lover: a poet willing to defy feuding families and cross dangerous lands for the person they love.",
    ),
    (
        "Seraphina Dubois",
        "A Lover: motivated by deep connection, must choose between personal happiness and a duty that demands sacrifice.",
    ),
    (
        "Dr. Aris Thorne",
        "An Explorer: a driven archaeologist seeking a lost city, pushing into dangerous territory despite warnings.",
    ),
    (
        "Captain Eva Rostova",
        "An Explorer: charts unknown star systems, driven by curiosity and the thrill of discovery.",
    ),
    (
        "Master Elias Thorne",
        "A Creator: a brilliant artificer whose inventions could save the world or destroy it, wrestling with the consequences.",
    ),
    (
        "Silas Vance",
        "A Creator: a bio-engineer who crafts new life forms, facing ethical dilemmas about playing god.",
    ),
    (
        "Brother Thomas",
        "A Caregiver: a selfless healer tending to the afflicted in a plague-ridden city, offering comfort amidst despair.",
    ),
    (
        "Mara Stone",
        "A Caregiver: protects a group of orphans in a post-apocalyptic wasteland, driven by compassion.",
    ),
    (
        "Zane 'Razor' Riley",
        "An Anti-Hero: a cynical mercenary haunted by his past, takes morally grey jobs for survival but shows glimmers of conscience.",
    ),
    (
        "Silas Blackwood",
        "An Anti-Hero: a disgraced sorcerer seeking power for selfish reasons, often resorting to dark methods, but occasionally aiding others.",
    ),
]
# IX. PACING
PACING = [
    "Slow Pacing (for depth, atmosphere, suspense)",
    "Fast Pacing (for excitement, urgency, action)",
    "Varied Pacing (shifting tempo for dynamics)",
    "Breakneck Pacing (relentless action, high energy)",
    "Measured / Deliberate Pacing (steady, controlled unfolding)",
]
# X. COMPLICATIONS
COMPLICATIONS = [
    "An unexpected character arrival changes everything",
    "New information invalidates the current goal or reveals a deeper conspiracy",
    "A major external event (disaster, war) drastically alters the context",
    "A vital object or resource is lost or destroyed",
    "A sudden deadline or rule change makes the plan impossible",
    "The antagonist reveals an unexpected strength or strategy",
    "A supposedly safe location turns out to be a trap",
    "A necessary tool/method has dangerous, unforeseen side effects",
    "The protagonist makes a significant error in judgment with dire consequences",
    "A character's flaw (addiction, fear, pride) sabotages efforts",
    "A moral dilemma forces a choice with only bad options",
    "An intrusive thought or doubt causes critical hesitation/error",
    "Past mistakes or secrets resurface and cause problems",
    "The protagonist develops conflicting goals or desires",
    "A character undergoes a belief change altering their motivation",
    "A trusted friend, ally, or mentor betrays the protagonist",
    "A romantic entanglement creates conflicts of interest or forces hard choices",
    "An authority figure gives contradictory or impossible orders",
    "A misunderstanding between allies leads to disaster",
    "A rival actively works to sabotage the protagonist's goal",
    "Deepening relationships raise the personal stakes of failure",
    "Social pressure or public opinion turns against the protagonist",
]
# Helper list for random actions/situations
ACTIONS_SITUATIONS = [
    "investigating a mysterious event",
    "escaping a dangerous situation",
    "negotiating with a rival",
    "discovering a hidden truth",
    "facing a moral quandary",
    "protecting an ally",
    "planning a risky maneuver",
    "reflecting on a past failure",
    "using their unique skills",
    "navigating a complex social interaction",
    "confronting an antagonist",
]

DEFAULT_NOVEL_DATA_PATH = "novel_data.json"


def load_novel_data(file_path=DEFAULT_NOVEL_DATA_PATH) -> Dict[str, Any]:
    """Loads novel generation parameters from a JSON file, with fallbacks to hardcoded defaults."""
    data = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Novel data loaded successfully from {file_path}")
    except FileNotFoundError:
        logger.warning(
            f"{file_path} not found. Using hardcoded default parameter lists."
        )
    except json.JSONDecodeError:
        logger.error(
            f"Error decoding JSON from {file_path}. Using hardcoded default parameter lists."
        )
    except Exception as e:
        logger.error(
            f"Error loading {file_path}: {e}. Using hardcoded default parameter lists."
        )

    # Fallback to global constants if not found in file or file load failed
    return {
        "genres": data.get("genres", GENRES),
        "settings": data.get("settings", SETTINGS),
        "target_audiences": data.get("target_audiences", TARGET_AUDIENCES),
        "concepts": data.get("concepts", CONCEPTS),
        "world_elements": data.get("world_elements", WORLD_ELEMENTS),
        "themes": data.get("themes", THEMES),
        "tones": data.get("tones", TONES),
        "protagonists": data.get("protagonists", PROTAGONISTS),
        "pacing": data.get("pacing", PACING),
        "complications": data.get("complications", COMPLICATIONS),
        "actions_situations": data.get("actions_situations", ACTIONS_SITUATIONS),
    }


# --- LLM Invocation Helper with Retries ---
def invoke_llm_with_retry(llm, prompt_template, input_data, parser=None, max_retries=1):
    """Invokes an LLM with a prompt and input, handling retries and optional parsing."""
    chain_no_parser = prompt_template | llm | StrOutputParser()
    last_exception = None
    raw_response = ""

    for attempt in range(max_retries + 1):
        try:
            raw_response = chain_no_parser.invoke(input_data)
            if not raw_response or not raw_response.strip():
                raise ValueError("LLM returned an empty or whitespace-only response.")

            if parser:
                result = parser.invoke(raw_response)
                if (
                    result is None
                ):  # Some parsers might return None for empty valid structures
                    # Check if the type of result is what we'd expect from a "truly empty" parse
                    # For Pydantic, an empty object might be valid. For Str, an empty string is not caught by "not result"
                    # The earlier check for raw_response handles empty strings.
                    # This is more about parser returning None when it shouldn't based on raw_response.
                    pass  # Allow None if parser legitimately produces it (e.g. optional field)
                    # Re-evaluating: if raw_response was non-empty, parser returning None might be an issue.
                    if (
                        not isinstance(result, (str, list, dict))
                        and result is None
                        and raw_response.strip()
                    ):
                        # If raw response had content, but parser gave None (and it's not a string itself)
                        raise ValueError(
                            f"Parser returned None from non-empty raw response: '{raw_response[:100]}...'"
                        )

            else:
                result = raw_response
            return result

        except (ValidationError, ValueError) as ve:
            logger.warning(
                f"LLM invocation/parsing failed (Attempt {attempt + 1}/{max_retries + 1}): {ve}"
            )
            logger.warning(f"Raw LLM Response on failure: '{raw_response[:500]}...'")
            last_exception = ve
            if attempt == max_retries:
                logger.error("Max retries reached for LLM invocation/parsing.")
                raise last_exception
            time.sleep(1.5**attempt)

        except Exception as e:
            logger.warning(
                f"LLM invocation failed unexpectedly (Attempt {attempt + 1}/{max_retries + 1}): {e}"
            )
            last_exception = e
            if attempt == max_retries:
                logger.error("Max retries reached for LLM invocation.")
                raise last_exception
            time.sleep(1.5**attempt)
    # Should not be reached if max_retries >= 0
    raise RuntimeError(
        f"LLM invocation failed after retries. Last exception: {last_exception}"
    )


# --- Character Generation Enhancement ---
class EnhancedCharacter(BaseModel):
    name: str = Field(description="Character's full name")
    motivation_arc: str = Field(
        description="Core motivation and potential character arc, possibly incorporating role, relationship, and flaw for a richer description."
    )
    potential_role: str = Field(
        description="Potential narrative role (e.g., Antagonist, Mentor, Foil, Love Interest, Rival)"
    )
    key_relationship: str = Field(
        description="Brief description of a key relationship with the protagonist or another character, explaining its nature."
    )
    core_flaw: str = Field(
        description="A significant flaw relevant to the story concept/themes that could drive conflict or their arc."
    )


def enhance_characters_llm(
    characters: List[Dict],
    concept: str,
    genre: str,
    llm,
    parser: PydanticOutputParser,
    max_retries: int,
):
    """Uses LLM to flesh out characters, focusing on the second character."""
    logger.info("Attempting LLM call to enhance character details...")
    if not characters or len(characters) < 2:
        logger.warning(
            "Not enough characters provided (need at least 2) for enhancement. Skipping."
        )
        return characters

    # Ensure the base character dictionaries have the keys we might access
    for char_dict in characters:
        char_dict.setdefault("motivation_arc", "N/A")

    char_list_str_for_prompt = "\n".join(
        [f"- {c['name']}: {c['motivation_arc']}" for c in characters]
    )
    # Ensure second_char_name is defined before being used in the f-string template
    second_char_name = characters[1]["name"]

    # This is raw JSON text. It does not contain f-string placeholders itself.
    # The LLM is instructed NOT to include markdown in its output, so the example should be pure JSON.
    json_example_for_llm_output = """{
  "name": "Silas Blackwood",
  "motivation_arc": "A disgraced sorcerer seeking forbidden knowledge for perceived 'greater good', clashing ideologically with the protagonist. Their shared past as colleagues at the Arcane University now fuels their rivalry, with Silas's ruthless pragmatism (believing the ends justify any means) driving him down a dangerous path.",
  "potential_role": "Antagonist/Foil",
  "key_relationship": "Former colleague and intellectual rival of the protagonist from their university days; their paths diverged due to ethical disagreements.",
  "core_flaw": "Ruthless Pragmatism - believes the ends justify any means, often leading to morally compromising decisions."
}"""

    # System prompt where Python f-string substitution happens first.
    # LangChain variables are {{escaped}}.
    system_prompt_text = f"""You are a character development assistant. Your task is to enhance the details for the SECOND character listed ('{second_char_name}'), based on the provided context.

Output ONLY a single, valid JSON object that strictly adheres to the Pydantic format instructions provided below.
Do NOT include any markdown ```json delimiters or any text before or after the JSON object.

Context:
- Core Concept: {{concept}}
- Genre: {{genre}}
- Initial Characters:
{{char_list_str_for_template}}
- Character to Enhance: '{second_char_name}'

Focus on these details for the enhanced character within the JSON:
- `name`: Keep the original name ('{second_char_name}').
- `motivation_arc`: Enhance the description of their motivation and potential character arc. This description should ideally be a compelling summary that could also subtly incorporate their role, key relationship, and flaw for a richer profile.
- `potential_role`: Determine a plausible narrative role (e.g., Antagonist, Mentor, Foil, Love Interest, Rival).
- `key_relationship`: Define a key relationship this character has, preferably with the protagonist or another significant character, explaining its nature briefly.
- `core_flaw`: Identify a relevant character flaw that could drive conflict or their arc.

Example of the EXACT JSON Structure required for your output (do NOT include the ```json markdown in your actual output):
{json_example_for_llm_output}

Pydantic Format Instructions (Strictly follow these for the JSON structure):
{{format_instructions}}
"""

    human_prompt_text = f"Enhance the character '{second_char_name}' and provide the details ONLY as a valid JSON object, adhering strictly to the Pydantic format instructions."

    # Create the ChatPromptTemplate with the processed strings
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt_text), ("human", human_prompt_text)]
    )

    # Variables to be filled by LangChain's .invoke()
    invoke_dict = {
        "concept": concept,
        "genre": genre,
        "char_list_str_for_template": char_list_str_for_prompt,  # Use the renamed variable
        "format_instructions": parser.get_format_instructions(),
    }

    try:
        # Ensure 'parser' is the PydanticOutputParser instance for EnhancedCharacter
        enhanced_data: EnhancedCharacter = invoke_llm_with_retry(
            llm, prompt, invoke_dict, parser=parser, max_retries=max_retries
        )

        if not enhanced_data:  # Should be caught by invoke_llm_with_retry if parser returns None from non-empty
            logger.warning(
                f"LLM for character enhancement returned empty parsable data for {second_char_name}. Skipping update."
            )
            return characters

        logger.info(
            f"LLM generated enhanced details for character: {enhanced_data.name}"
        )

        # Update the second character in the original list
        # It's good practice to ensure the name matches what we asked for.
        # The prompt explicitly tells the LLM to keep the original name.
        if characters[1]["name"] == enhanced_data.name:
            characters[1]["motivation_arc"] = enhanced_data.motivation_arc
            characters[1]["potential_role"] = enhanced_data.potential_role
            characters[1]["key_relationship"] = enhanced_data.key_relationship
            characters[1]["core_flaw"] = enhanced_data.core_flaw
            # Add new keys if they don't exist, or update if they do.
            # This makes the character dict richer.
            logger.info(
                f"Successfully updated '{enhanced_data.name}' with full enhanced details."
            )
        else:
            # This case means the LLM didn't follow instructions to keep the name.
            logger.warning(
                f"Name mismatch during character enhancement: Expected '{characters[1]['name']}', but LLM returned '{enhanced_data.name}'. Applying updates to original character entry '{characters[1]['name']}'."
            )
            # Still apply to the character at index 1, but acknowledge the LLM's name deviation.
            characters[1]["motivation_arc"] = enhanced_data.motivation_arc
            characters[1]["potential_role"] = enhanced_data.potential_role
            characters[1]["key_relationship"] = enhanced_data.key_relationship
            characters[1]["core_flaw"] = enhanced_data.core_flaw
            # Optionally, you could update the name too if you trust the LLM's correction:
            # characters[1]['name'] = enhanced_data.name

    except ValidationError as ve:
        logger.warning(
            f"Pydantic validation error for '{second_char_name}' after LLM call: {ve}. Raw LLM output might have been non-compliant. Using initial characters for this one."
        )
        # The invoke_llm_with_retry should log raw output on parsing failure.
    except KeyError as ke:
        # This specific KeyError was the original problem.
        logger.error(
            f"KeyError during character enhancement for '{second_char_name}': {ke}. This indicates a prompt template variable issue.",
            exc_info=True,
        )
        # This error should ideally be caught by invoke_llm_with_retry if it's a prompt input issue,
        # but catching it here provides specific context.
    except Exception as e:
        logger.error(
            f"Could not enhance character '{second_char_name}' using LLM due to an unexpected error. Using initial characters. Error: {type(e).__name__} - {e}",
            exc_info=True,
        )

    return characters


def generate_novel_parameters(
    config: Dict,
    args: argparse.Namespace,
    char_gen_llm=None,
    param_lists: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates parameters for the novel, allowing overrides and optional LLM enhancement."""
    logger.info("Generating novel parameters...")
    params = {}
    overrides = config.get("novel_override_parameters", {})

    # Use provided lists from param_lists, or fallback to global constants if param_lists is None or key missing
    _genres = param_lists.get("genres", GENRES) if param_lists else GENRES
    _settings = param_lists.get("settings", SETTINGS) if param_lists else SETTINGS
    _target_audiences = (
        param_lists.get("target_audiences", TARGET_AUDIENCES)
        if param_lists
        else TARGET_AUDIENCES
    )
    _concepts = param_lists.get("concepts", CONCEPTS) if param_lists else CONCEPTS
    _world_elements_map = (
        param_lists.get("world_elements", WORLD_ELEMENTS)
        if param_lists
        else WORLD_ELEMENTS
    )
    _themes = param_lists.get("themes", THEMES) if param_lists else THEMES
    _tones = param_lists.get("tones", TONES) if param_lists else TONES
    _protagonists_list = (
        param_lists.get("protagonists", PROTAGONISTS) if param_lists else PROTAGONISTS
    )
    _pacing_list = param_lists.get("pacing", PACING) if param_lists else PACING
    _complications_list = (
        param_lists.get("complications", COMPLICATIONS)
        if param_lists
        else COMPLICATIONS
    )
    _actions_situations_list = (
        param_lists.get("actions_situations", ACTIONS_SITUATIONS)
        if param_lists
        else ACTIONS_SITUATIONS
    )

    params["genre"] = args.genre or overrides.get("genre") or random.choice(_genres)
    params["setting"] = (
        args.setting or overrides.get("setting") or random.choice(_settings)
    )
    params["target_audience"] = (
        args.target_audience
        or overrides.get("target_audience")
        or random.choice(_target_audiences)
    )
    params["core_concept"] = (
        args.concept or overrides.get("core_concept") or random.choice(_concepts)
    )

    if "world_element_key" in overrides:
        params["world_element_key"] = overrides["world_element_key"]
        params["world_element_desc"] = _world_elements_map.get(
            params["world_element_key"], "Description missing in WORLD_ELEMENTS list."
        )
    else:
        params["world_element_key"] = random.choice(list(_world_elements_map.keys()))
        params["world_element_desc"] = _world_elements_map[params["world_element_key"]]

    params["themes"] = overrides.get("themes") or random.sample(
        _themes, k=random.randint(1, min(3, len(_themes)))
    )
    params["tone"] = args.tone or overrides.get("tone") or random.choice(_tones)

    initial_characters = []
    if (
        "characters" in overrides
        and isinstance(overrides["characters"], list)
        and len(overrides["characters"]) > 0
    ):
        initial_characters = overrides["characters"]
        logger.info("Using overridden characters.")
    else:
        logger.info("Generating random characters.")
        if not _protagonists_list:
            logger.warning(
                "Protagonists list is empty. Cannot generate random characters."
            )
            initial_characters = [
                {"name": "Default Protagonist", "motivation_arc": "Needs a story."}
            ]
        else:
            selected_protagonist_tuple = random.choice(_protagonists_list)
            possible_other_chars = [
                p for p in _protagonists_list if p[0] != selected_protagonist_tuple[0]
            ]
            selected_other_char_tuple = (
                random.choice(possible_other_chars)
                if possible_other_chars
                else (
                    _protagonists_list[0]
                    if len(_protagonists_list) == 1
                    and _protagonists_list[0][0] != selected_protagonist_tuple[0]
                    else ("Sidekick Zero", "Needs motivation/archetype")
                )
            )

            initial_characters = [
                {
                    "name": selected_protagonist_tuple[0],
                    "motivation_arc": selected_protagonist_tuple[1],
                },
            ]
            if (
                selected_other_char_tuple[0] != "Sidekick Zero"
                or len(initial_characters) < 2
            ):  # ensure second char if possible
                initial_characters.append(
                    {
                        "name": selected_other_char_tuple[0],
                        "motivation_arc": selected_other_char_tuple[1],
                    }
                )

    # In generate_novel_parameters, before calling enhance_characters_llm:
    if char_gen_llm and len(initial_characters) > 1:
        # Setup parser for enhanced character data
        char_parser = PydanticOutputParser(
            pydantic_object=EnhancedCharacter
        )  # Ensure this uses the class
        params["characters"] = enhance_characters_llm(
            initial_characters,
            params["core_concept"],
            params["genre"],
            char_gen_llm,
            char_parser,  # Pass the parser instance
            config["workflow_settings"]["max_llm_retries"],
        )
    else:
        params["characters"] = initial_characters
    params["novel_title"] = (
        overrides.get("novel_title")
        or f"{random.choice(['Shadows of', 'Echoes of', 'The Last', 'Chronicles of', 'Whispers in', 'Steel and'])} {random.choice(['Aethelgard', 'Nebula', 'Kyoto', 'Arcadia', 'Oblivion', 'the Void', 'Ironhelm', 'Dreamfall'])}"
    )

    char_details = "N/A"
    main_char_name = "Protagonist"  # Fallback
    if params["characters"]:
        main_char_name = params["characters"][0]["name"]
        main_char_motiv = params["characters"][0]["motivation_arc"]
        other_char_details_list = []
        if len(params["characters"]) > 1:
            for char_idx in range(1, len(params["characters"])):
                other_char_details_list.append(
                    f'{params["characters"][char_idx]["name"]}: "{params["characters"][char_idx]["motivation_arc"]}"'
                )
        other_char_details = "; ".join(other_char_details_list)
        char_details = f'{main_char_name}: "{main_char_motiv}"'
        if other_char_details:
            char_details += f"; {other_char_details}"

    guidelines = f"""
    **Core Writing Instructions:**
    - Genre: {params["genre"]}
    - Setting: {params["setting"]}
    - Target Audience: {params["target_audience"]}
    - Concept: {params["core_concept"]}
    - Main Characters: {char_details}
    - Key World Element: {params["world_element_key"]} ({params["world_element_desc"]})
    - Themes: {", ".join(params["themes"])}
    - Tone: {params["tone"]}

    **Chapter Writing Requirements (Apply Continuously):**
    - **Show, Don't Tell:** Demonstrate emotions/states via actions, dialogue, internal thoughts, sensory details. Avoid stating feelings directly.
    - **Character Voice:** Reflect established voices/perspectives consistently. Refer to core traits/motivations/flaws. Dialogue should sound distinct for each character.
    - **Sensory Details:** Ground scenes vividly in the setting ({params["setting"]}) using specific sight, sound, smell, touch, taste details. Make them purposeful, contributing to mood or plot.
    - **Pacing:** Adjust sentence/paragraph length and detail level to control scene pace (e.g., {random.choice(_pacing_list if _pacing_list else ["Varied Pacing"])}). Vary pace strategically between action and reflection. Avoid long stretches of uniform pace.
    - **Consistency:** Maintain strict continuity with characters (actions, knowledge, abilities), world rules ({params["world_element_key"]}), established plot events, setting details, and overall tone ({params["tone"]}). Refer back to synopsis and notes.
    - **Theme Resonance:** Subtly weave in themes like '{random.choice(params["themes"] if params["themes"] else ["a central theme"])}' through character choices, conflicts, dialogue nuances, or symbolic imagery. Avoid explicit moralizing.
    - **Avoid Info-Dumps:** Integrate necessary exposition organically through dialogue arising from character need, actions revealing world aspects, or discoveries within the scene. Do not pause the narrative for lengthy explanations.
    - **Avoid Deus Ex Machina:** Solutions to problems must feel earned and plausible within the established world rules and character capabilities. Use foreshadowing effectively. Foreshadow {random.choice(["a hidden character motivation", "a consequence of the world element", "a societal conflict", "a personal weakness", "an environmental hazard", "a specific object's later use"])}. Ensure character actions have consequences.

    **Specific Scene Requirements:**
    - Scene Focus: Each scene should have a clear purpose (advance plot, reveal character, introduce conflict). Often include protagonist ({main_char_name}) in action/situation like {random.choice(_actions_situations_list if _actions_situations_list else ["a challenging situation"])}. Ensure scenes start and end effectively (e.g., hook, transition, mini-cliffhanger).
    - Chapter Endings: Aim for strong hooks, turning points, cliffhangers, significant revelations, or poignant emotional moments relevant to the chapter goal and overall arc. Avoid flat or insignificant endings.
    - Climax (Overall): Must feature the highest stakes, direct confrontation of the core conflict, peak emotional intensity, and a resolution (even if costly) driven by protagonist action/choice. Should provide payoff for major plot threads.
    - Post-Climax (Falling Action): Show the immediate consequences of the climax, emotional/physical toll on characters, reflection on events, and the initial steps towards establishing the 'new normal' or resolution.
    - Ending (Resolution): Provide clear payoff for main character arcs (transformation, acceptance, etc.). Resolve the central conflict. Reflect conclusively on the primary themes. Show the final state of the world/relationships affected by the story. Lingering questions or sequel hooks are acceptable *if* appropriate for the genre and intended effect, but the core story should feel complete.

    **Refinement Checks (Editor/Writer/Consistency Focus):**
    - Pacing: Is it varied appropriately? If rushed, add internal thought/sensory detail/beats. If slow, condense description/exposition, increase stakes, or inject action/dialogue.
    - Dialogue: Does it sound authentic to each character's voice/background/mood? Does it reveal character/advance plot/contain subtext? Eliminate generic or purely expositional dialogue.
    - Description: Is it specific, concrete, and sensory? Does it contribute to tone/atmosphere/worldbuilding? Replace vague adjectives/adverbs with stronger verbs/nouns and specific details relevant to {params["setting"]}.
    - Plot Points: Are motivations clear? Are consequences shown? Are stakes high enough? If underdeveloped, expand on character reactions, decision-making processes, and the ripple effects of events.
    - Conflict/Resolution: Is conflict compelling and plausible? Is resolution earned, avoiding easy fixes or deus ex machina? Introduce meaningful complications (e.g., {random.choice(_complications_list if _complications_list else ["an unexpected obstacle"])}) if conflict feels weak. Ensure obstacle resolution aligns with character skills/world rules.
    - Exposition: Is background info integrated smoothly or dumped? Rewrite info-dumps to be revealed through character experience, dialogue driven by need-to-know, or environmental storytelling.
    - Tropes: Are tropes used consciously? Rework clichés into unique versions fitting the specific world/characters, or subvert them intentionally.
    - Consistency Errors: Check names, timeline, established facts, character knowledge/abilities against previous chapters/notes/synopsis. Flag any contradictions.
    """
    params["writing_guidelines"] = guidelines
    logger.info("Novel parameters generated.")
    return params


# --- State Definition ---
class NovelWritingState(TypedDict, total=False):
    novel_parameters: Dict[str, Any]
    writing_guidelines: str
    novel_synopsis: Optional[str]
    character_notes: Optional[str]
    chapter_goal_list: Optional[List[str]]
    current_chapter_index: int
    current_chapter_goal: Optional[str]
    previous_chapter_summary: Optional[str]
    chapter_beat_sheet: Optional[str]
    draft_chapter: Optional[str]
    editor_feedback: Optional[str]
    refined_chapter: Optional[str]
    consistency_report: Optional[str]
    author_feedback: Optional[str]
    author_decision: Optional[str]
    revision_cycles: int
    max_revisions: int
    completed_chapters: List[str]
    last_error: Optional[str]


# --- Pydantic Models for Structured Output ---
class NovelPlan(BaseModel):
    synopsis: str = Field(
        description="Concise high-level plot outline (inciting incident, key rising action points/turning points, climax, falling action, resolution). Aim for 5-8 key plot points, combined into a single string with newlines if needed."
    )
    character_notes: str = Field(
        description="Notes expanding on core traits, goals, flaws, key relationships, and planned arcs for the main characters, combined into a single string with newlines if needed."
    )
    chapter_goals: List[str] = Field(
        description="Numbered list of primary goals for each chapter (e.g., '1. Protagonist receives the call to action.'), logically derived from the synopsis. Should cover the entire story arc."
    )


# --- Vector Store Setup ---
vector_store = None
embedding_function = None
text_splitter = None


def initialize_vector_store(config: Dict):
    global vector_store, embedding_function, text_splitter
    vs_config = config.get("vector_store_settings", {})
    use_vs = vs_config.get("use_vector_store", False)

    if not VECTOR_STORE_ENABLED or not use_vs:
        logger.warning("Vector store usage is disabled by config or missing libraries.")
        vector_store = None  # Ensure it's None
        return

    try:
        logger.info("Initializing vector store...")
        emb_model_name = vs_config.get("embedding_model", "all-MiniLM-L6-v2")
        collection_name = vs_config.get("collection_name", "novel_chapters")
        output_dir = config["workflow_settings"]["output_directory"]
        persist_dir = os.path.join(output_dir, "vector_store")

        logger.info(f"Using embedding model: {emb_model_name}")
        embedding_function = SentenceTransformerEmbeddings(model_name=emb_model_name)
        os.makedirs(persist_dir, exist_ok=True)

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_dir,
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        logger.info(
            f"Vector store initialized (Collection: {collection_name}, Persist Dir: {persist_dir})"
        )
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}", exc_info=True)
        vector_store = None


def add_chapter_to_vector_store(chapter_index: int, chapter_text: str, summary: str):
    if not vector_store or not text_splitter:
        return
    try:
        logger.info(f"Adding Chapter {chapter_index} to vector store...")
        chapter_docs = text_splitter.create_documents(
            [chapter_text],
            metadatas=[{"type": "chapter_content", "chapter": chapter_index}],
        )
        summary_doc = text_splitter.create_documents(
            [summary], metadatas=[{"type": "chapter_summary", "chapter": chapter_index}]
        )
        if chapter_docs:
            vector_store.add_documents(chapter_docs)
        if summary_doc:
            vector_store.add_documents(summary_doc)
        # vector_store.persist() # Chroma client usually persists on add/update for file-based.
        logger.info(
            f"Successfully added Chapter {chapter_index} content and summary to vector store."
        )
    except Exception as e:
        logger.error(
            f"Failed to add Chapter {chapter_index} to vector store: {e}", exc_info=True
        )


def get_relevant_context_from_vector_store(
    query: str, k: int = 3, max_chars: int = 1500
) -> str:
    if not vector_store:
        return "No relevant context available from vector store."
    try:
        logger.info(f"Querying vector store for context related to: '{query[:50]}...'")
        retrieved_docs = vector_store.similarity_search(query, k=k)
        context = ""
        current_chars = 0
        for doc in retrieved_docs:
            doc_text = f"[Context from Chapter {doc.metadata.get('chapter', 'N/A')} ({doc.metadata.get('type', 'content')})]:\n{doc.page_content}\n...\n"
            if current_chars + len(doc_text) <= max_chars:
                context += doc_text
                current_chars += len(doc_text)
            else:
                remaining_chars = max_chars - current_chars
                if remaining_chars > 100:
                    context += doc_text[:remaining_chars] + "...\n"
                break
        logger.info(
            f"Retrieved {len(retrieved_docs)} docs, using {current_chars} chars of context."
        )
        return context if context else "No relevant context found in vector store."
    except Exception as e:
        logger.error(
            f"Failed to retrieve context from vector store: {e}", exc_info=True
        )
        return "Error retrieving context from vector store."


# --- Agent Node Functions ---
def signal_error(
    state: NovelWritingState,
    node_name: str,
    message: str,
    e: Optional[Exception] = None,
) -> Dict:
    full_message = f"Error in {node_name}: {message}"
    if e:
        full_message += f" | Exception: {type(e).__name__} - {e}"
        logger.error(full_message, exc_info=True)
    else:
        logger.error(full_message)
    return {"author_decision": "Error", "last_error": full_message}


# --- Author Agent (Planning & Review) ---
def author_agent(state: NovelWritingState, max_retries: int, llm_map: Dict):
    node_name = "Author Agent"
    logger.info(f"--- Entering {node_name} ---")
    params = state["novel_parameters"]
    guidelines = state["writing_guidelines"]
    author_llm = llm_map["author"]
    planning_llm = llm_map["planning"]

    if state.get("novel_synopsis") is None:  # Initial Planning
        logger.info(
            f"Performing initial planning for: {params.get('novel_title', 'Untitled Novel')}"
        )
        plan_parser = PydanticOutputParser(pydantic_object=NovelPlan)

        char_list = params.get("characters", [])
        char_list_str = "\n".join(
            [f"- {c['name']}: {c['motivation_arc']}" for c in char_list]
        )
        themes_str = ", ".join(params.get("themes", []))

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are a master novelist and planner. Based on the provided novel parameters, generate a coherent plan.

            **Your Task:** Output ONLY a single, valid JSON object containing the following keys:
            1.  `synopsis`: **(string)** Combine the 5-8 key plot points (inciting incident, rising action, climax, falling action, resolution) into a **single paragraph or string**, using newlines (`\\n`) if needed for readability. **Do NOT output a list for the synopsis.**
            2.  `character_notes`: **(string)** Combine the expanded notes for all main characters (core traits, goals, flaws, relationships, arcs) into a **single string**, using newlines (`\\n`) between character notes. **Do NOT output a list for character_notes.**
            3.  `chapter_goals`: **(list of strings)** A list where each string is a primary goal for a chapter (e.g., "1. Protagonist receives the call to action."). Create 15-35 sequential goals derived logically from the synopsis.

            **Think step-by-step:**
            1. Analyze the parameters.
            2. Develop the synopsis plot points. **Combine them into a single string for the 'synopsis' value.**
            3. Develop the character notes. **Combine them into a single string for the 'character_notes' value.**
            4. Break the synopsis into sequential chapter goals (as a list of strings for the 'chapter_goals' value).
            5. Construct the final JSON object with ONLY the keys `synopsis`, `character_notes`, and `chapter_goals` containing the generated content in the correct format (string, string, list of strings). Do NOT include ```json markdown delimiters or any other text.

            **Novel Parameters:**
            Title: {params.get("novel_title", "N/A")}
            Genre: {params.get("genre", "N/A")}
            Setting: {params.get("setting", "N/A")}
            Target Audience: {params.get("target_audience", "N/A")}
            Core Concept: {params.get("core_concept", "N/A")}
            Main Characters:
{char_list_str}
            Key World Element: {params.get("world_element_key", "N/A")} - {params.get("world_element_desc", "N/A")}
            Primary Themes: {themes_str}
            Overall Tone: {params.get("tone", "N/A")}

            Format Instructions: {{format_instructions}}
            """,
                ),
                (
                    "human",
                    "Generate the novel plan strictly as a JSON object with 'synopsis' (string), 'character_notes' (string), and 'chapter_goals' (list of strings) keys, adhering to the format instructions.",
                ),
            ]
        )

        invoke_dict = {
            "format_instructions": plan_parser.get_format_instructions()
            # Other parameters are now part of the f-string prompt itself
        }

        try:
            parsed_plan: NovelPlan = invoke_llm_with_retry(
                planning_llm,
                prompt,
                invoke_dict,
                parser=plan_parser,
                max_retries=max_retries,
            )
            if (
                not parsed_plan.synopsis
                or not parsed_plan.character_notes
                or not parsed_plan.chapter_goals
            ):
                return signal_error(
                    state,
                    node_name,
                    "LLM generated plan with missing critical elements (synopsis, notes, or goals).",
                )
            if len(parsed_plan.chapter_goals) < 5:
                logger.warning(
                    f"Author generated very few chapter goals ({len(parsed_plan.chapter_goals)}). Plan might be short/incomplete."
                )

            logger.info(
                f"Successfully generated novel plan. Synopsis: {parsed_plan.synopsis[:100]}... Goals#: {len(parsed_plan.chapter_goals)}"
            )
            return {
                "novel_synopsis": parsed_plan.synopsis,
                "character_notes": parsed_plan.character_notes,
                "chapter_goal_list": parsed_plan.chapter_goals,
                "current_chapter_index": 0,
                "current_chapter_goal": parsed_plan.chapter_goals[0],
                "revision_cycles": 0,
                "author_decision": None,  # Reset decision after planning
                "author_feedback": None,
                "completed_chapters": [],
                "previous_chapter_summary": None,
                "max_revisions": state.get(
                    "max_revisions"
                ),  # Pass through from initial state
            }
        except Exception as e:
            logger.error(
                f"Detailed error during initial planning: {traceback.format_exc()}"
            )
            return signal_error(
                state,
                node_name,
                "Failed during initial planning LLM call or parsing.",
                e,
            )

    else:  # Reviewing a refined chapter
        revision_cycles = state.get("revision_cycles", 0)
        max_revisions = state.get("max_revisions", 3)
        current_chapter_idx = state["current_chapter_index"]
        chapter_goal_list = state.get("chapter_goal_list", [])
        refined_chapter = state.get("refined_chapter")
        consistency_report = state.get(
            "consistency_report", "No consistency check performed."
        )

        if not chapter_goal_list or current_chapter_idx >= len(chapter_goal_list):
            return signal_error(
                state,
                node_name,
                f"Chapter goal list missing or index {current_chapter_idx} out of bounds.",
            )
        if not refined_chapter:
            return signal_error(
                state,
                node_name,
                f"No refined chapter found for review (Chapter {current_chapter_idx + 1}).",
            )

        logger.info(
            f"Reviewing refined Chapter {current_chapter_idx + 1}/{len(chapter_goal_list)}. Cycle: {revision_cycles + 1}/{max_revisions}"
        )

        review_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are the visionary author for '{params.get("novel_title", "Untitled Novel")}'. Review the refined chapter draft based *strictly* on the Chapter Goal, Overall Writing Guidelines, and the Consistency Report.

            **Assessment Criteria:**
            1.  **Goal Achievement:** Does the chapter fully accomplish the specific Chapter Goal?
            2.  **Guideline Adherence:** Does the chapter strongly follow the Overall Writing Guidelines (Show/Tell, Voice, Sensory Detail, Pacing, Consistency, Theme, No Info-Dumps/Deus Ex Machina)?
            3.  **Consistency:** Does the chapter contradict any points raised in the Consistency Report? (Address issues from the report if they are significant).
            4.  **Quality:** Is the prose engaging, clear, and well-written?

            **Your Task:**
            - If the chapter meets all criteria AND the Consistency Report shows no major issues (or issues are acceptable/minor), respond with **ONLY** the word "Approved".
            - If the chapter needs revisions OR if the Consistency Report highlights significant issues requiring changes, provide **concise, actionable feedback** starting with "Feedback:". Focus *only* on deviations from the goal/guidelines or necessary consistency fixes. Be specific. Do NOT approve if feedback is given.

            **Overall Writing Guidelines:**
            {guidelines}

            **Current Chapter Goal (Chapter {current_chapter_idx + 1}):**
            {state["current_chapter_goal"]}

            **Consistency Check Report:**
            {consistency_report}
            """,
                ),
                (
                    "human",
                    "Refined Chapter Draft to Review:\n```\n{refined_chapter}\n```\n\nReview the draft. Respond with 'Approved' OR 'Feedback: [Your specific feedback points]'",
                ),
            ]
        )

        try:
            review = invoke_llm_with_retry(
                author_llm,
                review_prompt_template,
                {"refined_chapter": refined_chapter},
                max_retries=max_retries,
            )
            logger.info(
                f"Author review received for Chapter {current_chapter_idx + 1}: {review[:150]}..."
            )

            decision = "Rewrite"
            feedback = (
                f"Chapter {current_chapter_idx + 1} requires revision based on review."
            )
            review_clean = review.strip().lower()

            if review_clean == "approved":
                decision = "Approved"
                feedback = f"Chapter {current_chapter_idx + 1} Approved."
            elif review_clean.startswith("feedback:"):
                decision = "Rewrite"
                feedback = review.strip()
            else:  # Ambiguous response
                decision = "Rewrite"
                feedback = f"Feedback: Author response unclear ('{review.strip()}'), requesting rewrite. Focus on goal, guidelines, and consistency report."
                logger.warning(
                    f"Author response for Ch {current_chapter_idx + 1} was ambiguous, forcing rewrite."
                )

            next_chapter_index = current_chapter_idx
            next_chapter_goal = state["current_chapter_goal"]
            new_completed_chapters = list(
                state.get("completed_chapters", [])
            )  # Ensure it's a mutable list
            next_revision_cycle = revision_cycles

            if decision == "Approved":
                logger.info(f"Author approved Chapter {current_chapter_idx + 1}.")
                new_completed_chapters.append(refined_chapter)
                if vector_store:
                    add_chapter_to_vector_store(
                        current_chapter_idx + 1,
                        refined_chapter,
                        state.get(
                            "previous_chapter_summary",
                            f"Summary for chapter preceding {current_chapter_idx + 1} N/A",
                        ),
                    )
                next_chapter_index = current_chapter_idx + 1
                if next_chapter_index < len(chapter_goal_list):
                    next_chapter_goal = chapter_goal_list[next_chapter_index]
                    logger.info(f"Preparing for Chapter {next_chapter_index + 1}.")
                else:
                    next_chapter_goal = None  # Signal end
                    logger.info("Final chapter approved. Novel completed!")
                next_revision_cycle = 0  # Reset for next chapter
                return {
                    "author_decision": decision,
                    "completed_chapters": new_completed_chapters,
                    "current_chapter_index": next_chapter_index,
                    "current_chapter_goal": next_chapter_goal,
                    "revision_cycles": next_revision_cycle,
                    "draft_chapter": None,
                    "editor_feedback": None,
                    "refined_chapter": None,
                    "author_feedback": None,
                    "consistency_report": None,
                    "chapter_beat_sheet": None,
                }
            elif decision == "Rewrite":
                next_revision_cycle = revision_cycles + 1
                logger.info(
                    f"Author requested rewrite for Chapter {current_chapter_idx + 1}. Cycle {next_revision_cycle}/{max_revisions}"
                )
                if next_revision_cycle >= max_revisions:
                    logger.warning(
                        f"Max revisions ({max_revisions}) reached for Ch {current_chapter_idx + 1}. Forcing approval."
                    )
                    decision = "Approved"  # Override
                    feedback += (
                        f"\n(Max revisions ({max_revisions}) reached, forced approval)"
                    )
                    new_completed_chapters.append(refined_chapter)
                    if vector_store:
                        add_chapter_to_vector_store(
                            current_chapter_idx + 1,
                            refined_chapter,
                            state.get(
                                "previous_chapter_summary",
                                f"Summary for chapter preceding {current_chapter_idx + 1} N/A",
                            ),
                        )
                    next_chapter_index = current_chapter_idx + 1
                    if next_chapter_index < len(chapter_goal_list):
                        next_chapter_goal = chapter_goal_list[next_chapter_index]
                    else:
                        next_chapter_goal = None
                    next_revision_cycle = 0  # Reset
                    return {
                        "author_decision": decision,
                        "author_feedback": feedback,  # Pass forced approval note
                        "completed_chapters": new_completed_chapters,
                        "current_chapter_index": next_chapter_index,
                        "current_chapter_goal": next_chapter_goal,
                        "revision_cycles": next_revision_cycle,
                        "draft_chapter": None,
                        "editor_feedback": None,
                        "refined_chapter": None,
                        "consistency_report": None,
                        "chapter_beat_sheet": None,
                    }
                else:  # Normal rewrite
                    return {
                        "author_decision": decision,
                        "author_feedback": feedback,
                        "revision_cycles": next_revision_cycle,
                        "draft_chapter": None,
                        "editor_feedback": None,
                        "refined_chapter": None,
                        "consistency_report": None,
                        "chapter_beat_sheet": None,
                    }
        except Exception as e:
            return signal_error(
                state,
                node_name,
                f"Error during Author's review Chapter {current_chapter_idx + 1}.",
                e,
            )


# --- Summarizer Agent ---
def summarizer_agent(state: NovelWritingState, max_retries: int, llm_map: Dict):
    node_name = "Summarizer Agent"
    logger.info(f"--- Entering {node_name} ---")
    summarizer_llm = llm_map["summarizer"]
    current_chapter_idx = state[
        "current_chapter_index"
    ]  # This is the index of the chapter ABOUT to be worked on

    if current_chapter_idx == 0:
        logger.info("No previous chapters to summarize (Starting Chapter 1).")
        return {"previous_chapter_summary": "This is the first chapter."}

    completed_chapters = state.get("completed_chapters", [])
    # The chapter just completed is at index current_chapter_idx - 1 in completed_chapters list
    prev_chapter_list_idx = current_chapter_idx - 1
    prev_chapter_human_num = (
        current_chapter_idx  # Human-readable number of chapter just finished
    )

    if (
        not completed_chapters
        or prev_chapter_list_idx < 0
        or prev_chapter_list_idx >= len(completed_chapters)
    ):
        err_msg = f"Cannot summarize chapter {prev_chapter_human_num}, completed list access error (idx {prev_chapter_list_idx} vs len {len(completed_chapters)})."
        logger.error(err_msg)
        # Don't signal fatal error from summarizer, provide placeholder
        return {
            "previous_chapter_summary": (
                f"Error: Could not retrieve Chapter {prev_chapter_human_num} text for summary."
            )
        }

    last_chapter_text = completed_chapters[prev_chapter_list_idx]
    logger.info(f"Summarizing Chapter {prev_chapter_human_num}")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert summarizer. Read the novel chapter text and provide a concise summary (2-4 sentences) focusing ONLY on information crucial for starting the NEXT chapter:
        - Key events occurred.
        - Significant changes in character state (emotional, physical, relational).
        - Major plot points resolved or introduced.
        - Unresolved questions or cliffhangers.
        Keep it brief and strictly functional for context continuity.""",
            ),
            (
                "human",
                "Chapter {prev_chapter_human_num} Text:\n```\n{chapter_text}\n```\n\nProvide the concise summary:",
            ),
        ]
    )
    try:
        summary = invoke_llm_with_retry(
            summarizer_llm,
            prompt,
            {
                "prev_chapter_human_num": prev_chapter_human_num,
                "chapter_text": last_chapter_text,
            },
            parser=StrOutputParser(),
            max_retries=max_retries,
        )
        logger.info(
            f"Summary generated for Chapter {prev_chapter_human_num}: {summary[:100]}..."
        )
        return {"previous_chapter_summary": summary.strip()}
    except Exception as e:
        logger.error(
            f"Summarization failed for Ch {prev_chapter_human_num}. Placeholder used. Error: {e}",
            exc_info=True,
        )
        return {
            "previous_chapter_summary": (
                f"Error summarizing Chapter {prev_chapter_human_num}. Key events need recalling."
            )
        }


# --- Beat Planner Agent (NEW) ---
def beat_planner_agent(state: NovelWritingState, max_retries: int, llm_map: Dict):
    node_name = "Beat Planner Agent"
    logger.info(f"--- Entering {node_name} ---")
    current_chapter_idx = state["current_chapter_index"]
    chapter_index_human = current_chapter_idx + 1
    beat_planner_llm = llm_map["beat_planner"]
    chapter_goal = state.get("current_chapter_goal")
    previous_summary = state.get("previous_chapter_summary", "N/A")
    synopsis = state.get("novel_synopsis", "N/A")

    if not chapter_goal:
        return signal_error(
            state,
            node_name,
            f"No current chapter goal found for Chapter {chapter_index_human}.",
        )

    logger.info(
        f"Planning beats for Chapter {chapter_index_human} (Goal: {chapter_goal[:80]}...)"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are a chapter planner. Based on the overall synopsis, the summary of the previous chapter, and the specific goal for the current chapter, create a concise bulleted list of 3-6 key scenes or 'beats' that need to happen in this chapter to achieve its goal and advance the plot logically.

        Think step-by-step:
        1. Understand the starting point (previous summary).
        2. Understand the target endpoint (chapter goal for Chapter {chapter_index_human}).
        3. Identify the necessary steps/scenes to bridge the gap.
        4. Consider character actions/reactions needed.
        5. List the beats concisely.

        Overall Synopsis: {synopsis}
        Summary of Previous Chapter ({chapter_index_human - 1}): {previous_summary}
        Current Chapter ({chapter_index_human}) Goal: {chapter_goal}

        Output ONLY the bulleted list of key beats for Chapter {chapter_index_human}:""",
            ),
            ("human", "Generate the beat sheet for the current chapter."),
        ]
    )
    try:
        beat_sheet = invoke_llm_with_retry(
            beat_planner_llm,
            prompt,
            {},  # Vars are in system prompt
            parser=StrOutputParser(),
            max_retries=max_retries,
        )
        logger.info(
            f"Generated beat sheet for Chapter {chapter_index_human}:\n{beat_sheet}"
        )
        return {"chapter_beat_sheet": beat_sheet.strip()}
    except Exception as e:
        return signal_error(
            state,
            node_name,
            f"Failed to generate beat sheet for Chapter {chapter_index_human}.",
            e,
        )


# --- Drafter Agent ---
def drafter_agent(
    state: NovelWritingState, max_retries: int, vs_config: dict, llm_map: Dict
):
    node_name = "Drafter Agent"
    logger.info(f"--- Entering {node_name} ---")
    current_chapter_idx = state["current_chapter_index"]
    chapter_index_human = current_chapter_idx + 1
    drafter_llm = llm_map["drafter"]
    guidelines = state["writing_guidelines"]
    params = state["novel_parameters"]
    chapter_goal = state.get("current_chapter_goal")
    beat_sheet = state.get("chapter_beat_sheet")
    retrieved_context = ""

    if not chapter_goal:
        return signal_error(
            state, node_name, f"No chapter goal for Ch {chapter_index_human}."
        )
    if not beat_sheet:
        return signal_error(
            state, node_name, f"No beat sheet for Ch {chapter_index_human}."
        )

    logger.info(
        f"Drafting Chapter {chapter_index_human} (Goal: {chapter_goal[:80]}...)"
    )

    if vector_store and vs_config.get("use_vector_store", False):
        query = f"Context for Chapter {chapter_index_human} goal: {chapter_goal}. Beats: {beat_sheet[:100]}"
        retrieved_context = get_relevant_context_from_vector_store(
            query, max_chars=vs_config.get("retrieved_context_chars", 1500)
        )
        logger.info("Retrieved context from vector store for drafting.")

    previous_summary = state.get(
        "previous_chapter_summary", "This is the first chapter."
    )
    char_notes = state.get("character_notes", "N/A - Character notes missing.")
    synopsis = state.get("novel_synopsis", "N/A - Synopsis missing.")
    rewrite_instruction = ""
    if state.get("author_decision") == "Rewrite" and state.get("author_feedback"):
        rewrite_instruction = f"**IMPORTANT REWRITE INSTRUCTIONS (Address these while following beat sheet and goal for Chapter {chapter_index_human}):**\n{state['author_feedback']}\n------\n"
        logger.info(
            f"Incorporating author feedback for rewrite of Ch {chapter_index_human}."
        )

    system_prompt = f"""You are the novelist drafting Chapter {chapter_index_human} of '{params.get("novel_title", "Untitled Novel")}'.

CONTEXT:
- Overall Synopsis: {synopsis}
- Character Notes: {char_notes}
- Summary of Previous Chapter ({chapter_index_human - 1}): {previous_summary}
- Relevant Context from Past Chapters (if any): {retrieved_context if retrieved_context else "N/A"}

CURRENT TASK:
- Write the **full prose** for Chapter {chapter_index_human}.
- Chapter Goal: **{chapter_goal}**
- Chapter Beat Sheet (Follow these key scenes/events):
{beat_sheet}
{rewrite_instruction}
INSTRUCTIONS:
1.  Follow the Chapter Beat Sheet step-by-step to structure the chapter.
2.  Ensure the chapter fully achieves the specified Chapter Goal by the end.
3.  Write engaging, high-quality prose. Continuously and meticulously apply the **Overall Writing Guidelines** provided below.
4.  Think step-by-step internally as you write each scene based on the beats.
5.  Output *only* the complete chapter text. No summaries, notes, or introductions.

Overall Writing Guidelines:
{guidelines}

Begin writing the prose for Chapter {chapter_index_human} now:"""
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
    try:
        draft = invoke_llm_with_retry(
            drafter_llm, prompt, {}, parser=StrOutputParser(), max_retries=max_retries
        )
        logger.info(
            f"Draft generated for Chapter {chapter_index_human} (Length: {len(draft)} chars)."
        )
        return {
            "draft_chapter": draft,
            "author_feedback": None,
        }  # Clear feedback after use
    except Exception as e:
        return signal_error(
            state, node_name, f"Error during drafting Chapter {chapter_index_human}.", e
        )


# --- Editor Agent ---
def editor_agent(state: NovelWritingState, max_retries: int, llm_map: Dict):
    node_name = "Editor Agent"
    logger.info(f"--- Entering {node_name} ---")
    current_chapter_idx = state["current_chapter_index"]
    chapter_index_human = current_chapter_idx + 1
    editor_llm = llm_map["editor"]
    guidelines = state["writing_guidelines"]
    draft_chapter = state.get("draft_chapter")
    chapter_goal = state.get("current_chapter_goal")

    if not draft_chapter:
        return signal_error(
            state, node_name, f"No draft chapter for Ch {chapter_index_human}."
        )
    if not chapter_goal:
        return signal_error(
            state, node_name, f"No chapter goal for Ch {chapter_index_human} review."
        )

    logger.info(f"Editing draft for Chapter {chapter_index_human}")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are a meticulous novel editor reviewing the draft of Chapter {chapter_index_human}.
        Critically assess the draft based *only* on:
        1.  **Chapter Goal Achievement:** Does the draft successfully and clearly accomplish the specific goal? Is the goal fulfillment evident?
        2.  **Overall Writing Guidelines Adherence:** Evaluate rigorously against the guidelines, *especially* the 'Refinement Checks'. Focus on identifying specific flaws in: Pacing, Dialogue, Description, Plot Point Development, Conflict/Resolution, Exposition, Trope Usage, and Consistency.

        Provide **specific, actionable, numbered feedback points** targeting *only* areas needing improvement. Be precise.
        If, after careful review, no significant issues are found, respond with the single phrase: "No major issues found." Do NOT offer praise.

        Overall Writing Guidelines (Focus on Refinement Checks within):
        {guidelines}

        Chapter Goal (Chapter {chapter_index_human}): {{chapter_goal}}
        """,
            ),
            (
                "human",
                "Draft Chapter Text:\n```\n{draft}\n```\n\nProvide numbered editorial feedback points OR state 'No major issues found.'",
            ),
        ]
    )
    try:
        feedback = invoke_llm_with_retry(
            editor_llm,
            prompt,
            {"chapter_goal": chapter_goal, "draft": draft_chapter},
            parser=StrOutputParser(),
            max_retries=max_retries,
        )
        feedback_clean = feedback.strip()
        if feedback_clean.lower() == "no major issues found.":
            feedback_for_writer = "No major issues identified by the editor."
            logger.info(
                f"Editor Result for Chapter {chapter_index_human}: No major issues found."
            )
        else:
            feedback_for_writer = feedback_clean
            logger.info(
                f"Editor Result for Chapter {chapter_index_human}: Feedback provided."
            )
        return {"editor_feedback": feedback_for_writer}
    except Exception as e:
        return signal_error(
            state, node_name, f"Error during editing Chapter {chapter_index_human}.", e
        )


# --- Writer Agent ---
def writer_agent(state: NovelWritingState, max_retries: int, llm_map: Dict):
    node_name = "Writer Agent"
    logger.info(f"--- Entering {node_name} ---")
    current_chapter_idx = state["current_chapter_index"]
    chapter_index_human = current_chapter_idx + 1
    writer_llm = llm_map["writer"]
    guidelines = state["writing_guidelines"]
    draft_chapter = state.get("draft_chapter")
    editor_feedback = state.get("editor_feedback")

    if not draft_chapter:
        return signal_error(
            state, node_name, f"Missing draft for Ch {chapter_index_human}."
        )
    if not editor_feedback:
        logger.warning(
            f"Missing editor feedback for Ch {chapter_index_human}. General polish pass."
        )
        editor_feedback = "No specific feedback. Perform general polish: prose clarity, flow, consistency, sensory details, guideline adherence."

    logger.info(f"Revising Chapter {chapter_index_human} based on editor feedback.")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are a skilled novel writer revising Chapter {chapter_index_human}.
        Revise the Original Draft based *specifically* on the Editor's Feedback. Address each point.
        If feedback was "No major issues...", perform a light polish (flow, word choice, minor errors), adhering to guidelines.

        Output *only* the complete, refined chapter text for Chapter {chapter_index_human}. No notes.

        Overall Writing Guidelines:
        {guidelines}

        Editor Feedback to Address:
        {{feedback}}

        Original Draft:
        """,
            ),
            (
                "human",
                "{draft}\n\nBased *only* on Original Draft and Editor Feedback, write the complete Refined Chapter {chapter_index_human} now:",
            ),
        ]
    )
    try:
        refined = invoke_llm_with_retry(
            writer_llm,
            prompt,
            {
                "draft": draft_chapter,
                "feedback": editor_feedback,
                "chapter_index_human": chapter_index_human,
            },
            parser=StrOutputParser(),
            max_retries=max_retries,
        )
        logger.info(
            f"Writer refined Chapter {chapter_index_human} (Length: {len(refined)} chars)."
        )
        return {"refined_chapter": refined, "editor_feedback": None}  # Clear feedback
    except Exception as e:
        return signal_error(
            state,
            node_name,
            f"Error during writing/refining Chapter {chapter_index_human}.",
            e,
        )


# --- Consistency Checker Agent (NEW) ---
def consistency_checker_agent(
    state: NovelWritingState, max_retries: int, vs_config: dict, llm_map: Dict
):
    node_name = "Consistency Checker Agent"
    logger.info(f"--- Entering {node_name} ---")
    current_chapter_idx = state["current_chapter_index"]
    chapter_index_human = current_chapter_idx + 1
    consistency_llm = llm_map["consistency"]
    refined_chapter = state.get("refined_chapter")
    synopsis = state.get("novel_synopsis", "N/A")
    char_notes = state.get("character_notes", "N/A")
    previous_summary = state.get("previous_chapter_summary", "N/A")
    retrieved_context = ""

    if not refined_chapter:
        return signal_error(
            state,
            node_name,
            f"No refined chapter for Ch {chapter_index_human} consistency check.",
        )

    logger.info(f"Checking consistency for Chapter {chapter_index_human}.")
    if vector_store and vs_config.get("use_vector_store", False):
        query = f"Facts, character states, plot points before Chapter {chapter_index_human}. Synopsis: {synopsis[:100]}. Chars: {char_notes[:100]}. PrevSum: {previous_summary[:100]}."
        retrieved_context = get_relevant_context_from_vector_store(
            query,
            k=5,
            max_chars=vs_config.get("retrieved_context_chars_consistency", 2000),
        )
        logger.info("Retrieved context for consistency check.")

    condensed_context = f"""Synopsis Snippet: {synopsis[:500]}...
Character Notes Snippet: {char_notes[:500]}...
Previous Chapter Summary: {previous_summary}
Retrieved Context Snippets (if available): {retrieved_context if retrieved_context else "N/A"}"""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are a continuity editor. Read Refined Chapter {chapter_index_human} and check *only* for inconsistencies against provided Context (synopsis, character notes, previous summary, retrieved context).

        Look for contradictions related to: established plot points, character knowledge/abilities/appearance/motivations, world rules, setting details, timeline.

        Output:
        - If NO inconsistencies: "Consistent".
        - If inconsistencies: List them as bullet points, explaining (e.g., "- Character X claims not to know magic, but used a spell in Chapter 3's summary.").
        Focus *only* on factual consistency.

        Context for Checking:
        {condensed_context}
        """,
            ),
            (
                "human",
                "Refined Chapter {chapter_index_human} to Check:\n```\n{refined_chapter}\n```\n\nReport consistency issues or state 'Consistent'.",
            ),
        ]
    )
    try:
        report = invoke_llm_with_retry(
            consistency_llm,
            prompt,
            {
                "refined_chapter": refined_chapter,
                "chapter_index_human": chapter_index_human,
            },  # Vars in system prompt
            parser=StrOutputParser(),
            max_retries=max_retries,
        )
        report_clean = report.strip()
        logger.info(
            f"Consistency Check Report for Chapter {chapter_index_human}: {report_clean[:150]}..."
        )
        return {"consistency_report": report_clean}
    except Exception as e:
        logger.error(
            f"Consistency check failed for Ch {chapter_index_human}. No report. Error: {e}",
            exc_info=True,
        )
        return {"consistency_report": f"Error during consistency check: {e}"}


# --- Conditional Edge Functions ---
def route_after_author(state: NovelWritingState):
    logger.debug("--- Routing after Author ---")
    author_decision = state.get("author_decision")
    last_error = state.get(
        "last_error"
    )  # Check if Author node itself signaled an error

    if author_decision == "Error":
        logger.error(f"Routing to END due to error signaled. Error: {last_error}")
        return END

    # This case handles transition from planning to first chapter summarization
    is_initial_planning_complete = (
        state.get("novel_synopsis") is not None
        and state.get("current_chapter_index") == 0
        and author_decision is None
    )  # Author decision is None after successful planning

    if is_initial_planning_complete:
        logger.info(
            "Initial plan generated. Proceeding to Summarizer for Chapter 1 context."
        )
        if not state.get("current_chapter_goal"):  # Should be set by planning
            logger.error("Routing Error: Plan ok but no Chapter 1 goal. Ending.")
            state["author_decision"] = "Error"  # Force error state
            state["last_error"] = (
                "Routing Error: Initial plan completed but no Chapter 1 goal was set."
            )
            return END
        return "summarizer_agent"

    # This handles review decisions
    current_chapter_idx = state.get(
        "current_chapter_index", 0
    )  # Default to 0 if not set
    chapter_goal_list = state.get("chapter_goal_list", [])

    if author_decision == "Approved":
        logger.info("Author decision: Approved.")
        if current_chapter_idx < len(chapter_goal_list):  # More chapters to go
            logger.info(
                "Chapter approved. Routing to Summarizer for next chapter context."
            )
            return "summarizer_agent"
        else:  # All chapters done
            logger.info("Final chapter approved. Novel finished. Routing to End.")
            return END
    elif author_decision == "Rewrite":
        logger.info(
            f"Author decision: Rewrite Chapter {current_chapter_idx + 1}. Routing back to Summarizer."
        )
        return "summarizer_agent"
    else:  # Should not happen if author_agent logic is correct
        err_msg = f"Routing Error: Unexpected state after Author. Decision: {author_decision}, Index: {current_chapter_idx}"
        logger.error(err_msg)
        state["author_decision"] = "Error"
        state["last_error"] = err_msg
        return END


def route_after_consistency_check(state: NovelWritingState):
    logger.debug("--- Routing after Consistency Check ---")
    if (
        state.get("author_decision") == "Error"
    ):  # Check if checker itself errored via signal_error
        logger.error(
            f"Routing to END due to error during/before Consistency Check: {state.get('last_error')}"
        )
        return END
    if state.get("refined_chapter") is None:
        err_msg = "Routing Error: Refined chapter missing after Consistency Check."
        logger.error(err_msg)
        state["author_decision"] = "Error"
        state["last_error"] = err_msg
        return END
    logger.info(
        f"Consistency check complete for Chapter {state['current_chapter_index'] + 1}. Proceeding to Author review."
    )
    return "author_agent"


# --- Build the Graph ---
def build_graph(config: Dict, llm_map: Dict) -> StateGraph:
    workflow = StateGraph(NovelWritingState)
    from functools import partial

    max_retries = config["workflow_settings"]["max_llm_retries"]
    vs_config = config.get("vector_store_settings", {})

    workflow.add_node(
        "author_agent", partial(author_agent, max_retries=max_retries, llm_map=llm_map)
    )
    workflow.add_node(
        "summarizer_agent",
        partial(summarizer_agent, max_retries=max_retries, llm_map=llm_map),
    )
    workflow.add_node(
        "beat_planner_agent",
        partial(beat_planner_agent, max_retries=max_retries, llm_map=llm_map),
    )
    workflow.add_node(
        "drafter_agent",
        partial(
            drafter_agent, max_retries=max_retries, vs_config=vs_config, llm_map=llm_map
        ),
    )
    workflow.add_node(
        "editor_agent", partial(editor_agent, max_retries=max_retries, llm_map=llm_map)
    )
    workflow.add_node(
        "writer_agent", partial(writer_agent, max_retries=max_retries, llm_map=llm_map)
    )
    workflow.add_node(
        "consistency_checker_agent",
        partial(
            consistency_checker_agent,
            max_retries=max_retries,
            vs_config=vs_config,
            llm_map=llm_map,
        ),
    )

    workflow.set_entry_point("author_agent")
    workflow.add_conditional_edges(
        "author_agent",
        route_after_author,
        {END: END, "summarizer_agent": "summarizer_agent"},
    )
    workflow.add_edge("summarizer_agent", "beat_planner_agent")
    workflow.add_edge("beat_planner_agent", "drafter_agent")
    workflow.add_edge("drafter_agent", "editor_agent")
    workflow.add_edge("editor_agent", "writer_agent")
    workflow.add_edge("writer_agent", "consistency_checker_agent")
    workflow.add_conditional_edges(
        "consistency_checker_agent",
        route_after_consistency_check,
        {END: END, "author_agent": "author_agent"},
    )
    logger.info("Graph definition complete.")
    return workflow


# --- Saving Final Output ---
def save_final_novel(state: NovelWritingState, config: Dict):
    logger.info("--- Preparing Final Output ---")
    output_dir = config["workflow_settings"]["output_directory"]
    output_format = config["workflow_settings"].get("output_format", "txt").lower()
    os.makedirs(output_dir, exist_ok=True)

    final_params = state.get("novel_parameters", {})
    final_completed_list = state.get("completed_chapters", [])
    title = final_params.get("novel_title", "Untitled Novel")
    safe_title = re.sub(r"[^\w\-_\. ]", "_", title).replace(" ", "_").lower()

    if not final_completed_list:
        logger.warning("No completed chapters found to save.")
        param_filename = os.path.join(output_dir, f"{safe_title}_params_only.yaml")
        try:
            with open(param_filename, "w", encoding="utf-8") as f:
                yaml.dump(final_params, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved novel parameters to: {param_filename}")
        except Exception as e:
            logger.error(f"Error saving parameters file: {e}")
        return

    filename_base = os.path.join(output_dir, f"{safe_title}_complete")
    novel_filename = f"{filename_base}.{output_format}"
    try:
        logger.info(
            f"Saving novel content to: {novel_filename} (Format: {output_format})"
        )
        with open(novel_filename, "w", encoding="utf-8") as f:
            f.write(f"# Title: {title}\n\n")
            f.write(f"**Genre:** {final_params.get('genre', 'N/A')}\n")
            # ... (add other params as desired) ...
            f.write("## Characters\n\n")
            for char_info in final_params.get("characters", []):
                f.write(f"### {char_info.get('name', 'N/A')}\n")
                f.write(
                    f"- **Motivation/Arc:** {char_info.get('motivation_arc', 'N/A')}\n\n"
                )
            f.write("## Synopsis\n\n")
            f.write(f"{state.get('novel_synopsis', 'N/A')}\n\n---\n\n")
            for i, chapter_content in enumerate(final_completed_list):
                f.write(f"## Chapter {i + 1}\n\n{chapter_content.strip()}\n\n---\n\n")
        logger.info("Novel content saved successfully.")
        if final_completed_list:
            logger.info("--- First Completed Chapter (First 300 Chars) ---")
            print(final_completed_list[0][:300].strip() + "...")
            if len(final_completed_list) > 1:
                logger.info("--- Last Completed Chapter (First 300 Chars) ---")
                print(final_completed_list[-1][:300].strip() + "...")
    except Exception as e:
        logger.error(
            f"Error saving novel content to file {novel_filename}: {e}", exc_info=True
        )


def save_intermediate_novel(state: NovelWritingState, config: Dict):
    output_dir = config["workflow_settings"]["output_directory"]
    os.makedirs(output_dir, exist_ok=True)
    final_params = state.get("novel_parameters", {})
    completed_list = state.get("completed_chapters", [])
    # current_chapter_idx is the index of the chapter JUST APPROVED AND ADDED.
    # So, if current_chapter_index is 1, it means chapter 1 (list index 0) was just completed.
    # The number of completed chapters is len(completed_list).
    # The last completed chapter's human number is len(completed_list).
    num_chapters_done = len(completed_list)

    if not completed_list:
        return
    title = final_params.get("novel_title", "Untitled Novel")
    safe_title = re.sub(r"[^\w\-_\. ]", "_", title).replace(" ", "_").lower()
    filename = os.path.join(
        output_dir, f"{safe_title}_inprogress_ch{num_chapters_done}.txt"
    )
    try:
        logger.info(
            f"Saving intermediate progress (up to Chapter {num_chapters_done}) to: {filename}"
        )
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Title: {title}\nGenre: {final_params.get('genre', 'N/A')}\n")
            f.write(f"Chapters Completed: {num_chapters_done}\n{'=' * 30}\n\n")
            for i, chapter_content in enumerate(completed_list):
                f.write(
                    f"--- Chapter {i + 1} ---\n\n{chapter_content.strip()}\n\n{'=' * 30}\n\n"
                )
    except Exception as e:
        logger.error(f"Error saving intermediate novel progress: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph Novel Writer V3")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--thread-id", type=str, default=None, help="Unique ID for the writing session."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Attempt to resume from checkpoint."
    )
    parser.add_argument("--genre", type=str, help="Override genre.")
    parser.add_argument("--setting", type=str, help="Override setting.")
    parser.add_argument("--concept", type=str, help="Override core concept.")
    parser.add_argument("--tone", type=str, help="Override tone.")
    parser.add_argument("--target-audience", type=str, help="Override target audience.")
    parser.add_argument(
        "--output-format", choices=["txt", "md"], help="Override output format."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_format:
        config["workflow_settings"]["output_format"] = args.output_format
    output_dir = config["workflow_settings"]["output_directory"]
    setup_file_logging(output_dir)

    logger.info("Initializing LLMs...")
    llm_configs = config["llm_settings"]
    temps = llm_configs["temperature"]
    try:
        llm_map = {
            "author": ChatOllama(
                model=llm_configs["author_model"], temperature=temps.get("author", 0.7)
            ),
            "planning": ChatOllama(
                model=llm_configs["planning_model"],
                temperature=temps.get("planning", 0.5),
            ),
            "summarizer": ChatOllama(
                model=llm_configs["summarizer_model"],
                temperature=temps.get("summarizer", 0.3),
            ),
            "beat_planner": ChatOllama(
                model=llm_configs["beat_planner_model"],
                temperature=temps.get("beat_planner", 0.5),
            ),
            "drafter": ChatOllama(
                model=llm_configs["drafter_model"],
                temperature=temps.get("drafter", 0.8),
            ),
            "editor": ChatOllama(
                model=llm_configs["editor_model"], temperature=temps.get("editor", 0.4)
            ),
            "writer": ChatOllama(
                model=llm_configs["writer_model"], temperature=temps.get("writer", 0.6)
            ),
            "consistency": ChatOllama(
                model=llm_configs["consistency_model"],
                temperature=temps.get("consistency", 0.2),
            ),
            "char_gen": ChatOllama(
                model=llm_configs["char_gen_model"],
                temperature=temps.get("char_gen", 0.7),
            ),
        }
        logger.info("Ollama models initialized.")
    except Exception as e:
        logger.error(f"Error initializing Ollama models: {e}", exc_info=True)
        exit(1)

    initialize_vector_store(config)
    checkpoint_db_path = os.path.join(
        output_dir, config["workflow_settings"]["checkpoint_db"]
    )

    # Novel data parameters (genres, settings, etc.) are loaded via load_novel_data()
    # and passed to generate_novel_parameters if it's a new run.
    # The global constants serve as defaults within load_novel_data.

    with SqliteSaver.from_conn_string(checkpoint_db_path) as memory:
        logger.info(f"Checkpoint saver context entered (DB: {checkpoint_db_path})")
        app = build_graph(config, llm_map).compile(checkpointer=memory)
        logger.info("Graph compiled successfully.")

        thread_id = args.thread_id
        is_resuming = args.resume
        if not thread_id:
            if is_resuming:
                logger.error("Cannot resume without --thread-id.")
                exit(1)
            thread_id = f"novel-session-{int(time.time())}"
            logger.info(f"Generated new session thread_id: {thread_id}")
        else:
            logger.info(f"Using thread_id: {thread_id}")
        run_config = {"configurable": {"thread_id": thread_id}}

        initial_state_input = {}  # Input for app.stream
        final_state_values = {}  # To store the latest full state

        if is_resuming:
            existing_state = app.get_state(run_config)
            if existing_state and existing_state.values:
                logger.info(
                    f"Resuming workflow from checkpoint for thread_id: {thread_id}"
                )
                logger.info(
                    f" > Resuming at Chapter Index: {existing_state.values.get('current_chapter_index', 'N/A')}"
                )
                logger.info(
                    f" > Chapters Completed: {len(existing_state.values.get('completed_chapters', []))}"
                )
                final_state_values = existing_state.values  # Start with this
                # No initial_state_input needed when resuming, LangGraph handles it
            else:
                logger.error(
                    f"No checkpoint found for thread_id '{thread_id}'. Cannot resume."
                )
                exit(1)
        else:  # New run
            logger.info(
                f"Preparing initial state for new run with thread_id: {thread_id}"
            )
            novel_data_params = (
                load_novel_data()
            )  # Load parameter lists (JSON or defaults)
            novel_params = generate_novel_parameters(
                config, args, llm_map.get("char_gen"), param_lists=novel_data_params
            )
            initial_state_input = {
                "novel_parameters": novel_params,
                "writing_guidelines": novel_params["writing_guidelines"],
                "max_revisions": config["workflow_settings"][
                    "max_revisions_per_chapter"
                ],
                # Other initial fields like novel_synopsis=None are implicitly handled by TypedDict
            }
            final_state_values = (
                initial_state_input.copy()
            )  # Track with this initial state
            logger.info("\n--- STARTING NEW NOVEL WRITING WORKFLOW ---")
            print(f"Title: {novel_params.get('novel_title', 'N/A')}")  # etc.

        # Recursion limit calculation
        max_revisions_per_chapter = config["workflow_settings"][
            "max_revisions_per_chapter"
        ]
        num_chapters_planned = config["workflow_settings"].get(
            "max_chapters_estimate", 35
        )
        if (
            is_resuming
            and final_state_values.get("chapter_goal_list")
            and isinstance(final_state_values["chapter_goal_list"], list)
        ):
            if len(final_state_values["chapter_goal_list"]) > 0:
                num_chapters_planned = len(final_state_values["chapter_goal_list"])
        elif not is_resuming and initial_state_input.get(
            "novel_parameters"
        ):  # For new run, planning hasn't happened yet
            pass  # num_chapters_planned remains the estimate, will be refined after planning if desired

        estimated_steps_per_chapter = 7 * (
            1 + max_revisions_per_chapter
        )  # Plan,Sum,Beat,Draft,Edit,Write,Cons,Author
        estimated_total_steps = (
            num_chapters_planned * estimated_steps_per_chapter + 20
        )  # Buffer
        logger.info(
            f"Setting estimated recursion limit: {estimated_total_steps} (for ~{num_chapters_planned} chaps)"
        )
        run_config["recursion_limit"] = estimated_total_steps

        try:
            logger.info(f"Starting/Resuming workflow stream for thread_id: {thread_id}")
            stream_argument = initial_state_input if not is_resuming else None

            for step, s_update in enumerate(
                app.stream(stream_argument, run_config, stream_mode="updates")
            ):
                node_name = list(s_update.keys())[0]
                # state_diff = s_update[node_name] # if needed for fine-grained logging
                logger.info(f"--- Step {step + 1}: Completed Node '{node_name}' ---")

                current_full_state = app.get_state(
                    run_config
                )  # Get full state after update
                if not current_full_state or not current_full_state.values:
                    logger.error(
                        f"State is empty after node {node_name}. This should not happen. Terminating."
                    )
                    final_state_values["last_error"] = (
                        f"State became empty after node {node_name}."
                    )
                    final_state_values["author_decision"] = "Error"
                    break

                final_state_values = current_full_state.values  # Update our tracker

                if (
                    node_name == "author_agent"
                    and final_state_values.get("author_decision") == "Approved"
                ):
                    # Check if a chapter was actually completed (index might have advanced, or completed_chapters list grew)
                    # Intermediate save is based on len(completed_chapters)
                    save_intermediate_novel(final_state_values, config)

                if final_state_values.get("author_decision") == "Error":
                    logger.error(
                        f"Error state detected after node '{node_name}'. Stopping. Error: {final_state_values.get('last_error', 'N/A')}"
                    )
                    break

            # After loop, get final state one last time
            final_state_values = app.get_state(run_config).values

        except Exception as e:
            logger.error("\n--- WORKFLOW EXECUTION ERROR ---")
            logger.error(f"Error during graph execution: {e}", exc_info=True)
            # Try to get the state before error for saving
            try:
                final_state_values = app.get_state(run_config).values
            except Exception as se:
                logger.error(f"Could not retrieve state after graph error: {se}")
                if not final_state_values:  # If it was never set
                    final_state_values = {
                        "author_decision": "Error",
                        "last_error": f"Graph execution failed: {e}",
                    }
                elif "last_error" not in final_state_values:
                    final_state_values["last_error"] = f"Graph execution failed: {e}"
                    final_state_values["author_decision"] = "Error"

        finally:
            logger.info("\n--- WORKFLOW FINISHED (Inside Finally Block) ---")
            if final_state_values:
                # Determine final status for logging
                title = final_state_values.get("novel_parameters", {}).get(
                    "novel_title", "Untitled Novel"
                )
                author_decision = final_state_values.get("author_decision")
                num_chapters_completed = len(
                    final_state_values.get("completed_chapters", [])
                )
                last_error_msg = final_state_values.get("last_error")
                current_c_idx = final_state_values.get("current_chapter_index", 0)
                c_goal_list_len = len(final_state_values.get("chapter_goal_list", []))

                if author_decision == "Error":
                    logger.error(
                        f"FINAL STATUS: Novel '{title}' generation terminated due to error. Last Error: {last_error_msg}"
                    )
                # Successful completion: last decision was approve AND current_index is now at or beyond goal list length
                elif (
                    author_decision == "Approved"
                    and current_c_idx >= c_goal_list_len
                    and num_chapters_completed > 0
                    and c_goal_list_len > 0
                ):
                    logger.info(
                        f"FINAL STATUS: Novel '{title}' successfully completed with {num_chapters_completed} chapters."
                    )
                elif num_chapters_completed > 0:
                    logger.warning(
                        f"FINAL STATUS: Novel '{title}' stopped after {num_chapters_completed} chapters. Last Decision: {author_decision}. Error: {last_error_msg}"
                    )
                else:
                    logger.warning(
                        f"FINAL STATUS: Novel '{title}' stopped before any chapters completed. Last Decision: {author_decision}. Error: {last_error_msg}"
                    )

                save_final_novel(final_state_values, config)
            else:  # final_state_values is empty
                logger.error(
                    "Workflow finished without producing a retrievable final state. Check logs and checkpoints."
                )
            logger.info("Checkpoint context exiting.")
    logger.info("-------------------------\nScript End.")
