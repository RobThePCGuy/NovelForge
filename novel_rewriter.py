# novel_rewriter.py
import os
import argparse
import yaml
import logging
import time
import re
import json  # For novel_data.json

# Pydantic
from pydantic import ValidationError

# Langchain & Langgraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver  # Corrected import
from langchain_ollama import ChatOllama

# Vector Store (Optional, for context from *already rewritten* chapters)
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    VECTOR_STORE_ENABLED = True
except ImportError:
    print(
        "Warning: Vector Store libraries not found. Context from rewritten chapters will be limited."
    )
    VECTOR_STORE_ENABLED = False

from typing import TypedDict, List, Optional, Dict, Any

# --- Configuration & Data Loading ---
DEFAULT_CONFIG_PATH = "config_rewrite.yaml"  # Suggest different config for clarity
DEFAULT_NOVEL_DATA_PATH = "novel_data.json"


def load_config(config_path=DEFAULT_CONFIG_PATH):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Config loaded from {config_path}")
        # MODIFIED VALIDATION TO INCLUDE rewrite_settings
        if not all(
            k in config
            for k in ["llm_settings", "workflow_settings", "rewrite_settings"]
        ):
            raise ValueError(
                "Config file missing required top-level keys: llm_settings, workflow_settings, rewrite_settings."
            )
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file '{config_path}' not found. Exiting.")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        exit(1)


def load_novel_data_json(file_path=DEFAULT_NOVEL_DATA_PATH) -> Dict[str, Any]:
    """Loads structured data from novel_data.json. Required if features use it."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Novel data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        logger.warning(
            f"'{file_path}' not found. Any features relying on this data may not work as expected or might use empty defaults."
        )
        return {}  # Return empty dict, let consuming functions handle missing keys
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {file_path}. Returning empty data.")
        return {}
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}. Returning empty data.")
        return {}


# --- Logging Setup ---
log_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("NovelRewriter")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
file_handler = None


def setup_file_logging(log_dir):
    global file_handler
    if file_handler:
        logger.removeHandler(file_handler)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "novel_rewrite.log")
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info(f"File logging at: {log_file}")


# --- LLM Invocation Helper ---
def invoke_llm_with_retry(llm, prompt_template, input_data, parser=None, max_retries=1):
    chain_no_parser = prompt_template | llm | StrOutputParser()
    last_exception = None
    raw_response = ""
    for attempt in range(max_retries + 1):
        try:
            raw_response = chain_no_parser.invoke(input_data)
            if not raw_response or not raw_response.strip():
                raise ValueError("LLM returned empty/whitespace response.")
            if parser:
                result = parser.invoke(raw_response)
                if (
                    result is None and raw_response.strip()
                ):  # Parser gave None from non-empty
                    raise ValueError(
                        f"Parser returned None from raw response: '{raw_response[:100]}...'"
                    )
                return result
            return raw_response
        except (ValidationError, ValueError) as ve:
            logger.warning(
                f"LLM/Parse failed (Attempt {attempt + 1}/{max_retries + 1}): {ve}"
            )
            logger.debug(
                f"Raw LLM Response on parsing failure: '{raw_response[:500]}...'"
            )
            last_exception = ve
        except Exception as e:
            logger.warning(
                f"LLM invocation failed unexpectedly (Attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__} - {e}"
            )
            if "Input to ChatPromptTemplate is missing variables" in str(
                e
            ):  # Specific logging for this common issue
                logger.error(f"Prompt Key Error: {e}")  # Log the full key error
            last_exception = e
        if attempt == max_retries:
            logger.error(
                f"Max retries reached for LLM. Last error: {type(last_exception).__name__}"
            )
            raise last_exception
        time.sleep(1.5**attempt)
    raise RuntimeError(
        f"LLM invocation failed critically after retries. Last exception: {last_exception}"
    )


# --- Novel Loading ---
def load_original_novel_text(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"Original novel loaded from: {file_path}")
        return content
    except FileNotFoundError:
        logger.error(f"Original novel file not found: {file_path}. Exiting.")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading original novel: {e}")
        exit(1)


def split_text_into_chapters(text: str, delimiter_regex: str) -> List[str]:
    if not text:
        return []
    # The delimiter regex should capture the delimiter itself to split by it.
    # Example: r"(\n\#\# Chapter \d+\n|\n\#\#\# Chapter \d+\n)"
    # Using re.split and then filtering out empty strings and delimiter matches.

    # A simpler approach: if delimiter marks the START of a chapter.
    # Example delimiter: "## Chapter " (note the space)
    # We find all occurrences and slice the text.

    chapters = []
    if not delimiter_regex:  # Treat as single chapter if no delimiter
        logger.warning(
            "No chapter delimiter provided. Treating entire text as one chapter."
        )
        return [text.strip()] if text.strip() else []

    parts = re.split(
        f"({re.escape(delimiter_regex)}\\s*\\d+|{re.escape(delimiter_regex)}\\s*[IVXLCDM]+)",
        text,
        flags=re.IGNORECASE,
    )

    current_chapter_content = ""
    if parts[0].strip():  # Content before the first chapter delimiter
        current_chapter_content = parts[0].strip()

    idx = 1
    while idx < len(parts):
        header = parts[idx]  # This is the delimiter part e.g. "## Chapter 1"
        content = parts[idx + 1] if idx + 1 < len(parts) else ""

        if current_chapter_content and header:  # Start of a new chapter
            chapters.append(current_chapter_content.strip())
            current_chapter_content = (
                header.strip() + "\n" + content.strip()
            )  # Prepend header to new chapter
        else:  # Continuation or the very first segment
            current_chapter_content += (header if header else "") + (
                content if content else ""
            )
        idx += 2

    if current_chapter_content.strip():
        chapters.append(current_chapter_content.strip())

    if not chapters and text.strip():  # Fallback if split didn't work but text exists
        logger.warning(
            f"Delimiter splitting with '{delimiter_regex}' yielded no chapters, but text exists. Treating as one chapter."
        )
        chapters = [text.strip()]

    logger.info(
        f"Split novel into {len(chapters)} chapters using delimiter pattern: '{delimiter_regex}'"
    )
    return [ch for ch in chapters if ch]


# --- State Definition ---
class RewriteState(TypedDict, total=False):
    novel_parameters: Dict[str, Any]  # Title of rewrite, target genre/tone, etc.
    writing_guidelines: str  # Base guidelines + improvement focus
    improvement_focus: str

    original_novel_path: str
    chapter_delimiter: str
    original_chapters_text: List[str]
    rewritten_chapters_text: List[str]

    current_chapter_index: int
    current_original_chapter_text: Optional[str]
    current_rewritten_chapter_draft: Optional[str]
    editor_feedback: Optional[str]
    refined_chapter: Optional[str]  # After refiner agent
    consistency_report: Optional[str]
    author_feedback: Optional[str]
    author_decision: Optional[str]  # "Approved", "Rewrite", "Error"

    revision_cycles: int
    max_revisions: int
    last_error: Optional[str]
    previous_rewritten_chapter_summary: Optional[str]


# --- Vector Store (for rewritten content) ---
vector_store = None
embedding_function = None
text_splitter_vs = None  # Renamed to avoid conflict with other text_splitters


def initialize_vector_store_for_rewrite(config: Dict):
    global vector_store, embedding_function, text_splitter_vs
    vs_config = config.get("vector_store_settings", {})
    use_vs = vs_config.get("use_vector_store", False) and VECTOR_STORE_ENABLED

    if not use_vs:
        logger.warning(
            "Vector store for rewritten chapters disabled by config or missing libraries."
        )
        vector_store = None
        return
    try:
        logger.info("Initializing vector store for rewritten chapters...")
        emb_model = vs_config.get("embedding_model", "all-MiniLM-L6-v2")
        collection = (
            vs_config.get("collection_name", "rewritten_novel_chaps") + "_rewrite"
        )
        output_dir = config["workflow_settings"]["output_directory"]
        persist_dir = os.path.join(output_dir, "vector_store_rewrite")
        os.makedirs(persist_dir, exist_ok=True)

        embedding_function = SentenceTransformerEmbeddings(model_name=emb_model)
        vector_store = Chroma(
            collection_name=collection,
            embedding_function=embedding_function,
            persist_directory=persist_dir,
        )
        text_splitter_vs = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        logger.info(
            f"Vector store for rewrite initialized (Collection: {collection}, Dir: {persist_dir})"
        )
    except Exception as e:
        logger.error(f"Failed to init vector store for rewrite: {e}", exc_info=True)
        vector_store = None


def add_rewritten_chapter_to_vs(chapter_idx: int, chapter_text: str, summary: str):
    if not vector_store or not text_splitter_vs:
        return
    try:
        logger.info(f"Adding REWRITTEN Chapter {chapter_idx + 1} to vector store...")
        docs = text_splitter_vs.create_documents(
            [chapter_text],
            metadatas=[{"type": "rewritten_chapter", "chapter_index": chapter_idx}],
        )
        summary_docs = text_splitter_vs.create_documents(
            [summary],
            metadatas=[{"type": "rewritten_summary", "chapter_index": chapter_idx}],
        )
        if docs:
            vector_store.add_documents(docs)
        if summary_docs:
            vector_store.add_documents(summary_docs)
        logger.info(f"REWRITTEN Chapter {chapter_idx + 1} added to VS.")
    except Exception as e:
        logger.error(
            f"Failed to add REWRITTEN Ch {chapter_idx + 1} to VS: {e}", exc_info=True
        )


def get_context_from_rewritten_vs(query: str, k: int = 3, max_chars: int = 1500) -> str:
    if not vector_store:
        return "No context from rewritten chapters vector store."
    try:
        docs = vector_store.similarity_search(query, k=k)
        context = "".join(
            [
                f"[Context from REWRITTEN Chapter {d.metadata.get('chapter_index', -1) + 1}]:\n{d.page_content}\n...\n"
                for d in docs
            ]
        )
        return (
            context[:max_chars]
            if context
            else "No relevant context from rewritten chapters."
        )
    except Exception as e:
        logger.error(f"Failed to get context from rewritten VS: {e}", exc_info=True)
        return "Error retrieving context from rewritten VS."


# --- Agent Node Functions ---
def signal_error_node(
    state: RewriteState, node_name: str, message: str, e: Optional[Exception] = None
) -> Dict:
    full_message = f"Error in {node_name}: {message}"
    if e:
        full_message += f" | Exception: {type(e).__name__} - {e}"
    logger.error(full_message, exc_info=True if e else False)
    return {"author_decision": "Error", "last_error": full_message}


def setup_rewrite_planner_node(
    state: RewriteState, config: Dict, novel_data: Dict
) -> Dict[str, Any]:
    node_name = "Rewrite Planner"
    logger.info(f"--- Entering {node_name} ---")
    try:
        rewrite_cfg = config["rewrite_settings"]
        novel_path = rewrite_cfg["input_novel_file"]
        delimiter = rewrite_cfg["chapter_delimiter"]
        improvement_focus = rewrite_cfg.get(
            "overall_improvement_focus", "General quality and stylistic improvements."
        )

        original_text = load_original_novel_text(novel_path)
        original_chapters = split_text_into_chapters(original_text, delimiter)

        if not original_chapters:
            return signal_error_node(
                state, node_name, "No chapters found in the original novel."
            )

        # Base parameters for the rewrite project
        params = {
            "novel_title": f"Rewritten - {os.path.basename(novel_path)}",
            "target_genre": rewrite_cfg.get(
                "target_genre_override", "Same as original"
            ),
            "target_tone": rewrite_cfg.get(
                "target_tone_override", "Same as original or enhanced"
            ),
            # Potentially use novel_data for random choices if some aspects are to be changed randomly
            # Example: If tone needs to be more "ominous", pick a random "ominous" variant from novel_data['tones']
        }
        tones_list = novel_data.get("tones", [])  # Get from novel_data.json
        if params["target_tone"] == "Same as original or enhanced" and tones_list:
            # This is just an example if you wanted to pick a random tone for enhancement.
            # For a focused rewrite, you might set a specific target tone in config.
            pass  # params["target_tone"] = random.choice(tones_list)

        guidelines = f"""**Overall Writing Guidelines for This Rewrite:**
        - Target Genre (if specified): {params["target_genre"]}
        - Target Tone (if specified): {params["target_tone"]}
        - Preserve the core plot, characters, and sequence of events from the original chapter unless the improvement focus explicitly states otherwise.
        - Maintain character integrity: personalities, core motivations (unless being evolved by improvement focus), and established histories from the original text should be respected.
        - Ensure logical flow and consistency with previously rewritten material.
        - Apply standard best practices for prose: show don't tell, use vivid sensory details, ensure varied sentence structure, write clear and engaging dialogue.
        - Avoid introducing new plot holes or inconsistencies.

        **User's Core Improvement Focus for this Entire Rewrite Pass (Apply to every chapter):**
        {improvement_focus}
        """

        chapter_goals = [
            f"{i + 1}. Rewrite Original Chapter {i + 1} based on the overall improvement focus and writing guidelines."
            for i, _ in enumerate(original_chapters)
        ]

        return {
            "novel_parameters": params,
            "writing_guidelines": guidelines,
            "improvement_focus": improvement_focus,
            "original_novel_path": novel_path,
            "chapter_delimiter": delimiter,
            "original_chapters_text": original_chapters,
            "rewritten_chapters_text": [],
            "current_chapter_index": 0,
            "current_original_chapter_text": original_chapters[0],
            "current_chapter_goal": chapter_goals[
                0
            ],  # Langchain needs this key for some graph structures
            "revision_cycles": 0,
            "max_revisions": config["workflow_settings"]["max_revisions_per_chapter"],
            "previous_rewritten_chapter_summary": "This is the start of the novel rewrite.",
        }
    except Exception as e:
        return signal_error_node(
            state, node_name, "Failed during initial rewrite setup.", e
        )


def load_original_chapter_text_node(state: RewriteState) -> Dict[str, Any]:
    idx = state["current_chapter_index"]
    originals = state["original_chapters_text"]
    if idx < len(originals):
        original_text = originals[idx]
        logger.info(
            f"Loaded original text for Chapter {idx + 1} (Length: {len(original_text)}) for rewrite."
        )
        return {"current_original_chapter_text": original_text}
    else:
        logger.error(
            f"Attempted to load original chapter at index {idx}, but only {len(originals)} chapters exist."
        )
        return signal_error_node(
            state,
            "LoadOriginalChapter",
            f"Index {idx} out of bounds for original chapters.",
        )


def rewriter_agent_node(
    state: RewriteState, llm_map: Dict, max_retries: int, vs_config: Dict
) -> Dict[str, Any]:
    node_name = "Rewriter Agent"
    logger.info(f"--- Entering {node_name} ---")
    try:
        idx = state["current_chapter_index"]
        original_text = state["current_original_chapter_text"]
        guidelines = state["writing_guidelines"]  # Includes improvement focus
        params = state["novel_parameters"]
        prev_summary = state.get("previous_rewritten_chapter_summary", "N/A.")
        # chapter_goal = state.get('current_chapter_goal', f"Rewrite Chapter {idx+1}") # From planner

        if not original_text:
            return signal_error_node(
                state, node_name, f"Original text for Chapter {idx + 1} is missing."
            )

        author_feedback_on_previous_attempt = ""
        if state.get("author_decision") == "Rewrite" and state.get("author_feedback"):
            author_feedback_on_previous_attempt = f"\n**Feedback on Previous Rewrite Attempt (Address this specifically):**\n{state['author_feedback']}\n------\n"

        context_from_vs = ""
        if vector_store and vs_config.get("use_vector_store", False):
            query = f"Context for rewriting Chapter {idx + 1}, following themes of '{params.get('target_genre', '')}' and tone '{params.get('target_tone', '')}'. Previous summary: {prev_summary[:100]}"
            context_from_vs = get_context_from_rewritten_vs(
                query, max_chars=vs_config.get("retrieved_context_chars", 1500)
            )

        prompt_system = f"""You are a master novel rewriter. Your task is to rewrite the 'Original Chapter Text' for Chapter {idx + 1} of the novel titled '{params.get("novel_title", "Untitled Rewrite")}'.

        Your primary goal is to apply the 'User's Core Improvement Focus' and all 'Overall Writing Guidelines'.

        Context from previously rewritten parts:
        - Summary of previous REWRITTEN chapter: {prev_summary}
        - Relevant snippets from other REWRITTEN chapters (if any): {context_from_vs if context_from_vs else "N/A"}
        {author_feedback_on_previous_attempt}
        Original Chapter {idx + 1} Text to Rewrite:
        ```text
        {original_text}
        ```

        Overall Writing Guidelines (These include the User's Core Improvement Focus - Adhere to ALL points):
        {guidelines}

        Output *only* the complete rewritten chapter text for Chapter {idx + 1}. Do not add any commentary before or after the chapter text itself.
        """
        prompt = ChatPromptTemplate.from_messages([("system", prompt_system)])

        rewritten_draft = invoke_llm_with_retry(
            llm_map["drafter"], prompt, {}, max_retries=max_retries
        )
        logger.info(
            f"Rewritten draft for Chapter {idx + 1} generated (Length: {len(rewritten_draft)})."
        )
        return {
            "current_rewritten_chapter_draft": rewritten_draft,
            "author_feedback": None,
        }  # Clear author feedback
    except Exception as e:
        return signal_error_node(
            state,
            node_name,
            f"Error during rewrite of Chapter {state.get('current_chapter_index', -1) + 1}.",
            e,
        )


def editor_agent_node(
    state: RewriteState, llm_map: Dict, max_retries: int
) -> Dict[str, Any]:
    node_name = "Editor Agent (Rewrite)"
    logger.info(f"--- Entering {node_name} ---")
    try:
        idx = state["current_chapter_index"]
        original_text = state["current_original_chapter_text"]
        rewritten_draft = state["current_rewritten_chapter_draft"]
        guidelines = state["writing_guidelines"]  # Includes improvement focus

        if not original_text or not rewritten_draft:
            return signal_error_node(
                state, node_name, "Missing original or rewritten text for editing."
            )

        prompt_system = f"""You are a meticulous novel editor. Compare the 'Rewritten Chapter Draft' for Chapter {idx + 1} against the 'Original Chapter Text'.

        Key Assessment Criteria:
        1.  **Improvement Focus Achievement:** Does the rewrite successfully implement the 'User's Core Improvement Focus' (detailed in the 'Overall Writing Guidelines')? Are the changes effective?
        2.  **Preservation of Core Elements:** Does the rewrite maintain the core plot, characters, and essential scenes from the original chapter (unless specified by improvement focus)?
        3.  **Quality of Rewrite:** Is the rewritten prose engaging, clear, well-paced, and an improvement over the original?
        4.  **Guideline Adherence (General):** Does the rewrite adhere to general good writing practices in the 'Overall Writing Guidelines'?
        5.  **New Issues:** Has the rewrite introduced any new plot holes, inconsistencies, or stylistic problems?

        Provide **specific, actionable, numbered feedback points** targeting areas where the rewrite falls short or could be further improved in relation to the original and the improvement goals.
        If the rewritten chapter is a significant improvement, successfully meets the improvement focus, and has no major new issues, respond with the single phrase: "No major issues found."

        Original Chapter {idx + 1} Text:
        ```text
        {{original_chapter}}
        ```

        Overall Writing Guidelines (includes User's Core Improvement Focus):
        {guidelines}
        """
        prompt_human = "Rewritten Chapter {chapter_num} Draft:\n```\n{{rewritten_draft_text}}\n```\n\nProvide numbered editorial feedback or state 'No major issues found.'"
        prompt = ChatPromptTemplate.from_messages(
            [("system", prompt_system), ("human", prompt_human)]
        )

        feedback = invoke_llm_with_retry(
            llm_map["editor"],
            prompt,
            {
                "original_chapter": original_text,
                "rewritten_draft_text": rewritten_draft,
                "chapter_num": idx + 1,
            },
            max_retries=max_retries,
        )
        feedback_clean = feedback.strip()
        if feedback_clean.lower() == "no major issues found.":
            feedback_for_refiner = (
                "No major issues identified by the editor. Perform a light polish pass."
            )
        else:
            feedback_for_refiner = feedback_clean
        logger.info(f"Editor feedback for rewritten Chapter {idx + 1} generated.")
        return {"editor_feedback": feedback_for_refiner}
    except Exception as e:
        return signal_error_node(
            state,
            node_name,
            f"Error during editing of rewritten Chapter {state.get('current_chapter_index', -1) + 1}.",
            e,
        )


def refiner_agent_node(
    state: RewriteState, llm_map: Dict, max_retries: int
) -> Dict[str, Any]:
    node_name = "Refiner Agent (Rewrite)"
    logger.info(f"--- Entering {node_name} ---")
    try:
        idx = state["current_chapter_index"]
        draft_to_refine = state[
            "current_rewritten_chapter_draft"
        ]  # This is the output of rewriter_agent
        editor_feedback = state.get(
            "editor_feedback", "No specific feedback. Perform a general polish."
        )
        guidelines = state["writing_guidelines"]

        if not draft_to_refine:
            return signal_error_node(state, node_name, "Missing draft to refine.")

        prompt_system = f"""You are a skilled novel writer. Your task is to revise the 'Draft Chapter Text' for Chapter {idx + 1} based *only* on the 'Editor's Feedback'.

        Address each feedback point thoroughly. If feedback was "No major issues...", perform a light polish focusing on flow, word choice, and minor errors, while still adhering to the Overall Writing Guidelines.
        Ensure your revision maintains the core plot and characters from the draft unless the feedback explicitly requires alteration.

        Overall Writing Guidelines (includes User's Core Improvement Focus):
        {guidelines}

        Editor's Feedback to Address:
        {{editor_feedback_text}}

        Draft Chapter {idx + 1} Text to Refine:
        ```text
        {{chapter_draft}}
        ```
        Output *only* the complete, refined chapter text for Chapter {idx + 1}.
        """
        prompt = ChatPromptTemplate.from_messages([("system", prompt_system)])

        refined_text = invoke_llm_with_retry(
            llm_map["writer"],
            prompt,
            {"editor_feedback_text": editor_feedback, "chapter_draft": draft_to_refine},
            max_retries=max_retries,
        )
        logger.info(
            f"Refined rewritten Chapter {idx + 1} (Length: {len(refined_text)})."
        )
        return {
            "refined_chapter": refined_text,
            "editor_feedback": None,
        }  # Clear feedback
    except Exception as e:
        return signal_error_node(
            state,
            node_name,
            f"Error refining rewritten Chapter {state.get('current_chapter_index', -1) + 1}.",
            e,
        )


def summarizer_agent_node(
    state: RewriteState, llm_map: Dict, max_retries: int
) -> Dict[str, Any]:
    node_name = "Summarizer Agent (Rewrite)"
    logger.info(f"--- Entering {node_name} ---")
    try:
        # Summarizes the *last approved REWRITTEN chapter*
        current_idx = state[
            "current_chapter_index"
        ]  # Index of chapter ABOUT to be processed or just finished review
        rewritten_chapters = state.get("rewritten_chapters_text", [])

        # If we just approved chapter N (index N-1), its text is in rewritten_chapters[N-1].
        # We need this summary before processing chapter N+1 (index N).
        # So if current_idx points to the *next* chapter to be loaded, then the chapter just approved was current_idx - 1.

        idx_of_chapter_to_summarize = current_idx - 1

        if (
            idx_of_chapter_to_summarize < 0
        ):  # No previous rewritten chapter (e.g. before first chapter is approved)
            logger.info("No previous REWRITTEN chapter to summarize.")
            # The initial summary is set in setup_rewrite_planner_node
            return {
                "previous_rewritten_chapter_summary": state.get(
                    "previous_rewritten_chapter_summary",
                    "This is the start of the rewrite.",
                )
            }

        if idx_of_chapter_to_summarize >= len(rewritten_chapters):
            logger.error(
                f"Cannot summarize rewritten chapter at index {idx_of_chapter_to_summarize}, only {len(rewritten_chapters)} rewritten."
            )
            return {
                "previous_rewritten_chapter_summary": "Error: Could not find previous rewritten chapter to summarize."
            }

        last_rewritten_chapter_text = rewritten_chapters[idx_of_chapter_to_summarize]

        prompt_system = """You are an expert summarizer. Read the novel chapter text (which is a REWRITTEN version) and provide a concise summary (2-4 sentences) focusing ONLY on information crucial for starting the NEXT rewritten chapter:
        - Key events that occurred in this rewritten chapter.
        - Significant changes in character state (emotional, physical, relational).
        - Major plot points resolved or introduced in this rewritten chapter.
        - Unresolved questions or cliffhangers.
        Keep it brief and strictly functional for context continuity for the rewriter."""
        prompt_human = "Chapter {chapter_num_human} Text (Rewritten):\n```\n{chapter_text}\n```\n\nProvide the concise summary:"
        prompt = ChatPromptTemplate.from_messages(
            [("system", prompt_system), ("human", prompt_human)]
        )

        summary = invoke_llm_with_retry(
            llm_map["summarizer"],
            prompt,
            {
                "chapter_num_human": idx_of_chapter_to_summarize + 1,
                "chapter_text": last_rewritten_chapter_text,
            },
            max_retries=max_retries,
        )
        logger.info(
            f"Summary for REWRITTEN Chapter {idx_of_chapter_to_summarize + 1} generated."
        )
        return {"previous_rewritten_chapter_summary": summary.strip()}
    except Exception as e:
        return signal_error_node(
            state, node_name, "Error during summarization of rewritten chapter.", e
        )


def author_review_node(
    state: RewriteState, llm_map: Dict, max_retries: int
) -> Dict[str, Any]:
    node_name = "Author Review (Rewrite)"
    logger.info(f"--- Entering {node_name} ---")
    try:
        idx = state["current_chapter_index"]
        original_text = state["current_original_chapter_text"]
        rewritten_refined_text = state["refined_chapter"]  # Output of refiner
        guidelines = state["writing_guidelines"]
        improvement_focus = state["improvement_focus"]
        max_rev = state["max_revisions"]
        rev_cycles = state.get("revision_cycles", 0)

        if not original_text or not rewritten_refined_text:
            return signal_error_node(
                state,
                node_name,
                "Missing original or refined rewritten text for author review.",
            )

        prompt_system = f"""You are the Author. Review the 'Refined Rewritten Chapter' against the 'Original Chapter Text' and the 'User's Core Improvement Focus'.

        Assessment Criteria:
        1.  **Improvement Focus:** Is the User's Core Improvement Focus clearly and effectively addressed in the rewrite?
        2.  **Quality vs. Original:** Is the rewritten version a clear improvement in prose quality, engagement, clarity, and pacing compared to the original?
        3.  **Core Preservation:** Are essential plot points and character integrity from the original maintained (unless the improvement focus dictated changes)?
        4.  **Overall Satisfaction:** As the author, are you satisfied with this rewritten chapter as a replacement for the original?

        Respond with **ONLY** "Approved" if satisfied.
        Otherwise, provide **concise, actionable feedback** starting with "Feedback:" for another rewrite iteration, explaining what still needs to be improved in relation to the original and the improvement goals.

        User's Core Improvement Focus (from Overall Writing Guidelines):
        {improvement_focus}

        Original Chapter {idx + 1} Text:
        ```text
        {{original_chapter_text_for_prompt}}
        ```

        Overall Writing Guidelines (for context):
        {guidelines}
        """  # Removed consistency report from prompt for simplicity, can be added back
        prompt_human = "Refined Rewritten Chapter {chapter_num_for_prompt}:\n```\n{{refined_rewritten_text_for_prompt}}\n```\n\nReview and respond with 'Approved' or 'Feedback: ...'"
        prompt = ChatPromptTemplate.from_messages(
            [("system", prompt_system), ("human", prompt_human)]
        )

        review_decision_text = invoke_llm_with_retry(
            llm_map["author"],
            prompt,
            {
                "original_chapter_text_for_prompt": original_text,
                "refined_rewritten_text_for_prompt": rewritten_refined_text,
                "chapter_num_for_prompt": idx + 1,
            },
            max_retries=max_retries,
        )

        review_clean = review_decision_text.strip().lower()
        current_decision = "Rewrite"  # Default
        author_fb = review_decision_text.strip()

        if review_clean == "approved":
            current_decision = "Approved"
            author_fb = f"Chapter {idx + 1} rewrite approved."
            logger.info(author_fb)

            newly_completed_chapters = list(state.get("rewritten_chapters_text", []))
            newly_completed_chapters.append(
                rewritten_refined_text
            )  # Add approved chapter

            # Add to Vector Store
            if vector_store:
                add_rewritten_chapter_to_vs(
                    idx,
                    rewritten_refined_text,
                    state.get("previous_rewritten_chapter_summary", "N/A"),
                )

            next_idx = idx + 1
            if next_idx < len(state["original_chapters_text"]):
                next_goal = (
                    f"{next_idx + 1}. Rewrite Original Chapter {next_idx + 1}..."
                )
                return {
                    "author_decision": current_decision,
                    "author_feedback": author_fb,
                    "rewritten_chapters_text": newly_completed_chapters,
                    "current_chapter_index": next_idx,
                    "current_chapter_goal": next_goal,
                    "revision_cycles": 0,  # Reset for next chapter
                    "current_rewritten_chapter_draft": None,
                    "editor_feedback": None,
                    "refined_chapter": None,
                    "consistency_report": None,
                }
            else:  # All chapters rewritten
                logger.info("All chapters rewritten and approved!")
                return {
                    "author_decision": current_decision,
                    "author_feedback": author_fb,
                    "rewritten_chapters_text": newly_completed_chapters,
                    "current_chapter_index": next_idx,  # Will be out of bounds, signaling end
                    "current_chapter_goal": None,
                    "current_original_chapter_text": None,  # Clear
                }
        else:  # Decision is "Rewrite" (either explicitly or default)
            if not review_clean.startswith("feedback:"):
                author_fb = f"Feedback: (Implicit from non-approval) {review_decision_text.strip()}"

            logger.info(
                f"Author requested rewrite for Chapter {idx + 1}. Feedback: {author_fb}"
            )
            current_revision_cycles = rev_cycles + 1
            if current_revision_cycles >= max_rev:
                logger.warning(
                    f"Max revisions ({max_rev}) reached for Ch {idx + 1}. Forcing approval."
                )
                # Effectively same as "Approved" path above
                newly_completed_chapters = list(
                    state.get("rewritten_chapters_text", [])
                )
                newly_completed_chapters.append(
                    rewritten_refined_text
                )  # Add force-approved chapter
                if vector_store:
                    add_rewritten_chapter_to_vs(
                        idx,
                        rewritten_refined_text,
                        state.get("previous_rewritten_chapter_summary", "N/A"),
                    )
                next_idx = idx + 1
                next_goal = (
                    f"{next_idx + 1}. Rewrite Original Chapter {next_idx + 1}..."
                    if next_idx < len(state["original_chapters_text"])
                    else None
                )
                return {
                    "author_decision": "Approved",  # Forced
                    "author_feedback": author_fb
                    + "\n(Forced approval due to max revisions)",
                    "rewritten_chapters_text": newly_completed_chapters,
                    "current_chapter_index": next_idx,
                    "current_chapter_goal": next_goal,
                    "revision_cycles": 0,
                    "current_rewritten_chapter_draft": None,
                    "editor_feedback": None,
                    "refined_chapter": None,
                }
            else:  # Normal rewrite cycle
                return {
                    "author_decision": "Rewrite",
                    "author_feedback": author_fb,
                    "revision_cycles": current_revision_cycles,
                    "current_rewritten_chapter_draft": None,
                    "editor_feedback": None,
                    "refined_chapter": None,
                }
    except Exception as e:
        return signal_error_node(
            state,
            node_name,
            f"Error during author review of rewritten Chapter {state.get('current_chapter_index', -1) + 1}.",
            e,
        )


# --- Graph Definition ---
def route_after_author_review(state: RewriteState):
    decision = state.get("author_decision")
    if decision == "Error":
        return END
    if decision == "Approved":
        if state["current_chapter_index"] >= len(state["original_chapters_text"]):
            logger.info("All chapters rewritten and approved. Ending workflow.")
            return END  # Finished all chapters
        else:
            logger.info(
                "Chapter approved. Proceeding to summarize rewritten chapter then load next original."
            )
            return "summarizer_agent"  # Then to load_original_chapter_text_node
    elif decision == "Rewrite":
        logger.info(
            f"Author requested rewrite for Chapter {state['current_chapter_index'] + 1}. Returning to Rewriter Agent."
        )
        return "rewriter_agent"  # Loop back for another rewrite attempt on the same chapter
    logger.error(f"Unexpected author decision: {decision}. Ending.")
    return END


def build_rewrite_graph(
    config: Dict, llm_map: Dict, novel_data_loaded: Dict
) -> StateGraph:
    workflow = StateGraph(RewriteState)
    from functools import partial

    max_retries_llm = config["workflow_settings"]["max_llm_retries"]
    vs_conf = config.get("vector_store_settings", {})

    # Bind configs and LLMs
    bound_setup_planner = partial(
        setup_rewrite_planner_node, config=config, novel_data=novel_data_loaded
    )
    bound_rewriter = partial(
        rewriter_agent_node,
        llm_map=llm_map,
        max_retries=max_retries_llm,
        vs_config=vs_conf,
    )
    bound_editor = partial(
        editor_agent_node, llm_map=llm_map, max_retries=max_retries_llm
    )
    bound_refiner = partial(
        refiner_agent_node, llm_map=llm_map, max_retries=max_retries_llm
    )
    bound_summarizer = partial(
        summarizer_agent_node, llm_map=llm_map, max_retries=max_retries_llm
    )
    bound_author_review = partial(
        author_review_node, llm_map=llm_map, max_retries=max_retries_llm
    )

    workflow.add_node("setup_rewrite_planner", bound_setup_planner)
    workflow.add_node(
        "load_original_chapter", load_original_chapter_text_node
    )  # Simple node, no partial needed if no extra args
    workflow.add_node("rewriter_agent", bound_rewriter)
    workflow.add_node("editor_agent", bound_editor)
    workflow.add_node("refiner_agent", bound_refiner)
    workflow.add_node("summarizer_agent", bound_summarizer)
    workflow.add_node("author_review", bound_author_review)

    workflow.set_entry_point("setup_rewrite_planner")
    workflow.add_edge(
        "setup_rewrite_planner", "load_original_chapter"
    )  # After planning, load first original chapter
    workflow.add_edge("load_original_chapter", "rewriter_agent")
    workflow.add_edge("rewriter_agent", "editor_agent")
    workflow.add_edge("editor_agent", "refiner_agent")
    workflow.add_edge(
        "refiner_agent", "author_review"
    )  # Author reviews the output of Refiner

    workflow.add_conditional_edges(
        "author_review",
        route_after_author_review,
        {
            "summarizer_agent": "summarizer_agent",  # Approved, more chapters to go
            "rewriter_agent": "rewriter_agent",  # Rewrite current chapter
            END: END,  # All done or error
        },
    )
    # After summarizer (which runs if a chapter was approved and more exist), load the next original chapter
    workflow.add_edge("summarizer_agent", "load_original_chapter")

    logger.info("Rewrite graph definition complete.")
    return workflow


# --- Saving Final Rewritten Novel ---
def save_rewritten_novel(state: RewriteState, config: Dict):
    logger.info("--- Saving Final Rewritten Novel ---")
    output_dir = config["workflow_settings"]["output_directory"]
    # output_format = config['workflow_settings'].get('output_format', 'txt').lower() # Get from config
    params = state.get("novel_parameters", {})
    original_path = state.get("original_novel_path", "unknown_original.txt")
    rewritten_chapters = state.get("rewritten_chapters_text", [])

    if not rewritten_chapters:
        logger.warning("No rewritten chapters to save.")
        return

    safe_title_base = (
        re.sub(r"[^\w\-_\.]", "_", os.path.basename(original_path))
        .replace(" ", "_")
        .lower()
    )
    # Strip extension if any
    safe_title_base = os.path.splitext(safe_title_base)[0]

    novel_filename = os.path.join(
        output_dir, f"{safe_title_base}_rewritten.md"
    )  # Default to md for structure

    try:
        with open(novel_filename, "w", encoding="utf-8") as f:
            f.write(f"# Title: {params.get('novel_title', 'Rewritten Novel')}\n\n")
            if params.get("target_genre") != "Same as original":
                f.write(
                    f"**Target Genre (Rewrite):** {params.get('target_genre', 'N/A')}\n"
                )
            if params.get("target_tone") != "Same as original or enhanced":
                f.write(
                    f"**Target Tone (Rewrite):** {params.get('target_tone', 'N/A')}\n\n"
                )
            f.write(f"**Original Novel Source:** {original_path}\n\n")
            f.write(
                f"**Core Improvement Focus for this Rewrite:**\n{state.get('improvement_focus', 'N/A')}\n\n"
            )
            f.write("---\n\n")

            for i, chapter_content in enumerate(rewritten_chapters):
                f.write(f"## Rewritten Chapter {i + 1}\n\n")
                f.write(chapter_content.strip())
                f.write("\n\n---\n\n")
        logger.info(f"Rewritten novel saved successfully to: {novel_filename}")
    except Exception as e:
        logger.error(f"Error saving rewritten novel: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph Novel Rewriter")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file for rewriting.",
    )
    parser.add_argument("--thread-id", type=str, help="Thread ID for resuming.")
    parser.add_argument("--resume", action="store_true", help="Attempt to resume.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config["workflow_settings"]["output_directory"]
    setup_file_logging(output_dir)

    novel_data_loaded = load_novel_data_json()
    if not novel_data_loaded:
        logger.warning(
            f"'{DEFAULT_NOVEL_DATA_PATH}' was empty or not found. Some dynamic parameter choices might be limited."
        )

    logger.info("Initializing LLMs for rewriting...")
    llm_configs = config["llm_settings"]
    temps = llm_configs.get("temperature", {})
    try:
        llm_map = {
            "author": ChatOllama(
                model=llm_configs["author_model"], temperature=temps.get("author", 0.6)
            ),
            "drafter": ChatOllama(
                model=llm_configs["drafter_model"],
                temperature=temps.get("drafter", 0.8),
            ),
            "editor": ChatOllama(
                model=llm_configs["editor_model"], temperature=temps.get("editor", 0.4)
            ),
            "writer": ChatOllama(
                model=llm_configs["writer_model"], temperature=temps.get("writer", 0.7)
            ),
            "summarizer": ChatOllama(
                model=llm_configs["summarizer_model"],
                temperature=temps.get("summarizer", 0.3),
            ),
        }
        logger.info("Ollama models for rewriting initialized.")
    except KeyError as ke:
        logger.error(
            f"Missing model definition in {args.config} under llm_settings: {ke}. Exiting."
        )
        exit(1)
    except Exception as e:
        logger.error(f"Error initializing Ollama models: {e}", exc_info=True)
        exit(1)

    initialize_vector_store_for_rewrite(config)

    checkpoint_db_path = os.path.join(
        output_dir,
        config["workflow_settings"].get("checkpoint_db", "rewrite_checkpoint.sqlite"),
    )

    # This is the start of the main 'with' block for the checkpointer
    with SqliteSaver.from_conn_string(checkpoint_db_path) as memory_checkpointer:
        logger.info(f"SqliteSaver checkpointer engaged (DB: {checkpoint_db_path})")
        app = build_rewrite_graph(config, llm_map, novel_data_loaded).compile(
            checkpointer=memory_checkpointer
        )
        logger.info("Rewrite graph compiled.")

        # --- This section was likely the source of indentation errors ---
        # Ensure it's correctly indented under the 'with' block
        thread_id = (
            args.thread_id if args.thread_id else f"rewrite-session-{int(time.time())}"
        )
        run_config = {"configurable": {"thread_id": thread_id}}
        logger.info(f"Using Thread ID: {thread_id}")

        initial_state_input = {}
        final_state_values = {}  # To store the latest full state from the graph

        if args.resume:
            logger.info(f"Attempting to resume workflow for thread_id: {thread_id}")
            existing_state_result = app.get_state(run_config)
            if not existing_state_result or not existing_state_result.values:
                logger.error(
                    f"No checkpoint found for thread_id '{thread_id}' or state is empty. Starting new run."
                )
                args.resume = False
            else:
                final_state_values = existing_state_result.values
                logger.info(
                    f"Resuming. Chapters rewritten so far: {len(final_state_values.get('rewritten_chapters_text', []))}"
                )

        if (
            not args.resume
        ):  # This will also run if resume failed and args.resume was set to False
            logger.info(f"Starting new rewrite workflow for thread_id: {thread_id}")
            # For a new run, initial_state_input is empty.
            # The 'setup_rewrite_planner' node is responsible for populating the initial state.
            pass  # initial_state_input remains {}

        estimated_num_chapters = config["rewrite_settings"].get(
            "estimated_number_of_chapters", 15
        )
        max_revisions_per_chapter = config["workflow_settings"].get(
            "max_revisions_per_chapter", 2
        )
        nodes_per_revision_loop = 5

        recursion_limit_est = (
            (
                estimated_num_chapters
                * max_revisions_per_chapter
                * nodes_per_revision_loop
            )
            + (estimated_num_chapters * 2)
            + 20
        )

        run_config["recursion_limit"] = config["workflow_settings"].get(
            "recursion_limit", recursion_limit_est
        )
        logger.info(
            f"Calculated recursion limit: {run_config['recursion_limit']} (based on ~{estimated_num_chapters} chapters)"
        )

        try:
            stream_payload = (
                initial_state_input if not args.resume else None
            )  # None when resuming from checkpoint

            for step, s_update in enumerate(
                app.stream(stream_payload, run_config, stream_mode="values")
            ):
                # In "values" mode, s_update is the entire state after each node.
                # The last key is the node that just ran.
                node_name = list(s_update.keys())[-1]
                final_state_values = s_update[
                    node_name
                ]  # The state *after* this node ran
                logger.info(f"--- Step {step + 1}: Node '{node_name}' completed ---")

                if final_state_values.get("author_decision") == "Error":
                    logger.error(
                        f"Error state detected after node '{node_name}'. Stopping. Error: {final_state_values.get('last_error', 'N/A')}"
                    )
                    break

            # After the loop, if no error broke it, get the very final state.
            if not (
                final_state_values
                and final_state_values.get("author_decision") == "Error"
            ):
                current_state_after_loop = app.get_state(run_config)
                if current_state_after_loop and current_state_after_loop.values:
                    final_state_values = current_state_after_loop.values
                else:
                    logger.warning(
                        "app.get_state returned None or empty values after stream loop completed without error signal."
                    )
                    # final_state_values would still hold the state from the last iteration in this case.

        except Exception as e:  # Catch errors during the app.stream() call itself
            logger.error(
                f"Workflow execution error during app.stream(): {e}", exc_info=True
            )
            try:
                # Try to get the state as it was when the error occurred
                current_st_on_error = app.get_state(run_config)
                if current_st_on_error and current_st_on_error.values:
                    final_state_values = current_st_on_error.values
            except Exception as se:
                logger.error(
                    f"Could not retrieve state after graph execution error: {se}"
                )

            # Ensure final_state_values is a dict and error is recorded
            if not isinstance(final_state_values, dict):
                final_state_values = {}
            if "last_error" not in final_state_values:
                final_state_values["last_error"] = str(e)
            if "author_decision" not in final_state_values:
                final_state_values["author_decision"] = "Error"

        finally:  # This 'finally' is for the 'try' block that wraps app.stream()
            logger.info("\n--- REWRITE WORKFLOW STREAMING COMPLETED OR ERRORED ---")
            if (
                final_state_values
            ):  # Check if final_state_values has been populated at all
                save_rewritten_novel(final_state_values, config)
                num_rewritten = len(
                    final_state_values.get("rewritten_chapters_text", [])
                )
                num_original = len(
                    final_state_values.get("original_chapters_text", [])
                )  # original_chapters_text should be in state

                if final_state_values.get("author_decision") == "Error":
                    logger.error(
                        f"FINAL STATUS: Rewrite stopped due to error: {final_state_values.get('last_error', 'Unknown error')}"
                    )
                elif num_original > 0 and num_rewritten >= num_original:
                    logger.info(
                        f"FINAL STATUS: Successfully rewritten {num_rewritten} of {num_original} original chapters."
                    )
                elif (
                    num_original == 0 and num_rewritten > 0
                ):  # Should not happen if setup is correct
                    logger.warning(
                        f"FINAL STATUS: Rewrote {num_rewritten} chapters, but original chapter count was zero in state."
                    )
                else:  # num_original > 0 and num_rewritten < num_original
                    logger.warning(
                        f"FINAL STATUS: Rewrite process ended. {num_rewritten}/{num_original} chapters rewritten."
                    )
            else:  # final_state_values is empty or None
                logger.error(
                    "Workflow finished without producing a retrievable final state. Check for early critical errors."
                )

            logger.info(
                "SqliteSaver context exiting. DB connection will be closed by 'with' statement."
            )

    # This line is now correctly outside the 'with' block
    logger.info("-------------------------\nScript End. (Programmed termination)")
