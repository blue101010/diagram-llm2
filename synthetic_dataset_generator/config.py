import os
import logging
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google Generative AI with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dataset_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------ CONFIGURATION & CONSTANTS ------------------

# Model identifiers
GEMINI_2_5_PRO = "gemini-2.5-pro-preview-03-25"

# API Rate Limiting
MAX_WORKERS = 3  
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # Seconds

# File to save all generated questions and diagrams
OUTPUT_FILE = "generated_questions.json"
MD_DIRECTORY = "md"

# Multi-line string constants for prompts
QUESTION_PROMPT = """\
You are an expert question generator for Mermaid diagrams.
Generate 100 diverse questions that require a Mermaid diagram as an answer.
The Mermaid diagram should be of type "{diagram_type}".
All generated questions must have multiple interacting components. They should not be about a single component.
They should be relatively complex and also make sure there are many complex questions with more than 5 components and many arrows and interactions.

The questions should focus on teaching proper Mermaid syntax rather than complex concepts.
Each question should specify EXACTLY what elements should be in the diagram.

For example:
- "Draw a pie chart with sections A (20%), B (70%), and C (10%)"
- "Draw a sequence diagram where User logs in to System, System validates credentials, and then System grants access"
- "Draw a flow diagram where A sends request to B, and B responds to the request with a delay of 2ms"

Include questions of varying complexity, covering ALL syntax features described in the documentation.
Your questions should systematically cover every element, feature, and syntax option in the documentation.

Here is the complete documentation for this diagram type that you should use to generate questions:
```
{doc_content}
```

Return your answer strictly as a JSON object with ONE key "questions" containing an array of 100 strings.
Do not include any additional text.

MAKE SURE THE QUESTION SPECIFIES EXACTLY WHAT SHOULD BE IN THE DIAGRAM. THERE SHOULD BE NO AMBIGUITY.
I don't want to see any questions with less than 3 components.
"""

MERMAID_PROMPT = """\
You are a Mermaid diagram generator. Below is the documentation for Mermaid diagrams of type "{diagram_type}":

## Documentation
```markdown
{doc_content}
```

## Question:
{question}

## Instructions
Based on the documentation above, generate a precise and correct Mermaid diagram that answers the question.
Return your answer strictly as a JSON object with ONE key "mermaid_diagram" whose value is a valid Mermaid code snippet.
Do not include any additional explanation or text - ONLY the Mermaid code.

The Mermaid code MUST be syntactically correct and render without errors according to the provided Mermaid documentation.
- Pay close attention to syntax rules, keywords, and allowed characters.
- Ensure the diagram properly represents what was asked in the question.
- THE OUTPUT DIAGRAM MUST LOOK PRETTY and properly spaced. It shouldn't be cluttered.
- Make sure when you're drawing arrows or writing large texts they aren't overlapping.
- Only give the mermaid code, no text not even markdown code wrapper. Just the mermaid code directly!

Eg - Create an architecture diagram with a group `cloud_env` (cloud icon, title 'Cloud Environment'). Inside this group, place two services: `app` (server icon, 'Application') and `storage` (disk icon, 'File Storage'). Connect `app`'s right port (`R`) to `storage`'s left port (`L`).
In the output the mermaid_diagram key should have the following value:
architecture-beta
    group cloud_env(cloud)[Cloud Environment]

    service app(server)[Application] in cloud_env
    service storage(disk)[File Storage] in cloud_env

    app:R -- L:storage
"""

# ------------------ STRUCTURED OUTPUT MODELS ------------------

class QuestionsOutput(BaseModel):
    questions: List[str]


class MermaidDiagramOutput(BaseModel):
    mermaid_diagram: str
