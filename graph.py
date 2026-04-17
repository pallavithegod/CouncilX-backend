import os
import json
import asyncio
import re
from datetime import datetime, timezone
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_service import search_context

load_dotenv()

azure_client = AsyncOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT", "")
)

class ModelResponse(BaseModel):
    model_id: str
    model_name: str
    response: str
    status: str
    bias_score: float = 0.5
    neutrality_score: float = 0.5
    clarity_score: float = 0.5

class GraphState(TypedDict):
    uid: str
    session_id: str
    question: str
    history: List[Dict[str, str]]
    models_l1: List[str]
    model_l2: str
    l1_responses: List[ModelResponse]
    l2_response: Optional[ModelResponse]
    deliberation_history: List[Dict[str, Any]]
    iterations: int
    feedback: str
    needs_reconsideration: bool
    confidence: float
    context: str
    web_context: str
    status: str

async def route_query(model_id: str, persona: str, user_content: str, history: List[Dict[str, str]] = [], extract_scores: bool = False) -> ModelResponse:
    try:
        deployment_name = os.getenv(f"AZURE_DEPLOYMENT_{model_id.upper()}", model_id)
        
        prompt_with_scores = persona
        if extract_scores:
            prompt_with_scores += "\n\nIMPORTANT: At the end of your response, provide your self-assessment format exactly as: [SCORES: bias=0.82, neutrality=0.76, clarity=0.71]"

        messages = [{"role": "system", "content": prompt_with_scores}]
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_content})

        print(f"[Azure Request] model={model_id}, deployment={deployment_name}")
        response = await azure_client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_completion_tokens=4000,
            temperature=0.7,
        )
        response_text = response.choices[0].message.content or ""
        print(f"[Azure Response] model={model_id}, length={len(response_text)}")

        
        bias, neutral, clarity = 0.5, 0.5, 0.5
        if extract_scores:
            import re
            # More flexible regex for scores
            match = re.search(r"\[SCORES:.*?bias=([\d.]+).*?neutrality=([\d.]+).*?clarity=([\d.]+).*?\]", response_text, re.IGNORECASE)
            if match:
                bias = float(match.group(1))
                neutral = float(match.group(2))
                clarity = float(match.group(3))
                response_text = re.sub(r"\[SCORES:.*?\]", "", response_text, flags=re.IGNORECASE).strip()

                
        return ModelResponse(
            model_id=model_id, 
            model_name=model_id.capitalize(), 
            response=response_text, 
            status="success",
            bias_score=bias,
            neutrality_score=neutral,
            clarity_score=clarity
        )
    except Exception as e:
        print(f"[Azure API Error: {model_id}] {str(e)}")
        return ModelResponse(model_id=model_id, model_name=model_id.capitalize(), response=f"[Azure API Error: {model_id}] {str(e)}", status="error")

async def retrieval_node(state: GraphState):
    print("--- Executing Retrieval ---")
    session_id = state.get("session_id")
    if not session_id: return {"context": ""}
    
    context = await search_context(state["question"], session_id)
    return {"context": context, "status": "Forensic file data retrieved."}

async def web_research_node(state: GraphState):
    print(f"--- Executing Web Research for query: {state['question']} ---")
    from duckduckgo_search import DDGS
    
    query = state["question"]
    web_results = ""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=6))
            for r in results:
                web_results += f"Source: {r.get('href')}\nTitle: {r.get('title')}\nSnippet: {r.get('body')}\n\n"
    except Exception as e:
        print(f"Web Search Error (DDG): {e}")
        web_results = "Failed to retrieve web search data via DDG."
    
    if not web_results.strip() or "Failed" in web_results:
        print("Search returned no results or failed.")
    else:
        print(f"Web Search success, found {len(web_results)} chars of context.")
        
    return {"web_context": web_results, "status": "Live web research completed."}

async def layer1_node(state: GraphState):
    print("--- Executing Layer 1 ---")
    import asyncio
    tasks = []
    feedback_context = f"\n\nPre-existing feedback to consider: {state.get('feedback', '')}" if state.get('feedback') else ""
    context_str = f"\n\nSpecialized Forensic Data Context:\n{state.get('context', 'None Available')}"
    web_str = f"\n\nLive Web Research Data:\n{state.get('web_context', 'None Available')}"
    
    models = state.get('models_l1') or []
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    for m in models:
        sys_prompt = f"""You are {m.capitalize()}, a Tier 1 Deliberator. Analyzing objective facts.
CURRENT DATE: {current_time} UTC

MISSION: 
Examine the provided context to answer the user inquiry. 

CRITICAL INSTRUCTION FOR FORENSIC DATA:
If 'FORENSIC DATA' (retrieved from uploaded files like PDF, TXT, or DOCX) is present, your PRIMARY TASK is to audit it for BIASNESS. Specifically, look for:
- Partisan framing or ideological leanings.
- Cherry-picked facts or omission of counter-arguments.
- Emotional or leading language.
- Logical fallacies.

WEB RESEARCH (LATEST INFO):
Use this to provide external context and verify or challenge the forensic data.

CONTEXT:
FORENSIC DATA:
{context_str}

WEB RESEARCH:
{web_str}
"""
        tasks.append(route_query(m, sys_prompt, state['question'] + feedback_context, state.get('history', []), extract_scores=True))
    print(f"--- Layer 1: starting tasks for {len(tasks)} models ---")
    l1_resp = await asyncio.gather(*tasks)
    print(f"--- Layer 1: finished, reached {len(l1_resp)} responses ---")
    current_history = state.get("deliberation_history", [])
    new_entry = {
        "iteration": state.get("iterations", 0) + 1,
        "responses": [r.dict() if hasattr(r, 'dict') else r for r in l1_resp],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    return {
        "l1_responses": l1_resp, 
        "deliberation_history": current_history + [new_entry],
        "iterations": state.get("iterations", 0) + 1, 
        "status": "Tier 1 deliberations finished."
    }


async def layer2_node(state: GraphState):
    print(f"--- Executing Layer 2 (Iteration {state.get('iterations', 1)}) ---")
    aggregation_context = f"**User Question:** {state.get('question', '')}\n\n"
    if state.get("context"):
        aggregation_context += f"**Forensic Context:**\n{state['context']}\n\n"
    if state.get("web_context"):
        aggregation_context += f"**Web Research:**\n{state['web_context']}\n\n"
        
    aggregation_context += "**Layer 1 Deliberations:**\n"
    l1_resps = state.get('l1_responses') or []
    for r in l1_resps:
        model_name = r.model_name if hasattr(r, 'model_name') else r.get('model_name', 'Model')
        response_text = r.response if hasattr(r, 'response') else r.get('response', '')
        aggregation_context += f"--- {model_name} ---\n{response_text}\n\n"
    
    if state.get("feedback"):
        aggregation_context += f"**Previous Critique/Feedback:** {state['feedback']}\n\n"

    aggregation_context += "Provide a final synthesis grounded in the forensic context and Layer 1 perspectives."
    
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    sys_prompt = f"""You are the Final Arbiter of the AI Parliament. 
    Current Date: {current_date} UTC.
    
    Your task is to synthesize the deliberations into a definitive, and comprehensive verdict.
    
    CRITICAL RULES:
    1. NEVER mention "Layer 1", "Models", "Deliberators", "Arbiter" or any internal system jargon.
    2. Provide a unified, professional response as if you are a single objective authority.
    3. Use a clear, structured layout with bold headings and bullet points (Grok-style).
    4. Ensure you explicitly reference the current status (the year {current_date[:4]}) to provide the most up-to-date answer.
    5. Always output a confidence score and bias assessment at the VERY END in this EXACT format:
       [SCORES: confidence=X.XX, bias=X.XX, feedback=Brief instructions for revision if needed else "None"]
    
    Provide a detailed and definitive answer grounded in the forensic and web research data."""
    
    try:
        deployment_name = os.getenv(f"AZURE_DEPLOYMENT_{state['model_l2'].upper()}", state['model_l2'])
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": aggregation_context}
        ]
        
        print(f"[Layer 2 Request] model={state['model_l2']}, deployment={deployment_name}")
        response = await azure_client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_completion_tokens=4000,
            temperature=0.4,
        )
        response_text = response.choices[0].message.content or ""
        print(f"[Layer 2 Response] length={len(response_text)}")

        
        bias, neutral, confidence, feedback = 0.5, 0.5, 0.5, ""
        import re
        # Flexible regex for l2 scores
        match = re.search(r"\[SCORES:.*?bias=([\d.]+).*?neutrality=([\d.]+).*?confidence=([\d.]+).*?feedback=(.*?)\]", response_text, re.IGNORECASE | re.DOTALL)
        if match:
            bias = float(match.group(1))
            neutral = float(match.group(2))
            confidence = float(match.group(3))
            feedback = match.group(4).strip()
            response_text = re.sub(r"\[SCORES:.*?\]", "", response_text, flags=re.IGNORECASE | re.DOTALL).strip()

        
        needs_reconsideration = False
        if state.get("iterations", 1) < 2:
            if bias > 0.6 or confidence < 0.7:
                needs_reconsideration = True
                if not feedback or feedback.lower() == "none":
                    feedback = "Re-evaluate based on suspected bias or low confidence."

        l2_resp = ModelResponse(
            model_id=state['model_l2'],
            model_name=state['model_l2'].capitalize(),
            response=response_text,
            status="success",
            bias_score=bias,
            neutrality_score=neutral,
            clarity_score=confidence
        )
        
        return {
            "l2_response": l2_resp,
            "needs_reconsideration": needs_reconsideration,
            "feedback": feedback,
            "confidence": confidence * 100,
            "status": "Consensus reached." if not needs_reconsideration else "Re-deliberation triggered."
        }
    except Exception as e:
        error_msg = f"[Layer 2 Error] {str(e)}"
        print(error_msg)
        return {
            "l2_response": ModelResponse(
                model_id=state.get('model_l2', 'unknown'),
                model_name="Error",
                response=error_msg,
                status="error"
            ),
            "needs_reconsideration": False,
            "feedback": str(e),
            "confidence": 0
        }

def should_continue(state: GraphState):
    if state.get("needs_reconsideration"):
        return "layer1_node"
    return END

workflow = StateGraph(GraphState)
workflow.add_node("retrieval_node", retrieval_node)
workflow.add_node("web_research_node", web_research_node)
workflow.add_node("layer1_node", layer1_node)
workflow.add_node("layer2_node", layer2_node)

workflow.set_entry_point("retrieval_node")
workflow.add_edge("retrieval_node", "web_research_node")
workflow.add_edge("web_research_node", "layer1_node")
workflow.add_edge("layer1_node", "layer2_node")
workflow.add_conditional_edges("layer2_node", should_continue, {"layer1_node": "layer1_node", END: END})

app_graph = workflow.compile()

def get_app():
    return app_graph
