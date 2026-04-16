import os
import json
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
    iterations: int
    feedback: str
    needs_reconsideration: bool
    confidence: float
    context: str

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

        response = await azure_client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_completion_tokens=4000,
            temperature=0.7,
        )
        response_text = response.choices[0].message.content or ""
        
        bias, neutral, clarity = 0.5, 0.5, 0.5
        if extract_scores:
            import re
            match = re.search(r"\[SCORES:\s*bias=([\d.]+),\s*neutrality=([\d.]+),\s*clarity=([\d.]+)\]", response_text)
            if match:
                bias = float(match.group(1))
                neutral = float(match.group(2))
                clarity = float(match.group(3))
                response_text = response_text.replace(match.group(0), "").strip()
                
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
    return {"context": context}

async def layer1_node(state: GraphState):
    print("--- Executing Layer 1 ---")
    import asyncio
    tasks = []
    feedback_context = f"\n\nPre-existing feedback to consider: {state.get('feedback', '')}" if state.get('feedback') else ""
    context_str = f"\n\nSpecialized Forensic Data Context:\n{state.get('context', 'None Available')}"
    
    models = state.get('models_l1') or []
    for m in models:
        sys_prompt = f"You are {m.capitalize()}, a Tier 1 Deliberator. Analyzing objective facts. Use the following Forensic Data Context to ground your response if relevant.\n{context_str}"
        tasks.append(route_query(m, sys_prompt, state['question'] + feedback_context, state.get('history', []), extract_scores=True))
    l1_resp = await asyncio.gather(*tasks)
    return {"l1_responses": l1_resp, "iterations": state.get("iterations", 0) + 1}

async def layer2_node(state: GraphState):
    print(f"--- Executing Layer 2 (Iteration {state.get('iterations', 1)}) ---")
    aggregation_context = f"**User Question:** {state.get('question', '')}\n\n"
    if state.get("context"):
        aggregation_context += f"**Forensic Context:**\n{state['context']}\n\n"
        
    aggregation_context += "**Layer 1 Deliberations:**\n"
    l1_resps = state.get('l1_responses') or []
    for r in l1_resps:
        model_name = r.model_name if hasattr(r, 'model_name') else r.get('model_name', 'Model')
        response_text = r.response if hasattr(r, 'response') else r.get('response', '')
        aggregation_context += f"--- {model_name} ---\n{response_text}\n\n"
    
    if state.get("feedback"):
        aggregation_context += f"**Previous Critique/Feedback:** {state['feedback']}\n\n"

    aggregation_context += "Provide a final synthesis grounded in the forensic context and Layer 1 perspectives."
    
    sys_prompt = (
        f"You are {state['model_l2'].capitalize()}, the Final Arbiter. "
        "Review Tier 1 logs and the provided context carefully. "
        "At the end, provide your scores in this format: [SCORES: bias=0.1, neutrality=0.9, confidence=0.85, feedback=NONE]"
    )
    
    try:
        deployment_name = os.getenv(f"AZURE_DEPLOYMENT_{state['model_l2'].upper()}", state['model_l2'])
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": aggregation_context}
        ]
        
        response = await azure_client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_completion_tokens=4000,
            temperature=0.4,
        )
        response_text = response.choices[0].message.content or ""
        
        bias, neutral, confidence, feedback = 0.5, 0.5, 0.5, ""
        import re
        match = re.search(r"\[SCORES:\s*bias=([\d.]+),\s*neutrality=([\d.]+),\s*confidence=([\d.]+),\s*feedback=(.*?)\]", response_text)
        if match:
            bias = float(match.group(1))
            neutral = float(match.group(2))
            confidence = float(match.group(3))
            feedback = match.group(4).strip()
            response_text = response_text.replace(match.group(0), "").strip()
        
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
            "confidence": confidence * 100
        }
    except Exception as e:
        print(f"[Layer 2 Error] {str(e)}")
        return {"needs_reconsideration": False}

def should_continue(state: GraphState):
    if state.get("needs_reconsideration"):
        return "layer1_node"
    return END

workflow = StateGraph(GraphState)
workflow.add_node("retrieval_node", retrieval_node)
workflow.add_node("layer1_node", layer1_node)
workflow.add_node("layer2_node", layer2_node)

workflow.set_entry_point("retrieval_node")
workflow.add_edge("retrieval_node", "layer1_node")
workflow.add_edge("layer1_node", "layer2_node")
workflow.add_conditional_edges("layer2_node", should_continue, {"layer1_node": "layer1_node", END: END})

app_graph = workflow.compile()

def get_app():
    return app_graph
