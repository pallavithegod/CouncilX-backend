import os
import json
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

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
    question: str
    history: List[Dict[str, str]]
    models_l1: List[str]
    model_l2: str
    model_l3: str
    model_l4: str
    l1_responses: List[ModelResponse]
    l2_response: Optional[ModelResponse]
    iterations: int
    feedback: str
    needs_reconsideration: bool
    confidence: int  # 0-100 scale

async def route_query(model_id: str, persona: str, user_content: str, history: List[Dict[str, str]] = [], extract_scores: bool = False) -> ModelResponse:
    try:
        deployment_name = os.getenv(f"AZURE_DEPLOYMENT_{model_id.upper()}", model_id)
        
        prompt_with_scores = persona
        if extract_scores:
            prompt_with_scores += "\n\nIMPORTANT: At the end of your response, provide your self-assessment format exactly as: [SCORES: bias=0.82, neutrality=0.76, clarity=0.71]"

        # Construct messages with history
        messages = [{"role": "system", "content": prompt_with_scores}]
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_content})

        response = await azure_client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_completion_tokens=800,
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
                response_text = response_text[:match.start()].strip()
                
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

async def layer1_node(state: GraphState):
    print("--- Executing Layer 1 ---")
    print(f"DEBUG: Layer 1 State Keys -> {list(state.keys())}")
    import asyncio
    tasks = []
    feedback_context = f"\n\nPre-existing feedback to consider: {state.get('feedback', '')}" if state.get('feedback') else ""
    models = state.get('models_l1') or []
    for m in models:
        sys_prompt = f"You are {m.capitalize()}, a Tier 1 Deliberator. Your mandate is absolute objectivity. You must analyze the user inquiry through a purely factual, non-partisan lens. BE CONCISE. Provide your response followed by the core logical reasons for your stance."
        tasks.append(route_query(m, sys_prompt, state['question'] + feedback_context, state.get('history', []), extract_scores=True))
    l1_resp = await asyncio.gather(*tasks)
    return {"l1_responses": l1_resp, "iterations": state.get("iterations", 0) + 1}

async def layer2_node(state: GraphState):
    print("--- Executing Layer 2 ---")
    print(f"DEBUG: Layer 2 State Keys -> {list(state.keys())}")
    aggregation_context = f"**User Question:** {state.get('question', '')}\n\n**Layer 1 Perspectives:**\n"
    l1_resps = state.get('l1_responses') or []
    for r in l1_resps:
        aggregation_context += f"--- {r.model_name} ---\n{r.response}\n\n"
    aggregation_context += "Critically review these Tier 1 perspectives for logical fallacies, emotional framing, or subjective bias. Synthesize them into a singular, verified objective framework."
    
    sys_prompt = f"You are {state['model_l2'].capitalize()}, the 1st Speaker. Your task is to verify Tier 1 deliberations and formulate an aggressively neutral synthesis. BE CONCISE. Provide reasoning for your synthesis decisions."
    l2_resp = await route_query(state['model_l2'], sys_prompt, aggregation_context, state.get('history', []))
    return {"l2_response": l2_resp}

workflow = StateGraph(GraphState)
workflow.add_node("layer1_node", layer1_node)
workflow.add_node("layer2_node", layer2_node)

workflow.set_entry_point("layer1_node")
workflow.add_edge("layer1_node", "layer2_node")
workflow.add_edge("layer2_node", END)

def get_app():
    return workflow.compile()

app_graph = workflow.compile()
