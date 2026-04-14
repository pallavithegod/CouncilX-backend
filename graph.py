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

class GraphState(TypedDict):
    question: str
    history: List[Dict[str, str]]
    models_l1: List[str]
    model_l2: str
    model_l3: str
    model_l4: str
    l1_responses: List[ModelResponse]
    l2_response: Optional[ModelResponse]
    l3_response: Optional[ModelResponse]
    l4_response: Optional[ModelResponse]
    iterations: int
    feedback: str
    needs_reconsideration: bool
    confidence: int  # 0-100 scale

async def route_query(model_id: str, persona: str, user_content: str, history: List[Dict[str, str]] = []) -> ModelResponse:
    try:
        deployment_name = os.getenv(f"AZURE_DEPLOYMENT_{model_id.upper()}", model_id)
        
        # Construct messages with history
        messages = [{"role": "system", "content": persona}]
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_content})

        response = await azure_client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_completion_tokens=800,
            temperature=0.7,
        )
        return ModelResponse(model_id=model_id, model_name=model_id.capitalize(), response=response.choices[0].message.content or "", status="success")
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
        tasks.append(route_query(m, sys_prompt, state['question'] + feedback_context, state.get('history', [])))
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

async def layer3_node(state: GraphState):
    print("--- Executing Layer 3 ---")
    print(f"DEBUG: Layer 3 State Keys -> {list(state.keys())}")
    l2_resp = state.get('l2_response')
    refinement_context = f"**User Question:** {state.get('question', '')}\n\n**1st Speaker Synthesis:**\n{l2_resp.response if l2_resp else ''}\n\nPerform a forensic audit on this synthesis. Evaluate its logical integrity, factual depth, and neutrality. \n\n1. Assign a 'CONFIDENCE_SCORE' between 0 and 100.\n2. If the confidence is less than 50, provide specific critical feedback and start with 'RECONSIDER'.\n3. If confidence is 50 or higher, refine the synthesis slightly if needed but do not trigger a rebuild.\n\nFORMAT: [CONFIDENCE_SCORE: X] followed by your audit."
    
    sys_prompt = f"You are {state.get('model_l3', 'Auditor').capitalize()}, the 2nd Speaker Auditor. You are an adversarial firewall against bias. Your mandate is to ensure the synthesis is ready for the Final Arbiter. BE CONCISE."
    l3_resp = await route_query(state.get('model_l3', 'gpt-4o'), sys_prompt, refinement_context, state.get('history', []))
    
    # Extract confidence score
    confidence = 100
    try:
        import re
        match = re.search(r"CONFIDENCE_SCORE:\s*(\d+)", l3_resp.response.upper())
        if match:
            confidence = int(match.group(1))
    except:
        pass

    needs_reconsideration = confidence < 50 and state['iterations'] < 3
    return {
        "l3_response": l3_resp, 
        "needs_reconsideration": needs_reconsideration, 
        "feedback": l3_resp.response if needs_reconsideration else "",
        "confidence": confidence
    }

def should_reconsider(state: GraphState):
    if state["needs_reconsideration"]:
        print(f">>> Auditor triggered RECONSIDERATION loop! (Iteration {state['iterations']}/3). Returning to Phase 1.")
        return "layer1_node"
    return "layer4_node"

async def layer4_node(state: GraphState):
    print("--- Executing Layer 4 ---")
    print(f"DEBUG: Layer 4 State Keys -> {list(state.keys())}")
    l3_resp = state.get('l3_response')
    final_context = f"**User Question:** {state.get('question', '')}\n\n**Auditor's Checked Review:**\n{l3_resp.response if l3_resp else ''}\n\nConstruct the final, universally objective verdict."
    
    sys_prompt = f"You are {state.get('model_l4', 'Arbiter').capitalize()}, the Final Arbiter. Your mandate is to provide the absolute, objective final answer to the user. DO NOT mention previous layers, deliberation steps, or technical reasons. Provide ONLY the direct, helpful, and distilled final response as a normal AI assistant would. Maintain extreme neutrality and factual accuracy."
    l4_resp = await route_query(state.get('model_l4', 'gpt-4o'), sys_prompt, final_context, state.get('history', []))
    return {"l4_response": l4_resp}

workflow = StateGraph(GraphState)
workflow.add_node("layer1_node", layer1_node)
workflow.add_node("layer2_node", layer2_node)
workflow.add_node("layer3_node", layer3_node)
workflow.add_node("layer4_node", layer4_node)

workflow.set_entry_point("layer1_node")
workflow.add_edge("layer1_node", "layer2_node")
workflow.add_edge("layer2_node", "layer3_node")
workflow.add_conditional_edges(
    "layer3_node",
    should_reconsider,
    {
        "layer1_node": "layer1_node",
        "layer4_node": "layer4_node"
    }
)
workflow.add_edge("layer4_node", END)

def get_app():
    return workflow.compile()

app_graph = workflow.compile()
