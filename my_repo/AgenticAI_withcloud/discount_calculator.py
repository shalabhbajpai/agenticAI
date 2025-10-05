# Conditional Workflow: Discount Calculator
# This workflow calculates a discount on a purchase based on the total amount:
# • If total ≥ 100, apply a 20% discount.
# • If 50 ≤ total < 100, apply a 10% discount.
# • If total < 50, no discount.
# The workflow takes the purchase amount, checks eligibility, and applies the appropriate discount.


from langgraph.graph import StateGraph, START,END
from typing import TypedDict, Dict

class DiscountState(TypedDict):
    total: float
    discount_rate: float
    final_price: float

def check_eligibility(state: DiscountState) -> Dict:
    total = state['total']
    if total >= 100:
        return {"discount_rate": 0.20}
    elif total >= 50:
        return {"discount_rate": 0.10}
    else:
        return {"discount_rate": 0.0}

def apply_discount(state: DiscountState) -> Dict:
    total = state['total']
    rate = state['discount_rate']
    final = total * (1 - rate)
    return {"final_price": final}

def no_discount(state: DiscountState) -> Dict:
    return {"final_price": state['total']}

def route_discount(state: DiscountState) -> str:
    rate = state['discount_rate']
    return "apply_discount" if rate > 0 else "no_discount"

graph = StateGraph(DiscountState)
graph.add_node("apply_discount", apply_discount)
graph.add_node("check_eligibility", check_eligibility)
graph.add_node("no_discount", no_discount)



graph.add_edge(START, "check_eligibility")


graph.add_conditional_edges("check_eligibility", route_discount, {
    "apply_discount": "apply_discount",
    "no_discount": "no_discount"
})



graph.add_edge("apply_discount", END)
#graph.add_edge("no_discount", END)

workflow = graph.compile()

# Test cases
#initial_state = {"total": 120.0}  # 20% discount
#print(workflow.invoke(initial_state))

#initial_state = {"total": 75.0}  # 10% discount
#print(workflow.invoke(initial_state))

initial_state = {"total": 30.0}  # No discount
print(workflow.invoke(initial_state))