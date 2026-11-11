"""
tools.py — ReAct tool definitions and execution dispatcher for KIRAN.
"""

import re
import json
from app.config import logger
from app.data_loader import vector_stores
from app.diagnostic import compute_health_score
import generate_data

# ═══════════════════════════════════════════════════════════════════════════════
# TOOL SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════
TOOL_DESCRIPTIONS = """
TOOLS YOU CAN CALL:
1. get_seller_metrics: parameters {"metric_group": "overview"|"products"|"cancellations"|"market"}
2. analyze_issue: parameters {"issue_type": "inventory"|"pricing"|"cancellations"|"returns"|"delivery"|"visibility"|"conversion"}
3. get_recommendations: parameters {"query": "search phrase"}
4. execute_account_action: parameters {"action": "pause_oos_ads"|"update_prices_to_market"|"enroll_festive_deals"|"pause_tv_ads"} (only upon explicit seller consent)
5. generate_call_summary: parameters {"primary_issues": ["issue 1", "issue 2"], "actions_agreed": ["action 1", "action 2"]} (when closing call)
"""


def _tool_get_metrics(session: dict, metric_group: str) -> str:
    kb = session.get("kb", {})
    if not kb:
        return "Session data not found."

    prof = kb["seller_profile"]
    lm = kb["performance"]["last_month"]
    tm = kb["performance"]["this_month"]
    market = kb["market_context"]
    products = kb["products"]

    if metric_group == "overview":
        rev_drop_pct = (lm["revenue"] - tm["revenue"]) / lm["revenue"] * 100
        return (
            f"PERFORMANCE OVERVIEW for {prof['name']} ({prof['store']}, {prof['city']}):\n"
            f"Revenue: ₹{lm['revenue']:,} → ₹{tm['revenue']:,} (DOWN {rev_drop_pct:.1f}%)\n"
            f"Orders: {lm['orders']} → {tm['orders']}\n"
            f"Cancellation Rate: {lm['cancellation_rate']*100:.0f}% → {tm['cancellation_rate']*100:.0f}% (platform avg {market['platform_avg_cancel_rate']*100:.0f}%)\n"
            f"Return Rate: {lm['return_rate']*100:.1f}% → {tm['return_rate']*100:.1f}%\n"
            f"Conversion Rate: {lm['conversion_rate']*100:.1f}% → {tm['conversion_rate']*100:.1f}%\n"
            f"Out-of-Stock: {lm['oos_products']} → {tm['oos_products']} SKUs\n"
            f"Health Score: {compute_health_score(kb)['score']}/100 ({compute_health_score(kb)['band'].upper()})"
        )
    elif metric_group == "products":
        lines = [f"SKU DETAILS ({len(products)} products):"]
        for p in products:
            price_note = f" ⚠ OVERPRICED vs mkt ₹{p['competitor_avg']}" if p["price"] > p["competitor_avg"] * 1.05 else ""
            quality_note = f"\n    ⚠ Quality issues: {p.get('quality_issues', '')}" if p.get("quality_issues") else ""
            lines.append(
                f"  {p['name']} [{p['status']}] ₹{p['price']}{price_note}\n"
                f"    Rating: {p['rating']} | Listing score: {p['listing_score']} | Stock: {p['stock']}"
                f"{quality_note}"
            )
        return "\n".join(lines)
    elif metric_group == "cancellations":
        cb = kb["cancellation_breakdown"]
        reasons = "\n".join(f"  {r['reason']}: {r['pct']}% ({r['count']} orders)" for r in cb["reasons"])
        return (
            f"CANCELLATION BREAKDOWN (total {cb['total_this_month']} cancellations this month):\n"
            f"{reasons}\n"
            f"Platform benchmark: {market['platform_avg_cancel_rate']*100:.0f}%\n"
            f"Current rate: {tm['cancellation_rate']*100:.0f}%"
        )
    elif metric_group == "market":
        extra = f"Avg delivery days: {market['avg_delivery_days']} (benchmark ≤4)" if market.get("avg_delivery_days") else ""
        return (
            f"MARKET CONTEXT for {prof['category']}:\n"
            f"Category growth this month: +{market['category_growth_this_month']*100:.0f}%\n"
            f"Seller growth: {market['seller_growth_this_month']*100:.1f}%\n"
            f"New competitors added: {market.get('active_competitors_added', 'N/A')}\n"
            f"Festive window: {market.get('festive_window', 'N/A')}\n"
            f"Seller enrolled in festive deals: {market['seller_enrolled_festive']}\n"
            f"OOS ads paused: {market.get('oos_ads_paused', False)}\n"
            f"{extra}"
        )
    return f"Unknown metric_group: {metric_group}"


def _tool_analyze_issue(session: dict, issue_type: str) -> str:
    kb = session.get("kb", {})
    if not kb:
        return "Session data not found."

    tm = kb["performance"]["this_month"]
    lm = kb["performance"]["last_month"]
    products = kb["products"]
    market = kb["market_context"]
    hypotheses = kb.get("causal_hypotheses", [])

    if issue_type == "inventory":
        oos_skus = [p for p in products if p["status"] == "Out of Stock"]
        oos_loss = sum((p["sales_last_month"] - p["sales_this_month"]) * p["price"] for p in oos_skus)
        h = next((h for h in hypotheses if "inventory" in h["title"].lower() or "stock" in h["title"].lower()), None)
        result = (
            f"INVENTORY ANALYSIS:\n"
            f"OOS SKUs: {', '.join(p['name'] for p in oos_skus) or 'None'}\n"
            f"Direct revenue lost from OOS: ₹{oos_loss:,}\n"
            f"OOS delta (last→this month): +{tm['oos_products'] - lm['oos_products']} stockouts\n"
        )
        if h:
            result += f"\nHypothesis: {h['title']}\nObservation: {h['observation']}\nAction: {h['action']}\nProjected recovery: {h['projected_recovery']}"
        return result

    elif issue_type == "pricing":
        overpriced = [p for p in products if p["price"] > p["competitor_avg"] * 1.05]
        lines = []
        for p in overpriced:
            net = p["competitor_avg"] - p["unit_cost"] - p["platform_fee"]
            margin_pct = (net / p["competitor_avg"]) * 100
            lines.append(f"  {p['name']}: ₹{p['price']} vs mkt ₹{p['competitor_avg']} → match to ₹{p['competitor_avg']} keeps ₹{net}/unit ({margin_pct:.1f}% net margin)")
        h = next((h for h in hypotheses if "pric" in h["title"].lower()), None)
        result = f"PRICING ANALYSIS:\n{'No overpriced SKUs found.' if not lines else chr(10).join(lines)}"
        if h:
            result += f"\n\nHypothesis: {h['title']}\nObservation: {h['observation']}\nProjected recovery: {h['projected_recovery']}"
        return result

    elif issue_type == "cancellations":
        cb = kb["cancellation_breakdown"]
        top = cb["reasons"][0]
        return (
            f"CANCELLATION ANALYSIS:\n"
            f"Rate: {lm['cancellation_rate']*100:.0f}% → {tm['cancellation_rate']*100:.0f}% (platform: {market['platform_avg_cancel_rate']*100:.0f}%)\n"
            f"Total this month: {cb['total_this_month']} cancellations\n"
            f"Top reason: {top['reason']} ({top['pct']}% = {top['count']} orders)\n"
            f"All reasons: {'; '.join(r['reason']+' '+str(r['pct'])+'%' for r in cb['reasons'])}"
        )

    elif issue_type == "returns":
        high_return = [p for p in products if p.get("quality_issues")]
        result = f"RETURN ANALYSIS:\nReturn rate: {lm['return_rate']*100:.1f}% → {tm['return_rate']*100:.1f}%\n"
        if high_return:
            result += f"SKUs with quality issues: {', '.join(p['name'] for p in high_return)}\n"
            result += "\n".join(f"  {p['name']}: {p['quality_issues']}" for p in high_return)
        h = next((h for h in hypotheses if "return" in h["title"].lower() or "quality" in h["title"].lower()), None)
        if h:
            result += f"\n\nHypothesis: {h['title']}\nAction: {h['action']}\nProjected recovery: {h['projected_recovery']}"
        return result

    elif issue_type == "visibility":
        low_score = [p for p in products if p["listing_score"] < 70]
        return (
            f"VISIBILITY ANALYSIS:\n"
            f"Category growth: +{market['category_growth_this_month']*100:.0f}%\n"
            f"Festive enrolled: {market['seller_enrolled_festive']}\n"
            f"SKUs with listing score <70: {', '.join(p['name']+' ('+str(p['listing_score'])+')' for p in low_score) or 'None'}"
        )

    elif issue_type == "conversion":
        return (
            f"CONVERSION ANALYSIS:\n"
            f"Conversion rate: {lm['conversion_rate']*100:.1f}% → {tm['conversion_rate']*100:.1f}%\n"
            f"Orders: {lm['orders']} → {tm['orders']}\n"
            f"Avg selling price: ₹{lm['avg_selling_price']} → ₹{tm['avg_selling_price']}\n"
            f"Likely drivers: pricing above market, low listing scores, high cancellation rate hurting search rank"
        )

    elif issue_type == "delivery":
        avg_days = market.get("avg_delivery_days", "N/A")
        cb = kb["cancellation_breakdown"]
        delay_cancel = next((r for r in cb["reasons"] if "delay" in r["reason"].lower()), None)
        result = f"DELIVERY ANALYSIS:\nAvg delivery time: {avg_days} days (benchmark ≤4 days)\n"
        if delay_cancel:
            result += f"Delivery-related cancellations: {delay_cancel['pct']}% ({delay_cancel['count']} orders)\n"
        h = next((h for h in hypotheses if "delivery" in h["title"].lower() or "sfc" in h["title"].lower()), None)
        if h:
            result += f"\nHypothesis: {h['title']}\nAction: {h['action']}\nProjected recovery: {h['projected_recovery']}"
        return result

    return f"Unknown issue_type: {issue_type}"


def _tool_get_recommendations(session: dict, query: str) -> str:
    seller_id = session.get("seller_id", generate_data.DEFAULT_SELLER_ID)
    vs = vector_stores.get(seller_id)
    if not vs:
        return "Vector store not found."
    snippets = vs.retrieve_sync(query, top_k=2)
    if not snippets:
        return "No matching playbook entries found."
    result = f"PLAYBOOK RECOMMENDATIONS for '{query}':\n"
    for s in snippets:
        result += f"\n[{s['id']}] {s['title']}\nApplies when: {s['applies_when']}\n{s['text']}\n"
    return result


def _tool_generate_summary(session: dict, primary_issues: list, actions_agreed: list) -> str:
    kb = session.get("kb", {})
    if not kb:
        return "Session data not found."

    prof = kb["seller_profile"]
    lm = kb["performance"]["last_month"]
    tm = kb["performance"]["this_month"]
    market = kb["market_context"]
    executed = session.get("executed_actions", [])
    hypotheses = kb.get("causal_hypotheses", [])

    rev_drop_pct = (lm["revenue"] - tm["revenue"]) / lm["revenue"] * 100

    day1_status = "[DONE ON CALL] " if market.get("oos_ads_paused") else ""
    day3_status = "[DONE ON CALL] " if any("price" in a.lower() for a in executed) else ""
    day7_status = "[DONE ON CALL] " if market.get("seller_enrolled_festive") else ""

    h1 = hypotheses[0] if len(hypotheses) > 0 else {}
    h2 = hypotheses[1] if len(hypotheses) > 1 else {}
    h3 = hypotheses[2] if len(hypotheses) > 2 else {}

    total_recovery = 0
    for h in hypotheses:
        try:
            rec = h.get("projected_recovery", "0")
            num = int(re.sub(r"[^\d]", "", rec.split("/")[0]))
            total_recovery += num
        except Exception:
            pass

    blueprint = {
        "seller": prof["name"],
        "period": f"{lm['label']} → {tm['label']}",
        "rev_drop_pct": f"{rev_drop_pct:.1f}%",
        "problem": f"Revenue fell {rev_drop_pct:.1f}% (₹{lm['revenue']:,} → ₹{tm['revenue']:,}) | Issues: {', '.join(primary_issues)}",
        "primary_hypothesis": h1.get("mechanism", "Restoring inventory and fixing core issues recovers lost revenue."),
        "target_metric": h1.get("metric", "Cancellation rate down; conversion up; revenue recovering."),
        "day1_action": day1_status + h1.get("action", actions_agreed[0] if actions_agreed else "Take immediate inventory action."),
        "day3_action": day3_status + h2.get("action", actions_agreed[1] if len(actions_agreed) > 1 else "Fix pricing and listing issues."),
        "day7_action": day7_status + h3.get("action", actions_agreed[2] if len(actions_agreed) > 2 else "Enroll in promotional campaigns."),
        "financial_effect": f"Projected monthly recovery: ₹{total_recovery:,} from resolving all identified issues",
        "stop_condition": h1.get("stop_condition", "Monitor weekly; revert if metrics worsen."),
        "followup": f"KIRAN will review progress with {prof['name']} in 3 days",
        "executed_this_call": executed,
    }

    session["blueprint"] = blueprint
    return json.dumps(blueprint, ensure_ascii=False, indent=2)


def _tool_execute_action(session: dict, action: str) -> str:
    kb = session.get("kb", {})
    if not kb:
        return "Session data not found."

    mkt = kb["market_context"]
    products = kb["products"]

    if action == "pause_oos_ads":
        if mkt.get("oos_ads_paused"):
            return "OOS ads already paused."
        mkt["oos_ads_paused"] = True
        tm = kb["performance"]["this_month"]
        old_cancel = tm["cancellation_rate"]
        tm["cancellation_rate"] = max(0.06, old_cancel - 0.04)
        oos_count = tm["oos_products"]
        msg = f"✅ EXECUTED: Paused sponsored ads on {oos_count} OOS listings. Auto-pause enabled below 2 units. Cancellation rate projected to drop from {old_cancel*100:.0f}% to {tm['cancellation_rate']*100:.0f}%."
        session["executed_actions"].append(msg)
        return msg

    elif action == "update_prices_to_market":
        changed = []
        for p in products:
            if p["price"] > p["competitor_avg"] * 1.05:
                old_price = p["price"]
                p["price"] = p["competitor_avg"]
                net = p["competitor_avg"] - p["unit_cost"] - p["platform_fee"]
                margin_pct = (net / p["competitor_avg"]) * 100
                changed.append(f"{p['name']}: ₹{old_price} → ₹{p['competitor_avg']} (net ₹{net}/unit = {margin_pct:.1f}% margin)")
        if not changed:
            return "No overpriced SKUs found — all prices already aligned to market."
        msg = f"✅ EXECUTED: Price-matched {len(changed)} SKUs to market rates: {'; '.join(changed)}"
        session["executed_actions"].append(msg)
        return msg

    elif action == "enroll_festive_deals":
        if mkt.get("seller_enrolled_festive"):
            return "Already enrolled in festive deals."
        mkt["seller_enrolled_festive"] = True
        eligible = [p["name"] for p in products if p["status"] == "Active" and p["stock"] > 30]
        msg = f"✅ EXECUTED: Enrolled {len(eligible)} in-stock SKUs in Snapdeal Festive Deals: {', '.join(eligible[:3])}. Organic impressions projected +35%."
        session["executed_actions"].append(msg)
        return msg

    elif action == "pause_tv_ads":
        tv_sku = next((p for p in products if "tv" in p["name"].lower() or "television" in p["name"].lower()), None)
        if not tv_sku:
            return "No TV SKU found."
        tv_sku["ads_paused"] = True
        msg = f"✅ EXECUTED: Paused sponsored ads on '{tv_sku['name']}' to prevent new sales of potentially defective units while quality investigation is underway."
        session["executed_actions"].append(msg)
        return msg

    return f"Unknown action: {action}"


def execute_tool(tool_name: str, tool_input: dict, session: dict) -> str:
    """Dispatches tool calls to the appropriate handler using session state."""
    try:
        if tool_name == "get_seller_metrics":
            return _tool_get_metrics(session, tool_input.get("metric_group", "overview"))
        elif tool_name == "analyze_issue":
            return _tool_analyze_issue(session, tool_input.get("issue_type", "inventory"))
        elif tool_name == "get_recommendations":
            return _tool_get_recommendations(session, tool_input.get("query", ""))
        elif tool_name == "generate_call_summary":
            return _tool_generate_summary(session, tool_input.get("primary_issues", []), tool_input.get("actions_agreed", []))
        elif tool_name == "execute_account_action":
            return _tool_execute_action(session, tool_input.get("action", ""))
        else:
            return f"Unknown tool: {tool_name}"
    except Exception as e:
        logger.error(f"[tool:{tool_name}] Error: {e}")
        return f"Tool error: {e}"
