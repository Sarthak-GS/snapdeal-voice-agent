"""
diagnostic.py — Dynamic health score computation and live account diagnostics.
"""

def compute_health_score(kb: dict) -> dict:
    """Computes a 0-100 account health score based on OOS, cancellations, pricing, and festive gaps."""
    p = kb["performance"]["this_month"]
    lm = kb["performance"]["last_month"]
    market = kb["market_context"]
    total_skus = kb["seller_profile"].get("total_catalog_skus", 25)
    score = 100

    oos_ratio = p["oos_products"] / total_skus
    score -= min(25, round(oos_ratio * 25))
    if market.get("oos_ads_paused"):
        score += 10

    cancel_rate = p["cancellation_rate"]
    if cancel_rate >= 0.15:
        score -= 20
    elif cancel_rate >= 0.10:
        score -= 14
    elif cancel_rate >= 0.08:
        score -= 8

    rev_drop = (lm["revenue"] - p["revenue"]) / lm["revenue"]
    if rev_drop >= 0.30:
        score -= 20
    elif rev_drop >= 0.15:
        score -= 10

    overpriced = sum(1 for prod in kb["products"] if prod["price"] > prod["competitor_avg"] * 1.05)
    score -= min(20, overpriced * 5)

    if not market["seller_enrolled_festive"] and not market.get("seller_enrolled_plus"):
        score -= 15
    elif not market["seller_enrolled_festive"] or not market.get("seller_enrolled_plus"):
        score -= 7

    score = max(0, min(100, score))
    band = "healthy" if score >= 80 else "watch" if score >= 60 else "at_risk" if score >= 40 else "critical"
    return {"score": score, "band": band}


def build_account_snapshot(kb: dict) -> str:
    """Creates a compact, factual diagnostic snapshot for LLM context injection."""
    prof = kb["seller_profile"]
    lm = kb["performance"]["last_month"]
    tm = kb["performance"]["this_month"]
    market = kb["market_context"]
    products = kb["products"]
    health = compute_health_score(kb)

    rev_drop = lm["revenue"] - tm["revenue"]
    rev_drop_pct = (rev_drop / lm["revenue"]) * 100
    order_drop = lm["orders"] - tm["orders"]
    oos_delta = tm["oos_products"] - lm["oos_products"]

    oos_skus = [p for p in products if p["status"] == "Out of Stock"]
    oos_loss = sum((p["sales_last_month"] - p["sales_this_month"]) * p["price"] for p in oos_skus)

    overpriced = []
    for p in products:
        if p["price"] > p["competitor_avg"] * 1.05:
            net = p["competitor_avg"] - p["unit_cost"] - p["platform_fee"]
            overpriced.append(f"{p['name']}: ₹{p['price']} vs mkt ₹{p['competitor_avg']} (net ₹{net}/unit @ mkt price)")

    top_cancel = kb["cancellation_breakdown"]["reasons"][0]

    lines = [
        "=== SELLER ACCOUNT SNAPSHOT ===",
        f"Seller: {prof['name']} | Store: {prof['store']} | {prof['city']} | {prof['tier']}",
        f"Category: {prof['category']} | Health: {health['score']}/100 ({health['band'].upper()})",
        "",
        f"PERFORMANCE ({lm['label']} → {tm['label']}):",
        f"  Revenue: ₹{lm['revenue']:,} → ₹{tm['revenue']:,} (DOWN {rev_drop_pct:.1f}% = ₹{rev_drop:,} lost)",
        f"  Orders: {lm['orders']} → {tm['orders']} (lost {order_drop} orders)",
        f"  Cancellation: {lm['cancellation_rate']*100:.0f}% → {tm['cancellation_rate']*100:.0f}% (platform avg: {market['platform_avg_cancel_rate']*100:.0f}%)",
        f"  Return Rate: {lm['return_rate']*100:.1f}% → {tm['return_rate']*100:.1f}%",
        f"  Conversion: {lm['conversion_rate']*100:.1f}% → {tm['conversion_rate']*100:.1f}%",
        f"  Out-of-Stock: {lm['oos_products']} → {tm['oos_products']} SKUs (+{oos_delta} new stockouts)",
        "",
        "ROOT CAUSES:",
        f"  OOS SKUs: {', '.join(p['name'] for p in oos_skus) or 'None'} → ₹{oos_loss:,} direct revenue lost",
        f"  Top cancel reason: {top_cancel['reason']} ({top_cancel['pct']}% = {top_cancel['count']} orders)",
        f"  Overpriced vs market: {'; '.join(overpriced) if overpriced else 'None — all aligned'}",
        f"  Category growth: +{market['category_growth_this_month']*100:.0f}% | Festive enrolled: {market['seller_enrolled_festive']}",
        f"  OOS ads paused: {market.get('oos_ads_paused', False)}",
    ]
    if market.get("big_diwali_sale_missed"):
        lines.append("  ⚠ Missed Big Diwali Sale enrollment window")
    if market.get("avg_delivery_days"):
        lines.append(f"  Avg delivery: {market['avg_delivery_days']} days (benchmark ≤4 days)")

    return "\n".join(lines)
