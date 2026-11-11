"""
generate_data.py — Offline Knowledge Base & Vector Embedding Generator.
Creates per-seller:
  data/{seller_id}_kb.json
  data/{seller_id}_embeddings.npz

Run:
    python3 generate_data.py
"""

import json
import re
from pathlib import Path
import numpy as np
import httpx

DATA_DIR = Path(__file__).parent / "data"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

SELLER_IDS = ["rahul", "priya", "farouk"]
DEFAULT_SELLER_ID = "rahul"


def kb_path(seller_id: str) -> Path:
    return DATA_DIR / f"{seller_id}_kb.json"


def embed_path(seller_id: str) -> Path:
    return DATA_DIR / f"{seller_id}_embeddings.npz"


# Legacy single-seller paths (kept so old main.py imports don't break immediately)
KB_PATH = kb_path(DEFAULT_SELLER_ID)
EMBED_PATH = embed_path(DEFAULT_SELLER_ID)


def get_raw_knowledge_base(seller_id: str) -> dict:
    if seller_id == "rahul":
        return {
            "seller_profile": {
                "id": "SD-78234",
                "name": "Rahul Sharma",
                "store": "Rahul Electronics & Accessories",
                "category": "Mobile Accessories",
                "city": "Pune",
                "tier": "Silver Seller",
                "rating": 4.1,
                "reviews": 1240,
                "total_catalog_skus": 25,
                "joined_months_ago": 18,
            },
            "performance": {
                "last_month": {
                    "label": "October 2026",
                    "revenue": 200000,
                    "orders": 400,
                    "avg_selling_price": 500,
                    "cancellation_rate": 0.08,
                    "oos_products": 5,
                    "return_rate": 0.04,
                    "conversion_rate": 0.024,
                },
                "this_month": {
                    "label": "November 2026",
                    "revenue": 135000,
                    "orders": 270,
                    "avg_selling_price": 500,
                    "cancellation_rate": 0.14,
                    "oos_products": 18,
                    "return_rate": 0.045,
                    "conversion_rate": 0.018,
                },
            },
            "products": [
                {
                    "id": "SKU-001", "name": "Fast USB-C Charging Cable 1.5m",
                    "category": "Cables", "price": 299, "competitor_avg": 249,
                    "unit_cost": 150, "platform_fee": 39, "rating": 4.3,
                    "listing_score": 62, "status": "Out of Stock", "stock": 0,
                    "lead_time_days": 3, "sales_last_month": 85, "sales_this_month": 12,
                },
                {
                    "id": "SKU-002", "name": "Heavy Duty 20000mAh Power Bank",
                    "category": "Power Banks", "price": 1299, "competitor_avg": 1099,
                    "unit_cost": 740, "platform_fee": 145, "rating": 4.0,
                    "listing_score": 55, "status": "Low Stock", "stock": 8,
                    "lead_time_days": 2, "sales_last_month": 72, "sales_this_month": 23,
                },
                {
                    "id": "SKU-003", "name": "Tempered Glass Screen Protector (Pack of 2)",
                    "category": "Screen Protectors", "price": 149, "competitor_avg": 99,
                    "unit_cost": 48, "platform_fee": 22, "rating": 4.2,
                    "listing_score": 48, "status": "Out of Stock", "stock": 0,
                    "lead_time_days": 4, "sales_last_month": 38, "sales_this_month": 0,
                },
                {
                    "id": "SKU-004", "name": "Wired Earphones with Mic",
                    "category": "Audio", "price": 399, "competitor_avg": 349,
                    "unit_cost": 210, "platform_fee": 52, "rating": 4.1,
                    "listing_score": 78, "status": "Active", "stock": 230,
                    "lead_time_days": 2, "sales_last_month": 110, "sales_this_month": 90,
                },
                {
                    "id": "SKU-005", "name": "Shockproof Silicone Back Cover",
                    "category": "Cases", "price": 199, "competitor_avg": 189,
                    "unit_cost": 95, "platform_fee": 30, "rating": 4.5,
                    "listing_score": 85, "status": "Active", "stock": 450,
                    "lead_time_days": 2, "sales_last_month": 95, "sales_this_month": 145,
                },
            ],
            "cancellation_breakdown": {
                "total_this_month": 38,
                "reasons": [
                    {"reason": "Out of stock post-order", "pct": 42, "count": 16},
                    {"reason": "Delivery delay expected", "pct": 25, "count": 10},
                    {"reason": "Price issue / found cheaper", "pct": 18, "count": 7},
                    {"reason": "Buyer changed mind", "pct": 15, "count": 5},
                ],
            },
            "market_context": {
                "category_growth_this_month": 0.08,
                "seller_growth_this_month": -0.325,
                "platform_avg_cancel_rate": 0.06,
                "active_competitors_added": 12,
                "festive_window": "Active — Diwali & November Gifting Season",
                "seller_enrolled_festive": False,
                "seller_enrolled_plus": False,
                "oos_ads_paused": False,
            },
            "causal_hypotheses": [
                {
                    "id": "H1", "priority": "P0 - Critical",
                    "title": "Inventory Restock & Listing Protection",
                    "observation": "18 of 25 SKUs OOS (+13 this month); ₹27,489 direct revenue lost; 42% cancellations from post-order stockouts.",
                    "mechanism": "Restoring 50-unit buffer on USB-C Cable & Screen Protector and pausing OOS ads drops cancellations 14%→≤6%.",
                    "action": "Restock top 2 SKUs within 24h; pause ads on all 18 OOS listings; auto-pause below 2 units.",
                    "metric": "Cancellation 14%→≤6%; weekly revenue +₹15,000.",
                    "stop_condition": "Switch to backup Pune distributor if primary exceeds 5 days.",
                    "projected_recovery": "₹27,489/mo",
                },
                {
                    "id": "H2", "priority": "P1 - High",
                    "title": "Margin-Safe Competitive Price Alignment",
                    "observation": "Top SKUs 14–50% above market (USB-C ₹299 vs ₹249; Power Bank ₹1,299 vs ₹1,099).",
                    "mechanism": "Matching market price preserves 19–24% net margin and rebounds conversion 1.8%→2.4%.",
                    "action": "Bulk-update USB-C Cable to ₹249 (24.1% net margin) and Power Bank to ₹1,099 (19.5% net margin).",
                    "metric": "Conversion 1.8%→2.4%; +25 orders/week.",
                    "stop_condition": "Halt if unit net margin drops below 15%.",
                    "projected_recovery": "₹14,500/mo",
                },
                {
                    "id": "H3", "priority": "P2 - Medium",
                    "title": "Festive Deal Campaign Onboarding",
                    "observation": "Mobile Accessories grew +8% but seller has 0 festive enrollments.",
                    "mechanism": "Enrolling in-stock listings in November Festive Deals lifts organic impressions ~35%.",
                    "action": "Enroll via Promotions tab in Seller Portal (10 mins).",
                    "metric": "Daily organic impressions +35%.",
                    "stop_condition": "Restrict to SKUs with >30 units stock.",
                    "projected_recovery": "₹18,000/mo",
                },
            ],
            "playbook": [
                {
                    "id": "pb_oos_emergency",
                    "title": "Out-of-Stock Emergency Response Protocol",
                    "tags": ["inventory", "oos", "out of stock", "stockout", "supplier", "delay", "restock"],
                    "applies_when": "oos_delta >= 5 or cancellation_rate >= 0.10",
                    "text": (
                        "Prioritise restocking the top 2 revenue-generating OOS SKUs (USB-C Cable and Screen Protector) within 24-48 hours. "
                        "Immediately pause sponsored ad campaigns on all 18 OOS listings to stop wasting budget. "
                        "Enable auto-pause when stock falls below 2 units to eliminate post-order stockout cancellations."
                    ),
                },
                {
                    "id": "pb_cancellation_triage",
                    "title": "Cancellation Rate & Search Rank Recovery",
                    "tags": ["cancellation", "cancelled", "cancel", "penalty", "ranking", "post-order", "orders"],
                    "applies_when": "cancellation_rate >= 0.10",
                    "text": (
                        "When cancellation rate exceeds 10% (currently 14% vs 6% platform benchmark), Snapdeal search demotes listing visibility. "
                        "42% of cancellations are caused by post-order stockouts and 25% by courier delays. "
                        "Enabling low-stock auto-pause and dispatching in-stock orders within 24 hours brings cancellations below 6%."
                    ),
                },
                {
                    "id": "pb_pricing_correction",
                    "title": "Margin-Safe Competitive Pricing Alignment",
                    "tags": ["pricing", "price", "expensive", "competitor", "margin", "conversion", "discount"],
                    "applies_when": "price > competitor_avg * 1.10",
                    "text": (
                        "Price-match USB-C Cable from ₹299 to ₹249 (retains ₹60 profit = 24.1% net margin) and Power Bank from ₹1,299 to ₹1,099 "
                        "(retains ₹214 profit = 19.5% net margin). Use Snapdeal's bulk price-update tool to recover conversion from 1.8% to 2.4%."
                    ),
                },
                {
                    "id": "pb_listing_quality",
                    "title": "Listing Quality Score & Image Optimisation",
                    "tags": ["listing", "images", "score", "visibility", "seo", "keywords", "photos"],
                    "applies_when": "listing_score < 70",
                    "text": (
                        "3 SKUs have listing scores below 70 (USB-C: 62, Power Bank: 55, Screen Protector: 48). "
                        "Upload at least 4 high-resolution product images and add 5 search keywords in bullet points to push scores above 75."
                    ),
                },
                {
                    "id": "pb_festive_enrollment",
                    "title": "November Festive Deals & Traffic Capture",
                    "tags": ["festive", "diwali", "promotion", "deals", "traffic", "growth", "competitors"],
                    "applies_when": "not seller_enrolled_festive",
                    "text": (
                        "The Mobile Accessories category grew 8% this month while store revenue fell 32.5%. "
                        "Enroll in-stock SKUs (Wired Earphones and Silicone Covers) into Snapdeal Festive Deals via the Promotions tab in 10 minutes."
                    ),
                },
            ],
        }

    elif seller_id == "priya":
        return {
            "seller_profile": {
                "id": "SD-45891",
                "name": "Priya Mehta",
                "store": "Priya's Fashion Studio",
                "category": "Women's Fashion",
                "city": "Mumbai",
                "tier": "Gold Seller",
                "rating": 4.3,
                "reviews": 2870,
                "total_catalog_skus": 38,
                "joined_months_ago": 29,
            },
            "performance": {
                "last_month": {
                    "label": "October 2026",
                    "revenue": 320000,
                    "orders": 640,
                    "avg_selling_price": 500,
                    "cancellation_rate": 0.06,
                    "oos_products": 3,
                    "return_rate": 0.08,
                    "conversion_rate": 0.031,
                },
                "this_month": {
                    "label": "November 2026",
                    "revenue": 210000,
                    "orders": 390,
                    "avg_selling_price": 538,
                    "cancellation_rate": 0.09,
                    "oos_products": 8,
                    "return_rate": 0.12,
                    "conversion_rate": 0.021,
                },
            },
            "products": [
                {
                    "id": "SKU-P01", "name": "Cotton Kurti Set with Dupatta",
                    "category": "Kurtis", "price": 749, "competitor_avg": 649,
                    "unit_cost": 320, "platform_fee": 95, "rating": 4.4,
                    "listing_score": 58, "status": "Out of Stock", "stock": 0,
                    "lead_time_days": 7, "sales_last_month": 180, "sales_this_month": 22,
                },
                {
                    "id": "SKU-P02", "name": "Embroidered Banarasi Saree",
                    "category": "Sarees", "price": 1299, "competitor_avg": 1199,
                    "unit_cost": 650, "platform_fee": 165, "rating": 3.9,
                    "listing_score": 44, "status": "Active", "stock": 42,
                    "lead_time_days": 3, "sales_last_month": 95, "sales_this_month": 51,
                },
                {
                    "id": "SKU-P03", "name": "Bridal Lehenga Choli Set",
                    "category": "Lehengas", "price": 2499, "competitor_avg": 2199,
                    "unit_cost": 1300, "platform_fee": 320, "rating": 4.5,
                    "listing_score": 71, "status": "Out of Stock", "stock": 0,
                    "lead_time_days": 10, "sales_last_month": 42, "sales_this_month": 5,
                },
                {
                    "id": "SKU-P04", "name": "Rayon Palazzo Pants",
                    "category": "Bottomwear", "price": 449, "competitor_avg": 429,
                    "unit_cost": 180, "platform_fee": 60, "rating": 4.2,
                    "listing_score": 76, "status": "Active", "stock": 310,
                    "lead_time_days": 3, "sales_last_month": 220, "sales_this_month": 198,
                },
                {
                    "id": "SKU-P05", "name": "Printed Cotton Casual Top",
                    "category": "Tops", "price": 299, "competitor_avg": 289,
                    "unit_cost": 110, "platform_fee": 42, "rating": 4.1,
                    "listing_score": 69, "status": "Active", "stock": 520,
                    "lead_time_days": 2, "sales_last_month": 103, "sales_this_month": 114,
                },
            ],
            "cancellation_breakdown": {
                "total_this_month": 35,
                "reasons": [
                    {"reason": "Out of stock post-order", "pct": 38, "count": 13},
                    {"reason": "Size not available / mismatch", "pct": 32, "count": 11},
                    {"reason": "Delivery delay expected", "pct": 20, "count": 7},
                    {"reason": "Buyer changed mind", "pct": 10, "count": 4},
                ],
            },
            "market_context": {
                "category_growth_this_month": 0.22,
                "seller_growth_this_month": -0.344,
                "platform_avg_cancel_rate": 0.06,
                "active_competitors_added": 28,
                "festive_window": "Active — Wedding Season & Diwali Gifting",
                "seller_enrolled_festive": False,
                "seller_enrolled_plus": True,
                "oos_ads_paused": False,
            },
            "causal_hypotheses": [
                {
                    "id": "H1", "priority": "P0 - Critical",
                    "title": "Wedding Season Inventory Crisis",
                    "observation": "Kurti Set & Bridal Lehenga OOS during peak wedding season (Nov); Women's Fashion grew +22% but Priya's revenue fell 34.4%. Direct OOS revenue loss: ₹99,610.",
                    "mechanism": "Restoring Kurti Set (180→100 unit buffer) and Lehenga (42→30 unit buffer) captures the wedding/gifting surge.",
                    "action": "Emergency restock Kurti Set & Lehenga from Mumbai textile suppliers; pause ads on 8 OOS SKUs immediately.",
                    "metric": "OOS cancellations (38%)→<10%; weekly revenue +₹24,000.",
                    "stop_condition": "If supplier lead time >7 days, source from Surat wholesale market.",
                    "projected_recovery": "₹99,610/mo",
                },
                {
                    "id": "H2", "priority": "P0 - Critical",
                    "title": "Size Chart & Listing Quality Improvement",
                    "observation": "32% of cancellations (11 orders) are size mismatches. Embroidered Saree listing score only 44 — missing size guide, fit chart, and care instructions.",
                    "mechanism": "Adding size charts and 'fits true to size' photos reduces size-related returns from 12% to <5%.",
                    "action": "Upload detailed size charts, measurement guide, and 5+ product images for Saree and Kurti listings.",
                    "metric": "Return rate 12%→<5%; listing score >75.",
                    "stop_condition": "Track 2-week return rate after update; revert to previous images if returns rise.",
                    "projected_recovery": "₹22,400/mo",
                },
                {
                    "id": "H3", "priority": "P1 - High",
                    "title": "Wedding Season Festive Deal Enrollment",
                    "observation": "Women's Fashion grew +22% in November (wedding + Diwali gifting). Priya not enrolled in festive deals despite Gold Seller status.",
                    "mechanism": "Enrolling in-stock Palazzo Pants and Cotton Tops in Festive Deals captures +40% organic impressions during peak season.",
                    "action": "Enroll Palazzo Pants, Cotton Tops, and active Saree listings in Snapdeal November Festive Deals today.",
                    "metric": "Organic impressions +40%; conversion 2.1%→3%.",
                    "stop_condition": "Restrict enrollment to SKUs with >50 units stock.",
                    "projected_recovery": "₹31,000/mo",
                },
            ],
            "playbook": [
                {
                    "id": "pb_wedding_season_oos",
                    "title": "Wedding Season Out-of-Stock Recovery",
                    "tags": ["inventory", "oos", "wedding", "season", "kurti", "lehenga", "restock", "supplier"],
                    "applies_when": "oos_products >= 5 and category == Women's Fashion",
                    "text": (
                        "Emergency restock Kurti Set and Bridal Lehenga within 48 hours from Mumbai textile suppliers or Surat wholesale market. "
                        "Pause ads on all 8 OOS listings immediately to prevent post-order cancellations. "
                        "Maintain 100-unit buffer for Kurti Set and 30-unit buffer for Lehenga during peak wedding season (Nov-Feb)."
                    ),
                },
                {
                    "id": "pb_size_return_reduction",
                    "title": "Size Mismatch & Return Rate Reduction",
                    "tags": ["returns", "size", "mismatch", "chart", "listing", "quality", "photos", "fit"],
                    "applies_when": "return_rate >= 0.10 or cancellation_reason == size mismatch",
                    "text": (
                        "32% of cancellations are size-related. Upload detailed size charts with measurements in cm and inches, "
                        "model height/weight for reference photos, and care washing instructions. "
                        "Adding 'fits true to size' verified badges reduces return rate from 12% to below 5%."
                    ),
                },
                {
                    "id": "pb_listing_quality_fashion",
                    "title": "Fashion Listing Quality & Search Visibility",
                    "tags": ["listing", "score", "images", "saree", "keywords", "seo", "visibility", "photos"],
                    "applies_when": "listing_score < 70",
                    "text": (
                        "Embroidered Saree listing score is only 44 — well below the 75+ threshold for search boosting. "
                        "Upload 6+ lifestyle images, add colour/fabric/occasion keywords in bullet points, and complete the size guide. "
                        "Listings above 75 score get 2x more organic search impressions on Snapdeal."
                    ),
                },
                {
                    "id": "pb_festive_fashion_deals",
                    "title": "Wedding & Festive Season Deal Campaign",
                    "tags": ["festive", "wedding", "deals", "promotion", "traffic", "season", "diwali"],
                    "applies_when": "not seller_enrolled_festive and category_growth >= 0.15",
                    "text": (
                        "Women's Fashion grew 22% in November. Enroll Palazzo Pants, Cotton Tops, and active Saree listings "
                        "in Snapdeal Festive Deals to capture +40% organic impressions. "
                        "Gold Seller status guarantees banner placement in category pages during the festive window."
                    ),
                },
                {
                    "id": "pb_pricing_fashion",
                    "title": "Fashion Pricing & Competitor Benchmarking",
                    "tags": ["pricing", "price", "competitor", "margin", "conversion", "expensive"],
                    "applies_when": "price > competitor_avg * 1.08",
                    "text": (
                        "Kurti Set priced ₹749 vs market ₹649 (15% premium) during peak demand. "
                        "Matching market price at ₹649 retains ₹234 net profit (36% margin) while rebounding conversion. "
                        "Embroidered Saree at ₹1,299 vs ₹1,199 market — dropping to ₹1,199 retains ₹384 net margin (32%)."
                    ),
                },
            ],
        }

    elif seller_id == "farouk":
        return {
            "seller_profile": {
                "id": "SD-33127",
                "name": "Mohammed Farouk",
                "store": "Farouk Electronics Hub",
                "category": "Home Electronics",
                "city": "Delhi",
                "tier": "Gold Seller",
                "rating": 3.8,
                "reviews": 3410,
                "total_catalog_skus": 22,
                "joined_months_ago": 36,
                "account_health_flag": "Warning: Return rate spike",
            },
            "performance": {
                "last_month": {
                    "label": "October 2026",
                    "revenue": 580000,
                    "orders": 290,
                    "avg_selling_price": 2000,
                    "cancellation_rate": 0.07,
                    "oos_products": 1,
                    "return_rate": 0.04,
                    "conversion_rate": 0.019,
                },
                "this_month": {
                    "label": "November 2026",
                    "revenue": 420000,
                    "orders": 192,
                    "avg_selling_price": 2188,
                    "cancellation_rate": 0.10,
                    "oos_products": 3,
                    "return_rate": 0.09,
                    "conversion_rate": 0.013,
                },
            },
            "products": [
                {
                    "id": "SKU-F01", "name": "32-inch Full HD LED Smart TV",
                    "category": "Televisions", "price": 12999, "competitor_avg": 11499,
                    "unit_cost": 8500, "platform_fee": 1300, "rating": 3.5,
                    "listing_score": 61, "status": "Active", "stock": 18,
                    "lead_time_days": 5, "sales_last_month": 44, "sales_this_month": 22,
                    "quality_issues": "High return rate: 'remote lag', 'WiFi drops after 2 hours', 'remote unresponsive'",
                },
                {
                    "id": "SKU-F02", "name": "Portable Bluetooth Speaker 20W",
                    "category": "Audio", "price": 1799, "competitor_avg": 1699,
                    "unit_cost": 980, "platform_fee": 230, "rating": 4.4,
                    "listing_score": 83, "status": "Active", "stock": 125,
                    "lead_time_days": 2, "sales_last_month": 98, "sales_this_month": 87,
                },
                {
                    "id": "SKU-F03", "name": "5.5L Digital Air Fryer with Timer",
                    "category": "Kitchen Appliances", "price": 3499, "competitor_avg": 2999,
                    "unit_cost": 1900, "platform_fee": 440, "rating": 4.6,
                    "listing_score": 88, "status": "Out of Stock", "stock": 0,
                    "lead_time_days": 8, "sales_last_month": 62, "sales_this_month": 11,
                },
                {
                    "id": "SKU-F04", "name": "Robot Vacuum Cleaner with Mapping",
                    "category": "Cleaning Appliances", "price": 8999, "competitor_avg": 7999,
                    "unit_cost": 5500, "platform_fee": 1100, "rating": 4.3,
                    "listing_score": 79, "status": "Out of Stock", "stock": 0,
                    "lead_time_days": 12, "sales_last_month": 28, "sales_this_month": 3,
                },
                {
                    "id": "SKU-F05", "name": "Mixer Grinder 750W 3-Jar",
                    "category": "Kitchen Appliances", "price": 2199, "competitor_avg": 2099,
                    "unit_cost": 1200, "platform_fee": 280, "rating": 4.2,
                    "listing_score": 74, "status": "Active", "stock": 67,
                    "lead_time_days": 3, "sales_last_month": 58, "sales_this_month": 69,
                },
            ],
            "cancellation_breakdown": {
                "total_this_month": 19,
                "reasons": [
                    {"reason": "Delivery delay expected (>7 days)", "pct": 43, "count": 8},
                    {"reason": "Out of stock post-order", "pct": 32, "count": 6},
                    {"reason": "Price issue / found cheaper", "pct": 15, "count": 3},
                    {"reason": "Buyer changed mind", "pct": 10, "count": 2},
                ],
            },
            "market_context": {
                "category_growth_this_month": 0.15,
                "seller_growth_this_month": -0.276,
                "platform_avg_cancel_rate": 0.06,
                "active_competitors_added": 7,
                "festive_window": "Active — Big Diwali Sale & Winter Gifting",
                "seller_enrolled_festive": False,
                "seller_enrolled_plus": True,
                "oos_ads_paused": False,
                "big_diwali_sale_missed": True,
                "avg_delivery_days": 8.3,
            },
            "causal_hypotheses": [
                {
                    "id": "H1", "priority": "P0 - Critical",
                    "title": "LED TV Return Rate Crisis — Quality Issue",
                    "observation": "LED TV return rate spiked to 18% this month (4x platform average). Top review complaints: 'remote lag', 'WiFi drops after 2 hours'. 43% of returns are from this single SKU. Rating dropped from 4.0 to 3.5.",
                    "mechanism": "Pausing LED TV ads stops new defective units shipping and prevents further rating damage. Engaging the supplier for a firmware fix or batch replacement recovers the SKU.",
                    "action": "Pause LED TV ads immediately; contact supplier for firmware update or replacement batch; add quality disclaimer to listing.",
                    "metric": "Return rate 18%→<4%; TV rating 3.5→4.0 within 30 days.",
                    "stop_condition": "If supplier cannot provide firmware fix within 14 days, delist the SKU.",
                    "projected_recovery": "₹55,000/mo from rating recovery",
                },
                {
                    "id": "H2", "priority": "P0 - Critical",
                    "title": "Delivery Speed — 43% Cancellations from Long Lead Time",
                    "observation": "Average delivery time 8.3 days (platform benchmark: ≤4 days). 43% of cancellations (8 orders) are due to buyers seeing long delivery estimates. Air Fryer and Robot Vacuum have 8-12 day lead times.",
                    "mechanism": "Switching to Snapdeal Fulfilment Centre (SFC) for high-velocity SKUs reduces delivery to 2-3 days, cutting delay-related cancellations by 80%.",
                    "action": "Enroll Bluetooth Speaker, Air Fryer, and Mixer Grinder in Snapdeal SFC; reduce Robot Vacuum lead time by pre-positioning units in Delhi warehouse.",
                    "metric": "Delivery-cancellation rate 43%→<10%; conversion 1.3%→1.9%.",
                    "stop_condition": "SFC onboarding takes 5 business days; manage inventory buffer during transition.",
                    "projected_recovery": "₹68,000/mo",
                },
                {
                    "id": "H3", "priority": "P1 - High",
                    "title": "Big Diwali Sale Missed — Festive Recovery Plan",
                    "observation": "Home Electronics grew +15% in November. Farouk missed the Big Diwali Sale enrollment window. Air Fryer and Robot Vacuum are OOS during peak gifting season.",
                    "mechanism": "Enrolling in Winter Gifting Sale and restocking Air Fryer (demand: 62/month) captures post-Diwali gifting surge.",
                    "action": "Restock Air Fryer within 72 hours; enroll Mixer Grinder and Bluetooth Speaker in Snapdeal Winter Gifting Sale.",
                    "metric": "Organic impressions +35%; monthly revenue +₹45,000.",
                    "stop_condition": "Restrict enrollment to SKUs with delivery SLA ≤4 days.",
                    "projected_recovery": "₹45,000/mo",
                },
            ],
            "playbook": [
                {
                    "id": "pb_tv_quality_crisis",
                    "title": "Electronics Return Rate & Quality Issue Resolution",
                    "tags": ["returns", "quality", "defect", "tv", "rating", "reviews", "firmware", "complaint"],
                    "applies_when": "return_rate >= 0.08 or rating < 3.8",
                    "text": (
                        "LED TV return rate at 18% (4x platform average). Immediately pause sponsored ads on the TV listing. "
                        "Contact supplier for firmware update addressing 'remote lag' and 'WiFi instability'. "
                        "Add quality assurance note to listing. If firmware fix not available in 14 days, initiate SKU delisting to protect overall account health."
                    ),
                },
                {
                    "id": "pb_delivery_speed",
                    "title": "Delivery Speed Improvement & SFC Enrollment",
                    "tags": ["delivery", "shipping", "sfc", "fulfilment", "fast", "cancellation", "delay", "warehouse"],
                    "applies_when": "avg_delivery_days > 5 or cancellation_reason == delivery delay",
                    "text": (
                        "43% of cancellations are from delivery delays (avg 8.3 days vs 4-day benchmark). "
                        "Enroll Bluetooth Speaker and Mixer Grinder in Snapdeal Fulfilment Centre (SFC) for 2-3 day delivery. "
                        "Pre-position Robot Vacuum units in Delhi SFC warehouse to reduce lead time from 12 to 3 days."
                    ),
                },
                {
                    "id": "pb_oos_electronics",
                    "title": "Electronics OOS Recovery — Air Fryer & Robot Vacuum",
                    "tags": ["oos", "out of stock", "air fryer", "robot vacuum", "restock", "gifting"],
                    "applies_when": "oos_products >= 2 and category == Home Electronics",
                    "text": (
                        "Air Fryer (62 units/month demand) and Robot Vacuum (28 units/month) are OOS during peak Diwali gifting season. "
                        "Emergency restock Air Fryer within 72 hours from supplier. "
                        "Pause ads on both OOS listings immediately to avoid cancellations and further account health penalties."
                    ),
                },
                {
                    "id": "pb_festive_electronics",
                    "title": "Post-Diwali Winter Gifting Campaign",
                    "tags": ["festive", "diwali", "winter", "gifting", "deals", "campaign", "electronics", "promotion"],
                    "applies_when": "not seller_enrolled_festive and category_growth >= 0.10",
                    "text": (
                        "Home Electronics grew 15% in November. Despite missing Big Diwali Sale, Winter Gifting campaign runs Nov 20 - Dec 15. "
                        "Enroll Mixer Grinder and Bluetooth Speaker (both in-stock, good ratings) in the Winter Gifting Sale. "
                        "Gold Seller status grants free banner placement in Electronics category pages."
                    ),
                },
                {
                    "id": "pb_pricing_electronics",
                    "title": "Electronics Pricing vs Market Benchmark",
                    "tags": ["pricing", "price", "competitor", "margin", "tv", "expensive", "conversion"],
                    "applies_when": "price > competitor_avg * 1.08",
                    "text": (
                        "LED TV priced ₹12,999 vs market ₹11,499 (13% premium) while having quality issues — double handicap. "
                        "Pause TV ads rather than price-cut during quality investigation. "
                        "Air Fryer at ₹3,499 vs ₹2,999 market — once restocked, price at ₹3,199 retaining 37% net margin."
                    ),
                },
            ],
        }

    raise ValueError(f"Unknown seller_id: {seller_id!r}")


def tokenize_with_ngrams(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    ngrams = [w[i : i + 4] for w in words if len(w) >= 4 for i in range(len(w) - 3)]
    return words + ngrams


def build_tfidf_embeddings(docs: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    tokenized = [tokenize_with_ngrams(d) for d in docs]
    vocab = sorted({tok for doc in tokenized for tok in doc})
    vocab_idx = {tok: i for i, tok in enumerate(vocab)}
    n_docs = len(docs)

    df = np.zeros(len(vocab), dtype=np.float32)
    tf = np.zeros((n_docs, len(vocab)), dtype=np.float32)
    for d_i, tokens in enumerate(tokenized):
        for tok in set(tokens):
            df[vocab_idx[tok]] += 1.0
        for tok in tokens:
            tf[d_i, vocab_idx[tok]] += 1.0

    idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
    tfidf = tf * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True) + 1e-9
    tfidf_normed = tfidf / norms
    return tfidf_normed, vocab, idf


def try_fetch_ollama_embeddings(docs: list[str]) -> np.ndarray | None:
    try:
        vecs = []
        with httpx.Client(timeout=3.0) as client:
            for d in docs:
                r = client.post(f"{OLLAMA_BASE_URL}/api/embeddings", json={"model": EMBED_MODEL, "prompt": d})
                r.raise_for_status()
                v = np.array(r.json()["embedding"], dtype=np.float32)
                v = v / (np.linalg.norm(v) + 1e-9)
                vecs.append(v)
        return np.vstack(vecs)
    except Exception as e:
        print(f"[generate_data] Ollama '{EMBED_MODEL}' offline ({e}); saving sub-word TF-IDF vectors.")
        return None


def build_knowledge_base(seller_id: str = DEFAULT_SELLER_ID):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kb = get_raw_knowledge_base(seller_id)
    out_kb = kb_path(seller_id)
    out_embed = embed_path(seller_id)

    with open(out_kb, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    print(f"✅ [{seller_id}] Saved KB to {out_kb}")

    playbook = kb["playbook"]
    docs = [f"{p['title']} {' '.join(p['tags'])} {p['applies_when']} {p['text']}" for p in playbook]
    tfidf_matrix, vocab, idf = build_tfidf_embeddings(docs)
    dense_matrix = try_fetch_ollama_embeddings(docs)

    save_kwargs = {
        "tfidf_matrix": tfidf_matrix,
        "vocab": np.array(vocab),
        "idf": idf,
        "has_dense": np.array([dense_matrix is not None]),
    }
    if dense_matrix is not None:
        save_kwargs["dense_matrix"] = dense_matrix
        print(f"✅ [{seller_id}] Stored {dense_matrix.shape} dense + TF-IDF embeddings")
    else:
        print(f"✅ [{seller_id}] Stored {tfidf_matrix.shape} TF-IDF embeddings")

    np.savez_compressed(out_embed, **save_kwargs)
    print(f"✅ [{seller_id}] Saved embeddings to {out_embed}")


if __name__ == "__main__":
    for sid in SELLER_IDS:
        build_knowledge_base(sid)
    print("\n✅ All seller knowledge bases and embeddings generated.")
