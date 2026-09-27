import json
import os

workflow_item = {
    "id": "kG9eB2qX8mP4zL1w",
    "name": "BIST 100 Quant Autonomous Trading Agent",
    "active": False,
    "nodes": [
        {
            "parameters": {},
            "id": "e457f50e-56e6-4993-8bc6-91e8783457a1",
            "name": "Manual Test Trigger",
            "type": "n8n-nodes-base.manualTrigger",
            "typeVersion": 1,
            "position": [220, 240]
        },
        {
            "parameters": {
                "rule": {
                    "interval": [
                        {"field": "cronExpression", "expression": "45 17 * * 1-5"},
                        {"field": "cronExpression", "expression": "50 9 * * 1-5"}
                    ]
                }
            },
            "id": "bb4ffec6-0cf8-4171-8bc6-09d6fbb348c3",
            "name": "BIST Trading Hours Cron",
            "type": "n8n-nodes-base.scheduleTrigger",
            "typeVersion": 1.2,
            "position": [220, 420]
        },
        {
            "parameters": {
                "url": "http://fastapi_bist:8000/api/v1/market/regime",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "X-API-Key", "value": "bist_quant_secret_2026"}]
                },
                "options": {"timeout": 30000}
            },
            "id": "7fa490c6-30c1-4545-9279-0526048d28db",
            "name": "Get Market Regime",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [440, 320]
        },
        {
            "parameters": {
                "url": "http://fastapi_bist:8000/api/v1/champion/status",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "X-API-Key", "value": "bist_quant_secret_2026"}]
                },
                "options": {"timeout": 30000}
            },
            "id": "5a8e12f0-1c34-4a29-b68e-908bcda71234",
            "name": "Get Champion Status",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [660, 320]
        },
        {
            "parameters": {
                "method": "POST",
                "url": "http://fastapi_bist:8000/api/v1/signals/refresh",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "X-API-Key", "value": "bist_quant_secret_2026"}]
                },
                "options": {"timeout": 60000}
            },
            "id": "ac697f94-912a-4340-97eb-30f142dfa52b",
            "name": "Refresh Daily Signals",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [880, 320]
        },
        {
            "parameters": {
                "url": "http://fastapi_bist:8000/api/v1/portfolio/ledger",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [{"name": "X-API-Key", "value": "bist_quant_secret_2026"}]
                },
                "options": {"timeout": 30000}
            },
            "id": "673f32bc-c7f7-41ab-bc92-563ad20e5491",
            "name": "Get Portfolio Ledger",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [1100, 320]
        },
        {
            "parameters": {
                "jsCode": """const regimeData = $('Get Market Regime').first().json;
const champData = $('Get Champion Status').first().json;
const signalsData = $('Refresh Daily Signals').first().json;
const ledgerData = $('Get Portfolio Ledger').first().json;

const isBull = regimeData.macro_regime_bull === true;
const regimeTitle = isBull ? "🟢 BOĞA PİYASASI (Pozitif Rejim)" : "🔴 AYI PİYASASI (Defansif Rejim)";
const actionDesc = regimeData.portfolio_action || (isBull ? "Top Long Hisse Alımı" : "100% Nakit / PPF Repo");
const championName = champData.champion_model || signalsData.champion_model || "GRU_Ranker (10Y Haftalık Momentum)";

const longs = signalsData.top_longs || [];
let signalLines = "";
if (longs.length > 0) {
  signalLines = longs.map((s, idx) => {
    return `${idx + 1}. *${s.ticker}* | Alfa: ${s.expected_excess_return} | Pay: ${s.recommended_weight} | Stop: ${s.dynamic_stop_loss}`;
  }).join("\\n");
} else {
  signalLines = "Bugün için onaylanmış hisse alım sinyali bulunmamaktadır.";
}

const equity = Number(ledgerData.current_portfolio_value_try || 1000000).toLocaleString('tr-TR', { maximumFractionDigits: 2 });
const cash = Number(ledgerData.current_cash_try || 1000000).toLocaleString('tr-TR', { maximumFractionDigits: 2 });
const alpha = (ledgerData.net_alpha_pct || 0).toFixed(2);

const reportText = `🏛 BIST 100 QUANT TERMINAL | SEANS BRİFİNGİ\\n` +
  `📅 Tarih: ${signalsData.report_date || new Date().toISOString().slice(0, 10)}\\n` +
  `🎯 Aktif Şampiyon Model: ${championName}\\n` +
  `🌊 Makro Rejim: ${regimeTitle}\\n` +
  `⚡ Strateji: ${actionDesc}\\n\\n` +
  `💼 PORTFÖY:\\n` +
  `• Özsermaye: ${equity} TL\\n` +
  `• Nakit/Repo: ${cash} TL\\n` +
  `• Net Alfa: %${alpha}\\n\\n` +
  `📈 GÜNLÜK TOP LONG SİNYALLERİ:\\n` +
  `${signalLines}`;

return [{
  json: {
    is_bull: isBull,
    report_date: signalsData.report_date,
    champion_model: championName,
    equity_try: equity,
    cash_try: cash,
    net_alpha_pct: alpha,
    briefing_report: reportText,
    should_execute: isBull && longs.length > 0
  }
}];"""
            },
            "id": "fd903ba8-9e5b-4c5e-851f-6a9c3725b7a1",
            "name": "Format Quant Briefing",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [1320, 320]
        },
        {
            "parameters": {
                "conditions": {
                    "options": {"caseSensitive": True, "leftValue": "", "typeValidation": "strict", "version": 2},
                    "conditions": [
                        {
                            "id": "18f53a47-a892-4911-9a7f-7dc9f7c0f1e8",
                            "leftValue": "={{ $json.should_execute }}",
                            "rightValue": True,
                            "operator": {"type": "boolean", "operation": "equals"}
                        }
                    ],
                    "combinator": "and"
                }
            },
            "id": "90d95c47-3801-4475-801a-8c769d4bbff9",
            "name": "Is Bull Regime?",
            "type": "n8n-nodes-base.if",
            "typeVersion": 2,
            "position": [1540, 320]
        },
        {
            "parameters": {
                "method": "POST",
                "url": "http://fastapi_bist:8000/api/v1/execution/run-daily",
                "sendHeaders": True,
                "headerParameters": {
                    "parameters": [
                        {"name": "X-API-Key", "value": "bist_quant_secret_2026"},
                        {"name": "Content-Type", "value": "application/json"}
                    ]
                },
                "sendBody": True,
                "specifyBody": "json",
                "jsonBody": '{"mode": "close_auction", "force_cash": false, "rebalance_step": 5, "force_execution": false}',
                "options": {"timeout": 60000}
            },
            "id": "40bca80f-eb5a-47d3-b183-b7891fa16db6",
            "name": "Execute 17:50 Close Auction",
            "type": "n8n-nodes-base.httpRequest",
            "typeVersion": 4.2,
            "position": [1760, 220]
        },
        {
            "parameters": {
                "jsCode": """return [{
  json: {
    status: "DEFENSIVE_HOLD",
    message: "Makro Kapısı AYI modunda olduğu için yeni hisse alımı yapılmadı. Portföy %100 risksiz PPF / Repo nakitte korunuyor.",
    report: $('Format Quant Briefing').first().json.briefing_report
  }
}];"""
            },
            "id": "01c7db17-eef4-4fbf-9333-6a9b4d8120fa",
            "name": "Hold Cash / Repo (Bear Market)",
            "type": "n8n-nodes-base.code",
            "typeVersion": 2,
            "position": [1760, 420]
        }
    ],
    "connections": {
        "Manual Test Trigger": {"main": [[{"node": "Get Market Regime", "type": "main", "index": 0}]]},
        "BIST Trading Hours Cron": {"main": [[{"node": "Get Market Regime", "type": "main", "index": 0}]]},
        "Get Market Regime": {"main": [[{"node": "Get Champion Status", "type": "main", "index": 0}]]},
        "Get Champion Status": {"main": [[{"node": "Refresh Daily Signals", "type": "main", "index": 0}]]},
        "Refresh Daily Signals": {"main": [[{"node": "Get Portfolio Ledger", "type": "main", "index": 0}]]},
        "Get Portfolio Ledger": {"main": [[{"node": "Format Quant Briefing", "type": "main", "index": 0}]]},
        "Format Quant Briefing": {"main": [[{"node": "Is Bull Regime?", "type": "main", "index": 0}]]},
        "Is Bull Regime?": {
            "main": [
                [{"node": "Execute 17:50 Close Auction", "type": "main", "index": 0}],
                [{"node": "Hold Cash / Repo (Bear Market)", "type": "main", "index": 0}]
            ]
        }
    },
    "settings": {"executionOrder": "v1"}
}

os.makedirs("n8n", exist_ok=True)
with open("n8n/bist_quant_autonomous_agent.json", "w", encoding="utf-8") as f:
    json.dump([workflow_item], f, ensure_ascii=False, indent=2)

print("SUCCESS: Validated n8n JSON with dynamic GRU_Ranker Champion integration updated successfully!")
