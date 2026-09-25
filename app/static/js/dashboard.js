// DLAI_BIST Quant Trading Terminal Frontend Script

let equityChartInstance = null;
let allocationChartInstance = null;

function getAuthHeaders(extra = {}) {
  const headers = { 'Content-Type': 'application/json', ...extra };
  if (window.DLAI_API_KEY) {
    headers['X-API-Key'] = window.DLAI_API_KEY;
  }
  return headers;
}

// Clock updates
function updateClock() {
  const now = new Date();
  const timeStr = now.toLocaleTimeString('tr-TR', { hour12: false });
  const dateStr = now.toLocaleDateString('tr-TR', { year: 'numeric', month: 'short', day: 'numeric' });
  const clockEl = document.getElementById('live-clock');
  if (clockEl) {
    clockEl.innerHTML = `<span class="text-slate-400 mr-2">${dateStr}</span> <span class="font-mono text-cyan-400 font-bold">${timeStr}</span>`;
  }
}
setInterval(updateClock, 1000);
updateClock();

// Toast helper
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;
  
  const toast = document.createElement('div');
  const bg = type === 'success' ? 'bg-emerald-950 border-emerald-500 text-emerald-300' :
             type === 'error' ? 'bg-rose-950 border-rose-500 text-rose-300' :
             'bg-slate-900 border-cyan-500 text-cyan-300';
             
  toast.className = `border px-4 py-3 rounded-lg shadow-xl text-sm flex items-center gap-2 transition-all duration-300 transform translate-y-2 opacity-0 ${bg}`;
  toast.innerHTML = `
    <span class="font-bold">${type === 'success' ? '✓' : type === 'error' ? '⚠' : 'ℹ'}</span>
    <span>${message}</span>
  `;
  
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.remove('translate-y-2', 'opacity-0');
  }, 50);
  
  setTimeout(() => {
    toast.classList.add('opacity-0', '-translate-y-2');
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// Fetch all dashboard data
async function loadDashboardData(isManual = false) {
  const icon = document.getElementById('refresh-icon');
  if (icon) icon.classList.add('animate-spin');

  try {
    const t = Date.now();
    const [regimeRes, championRes, signalsRes, ledgerRes, sentimentRes] = await Promise.all([
      fetch(`/api/v1/market/regime?_t=${t}`).then(r => r.json()).catch(() => null),
      fetch(`/api/v1/champion/status?_t=${t}`).then(r => r.json()).catch(() => null),
      fetch(`/api/v1/signals/daily?_t=${t}`).then(r => r.json()).catch(() => null),
      fetch(`/api/v1/portfolio/ledger?_t=${t}`).then(r => r.json()).catch(() => null),
      fetch(`/api/v1/sentiment/kap?limit=25&_t=${t}`).then(r => r.json()).catch(() => null)
    ]);

    if (regimeRes) renderMacroRegime(regimeRes);
    if (championRes) renderChampionStatus(championRes);
    if (signalsRes) renderSignals(signalsRes);
    if (ledgerRes) renderPortfolio(ledgerRes, signalsRes);
    if (sentimentRes) renderSentiment(sentimentRes);

  } catch (err) {
    console.error("Dashboard yüklenirken hata:", err);
    if (isManual) {
      showToast("Veriler yüklenirken bağlantı hatası oluştu.", "error");
    }
  } finally {
    if (icon) icon.classList.remove('animate-spin');
  }
}

// Manual Refresh with prominent UI feedback
async function manualRefreshData() {
  const btn = document.getElementById('btn-refresh');
  const icon = document.getElementById('refresh-icon');
  const text = document.getElementById('refresh-text');
  
  if (btn) btn.disabled = true;
  if (icon) icon.classList.add('animate-spin');
  if (text) text.innerText = 'Yenileniyor...';

  try {
    const minDelay = new Promise(resolve => setTimeout(resolve, 600));
    await Promise.all([loadDashboardData(true), minDelay]);
    
    showToast("Tüm kokpit verileri ve piyasa sinyalleri güncellendi!", "success");
    
    // Highlight KPI cards
    document.querySelectorAll('.fin-card').forEach(card => {
      card.classList.add('border-cyan-500/50');
      setTimeout(() => card.classList.remove('border-cyan-500/50'), 600);
    });
  } catch (err) {
    showToast("Veriler yenilenirken hata oluştu: " + err.message, "error");
  } finally {
    if (icon) icon.classList.remove('animate-spin');
    if (text) text.innerText = 'Verileri Yenile';
    if (btn) btn.disabled = false;
  }
}

// Recalculate Signals (re-checks today's signals & KAP veto)
async function recalculateSignals() {
  const btn = document.getElementById('btn-recalc-signals');
  const icon = document.getElementById('recalc-icon');
  
  if (btn) btn.disabled = true;
  if (icon) icon.classList.add('animate-spin');

  showToast("Şampiyon model ve KAP gatekeeper sinyalleri hesaplanıyor...", "info");

  try {
    const res = await fetch('/api/v1/signals/refresh', {
      method: 'POST',
      headers: getAuthHeaders()
    });
    
    if (res.ok) {
      await loadDashboardData(true);
      showToast("Sinyaller ve model kararları güncellendi!", "success");
    } else {
      showToast("Sinyal güncelleme hatası.", "error");
    }
  } catch (err) {
    showToast("Sinyal servisine ulaşılamadı.", "error");
  } finally {
    if (icon) icon.classList.remove('animate-spin');
    if (btn) btn.disabled = false;
  }
}

// Reset Portfolio Equity to 1 Million TRY
async function resetPortfolioEquity() {
  const confirmed = confirm("DİKKAT: Sanal portföy özsermayesini 1.000.000,00 TL nakde sıfırlamak istediğinize emin misiniz?\n\n(Mevcut defter yedeklenecek ve tüm getiri/kayıp geçmişi sıfırlanacaktır.)");
  if (!confirmed) return;

  const btn = document.getElementById('btn-reset-ledger');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<span class="animate-spin inline-block mr-1">↻</span> Sıfırlanıyor...`;
  }

  showToast("Özsermaye 1.000.000 TL'ye sıfırlanıyor...", "info");

  try {
    const res = await fetch('/api/v1/portfolio/reset?capital=1000000', {
      method: 'POST',
      headers: getAuthHeaders()
    });

    if (res.ok) {
      await loadDashboardData(true);
      showToast("Portföy özsermayesi 1.000.000 TL'ye başarıyla sıfırlandı!", "success");
    } else {
      const err = await res.json();
      showToast(`Sıfırlama hatası: ${err.detail || 'Bilinmeyen hata'}`, "error");
    }
  } catch (e) {
    showToast("Sunucuya ulaşılamadı.", "error");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
        <span>Sermayeyi 1M'a Sıfırla</span>
      `;
    }
  }
}

// 1. Macro Regime Render
function renderMacroRegime(regime) {
  const badgeEl = document.getElementById('regime-badge');
  const descEl = document.getElementById('regime-desc');
  
  if (!badgeEl) return;
  if (regime.macro_regime_bull) {
    badgeEl.className = "px-3 py-1 rounded-full text-xs font-semibold badge-bull flex items-center gap-1.5";
    badgeEl.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 live-dot"></span> BOĞA PİYASASI (Hisse Ağırlıklı)`;
  } else {
    badgeEl.className = "px-3 py-1 rounded-full text-xs font-semibold badge-cash flex items-center gap-1.5";
    badgeEl.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400 live-dot"></span> DEFANSİF REJİM (100% PPF Repo Nakit)`;
  }
  
  if (descEl) descEl.innerText = regime.description;
}

// 2. Champion Status Render
function renderChampionStatus(champ) {
  const modelNameEl = document.getElementById('champ-model-name');
  const modelStatsEl = document.getElementById('champ-model-stats');
  if (modelNameEl) modelNameEl.innerText = champ.champion_model || "GRU_Ranker";
  if (modelStatsEl) {
    const alpha = (champ.net_alpha_pct !== undefined) ? champ.net_alpha_pct : 0.0;
    const sharpe = (champ.sharpe_ratio !== undefined) ? champ.sharpe_ratio : 0.0;
    modelStatsEl.innerText = `Net Alfa: %${alpha} | Sharpe: ${sharpe}`;
  }
}

// 3. Signals Render
function renderSignals(data) {
  const tableBody = document.getElementById('signals-table-body');
  const actionSummaryEl = document.getElementById('signal-action-summary');
  
  if (actionSummaryEl) {
    actionSummaryEl.innerText = `${data.portfolio_action} | Rapor Tarihi: ${data.report_date}`;
  }

  if (!tableBody) return;
  tableBody.innerHTML = '';

  const longs = data.top_longs || [];
  if (longs.length === 0) {
    tableBody.innerHTML = `<tr><td colspan="7" class="text-center py-6 text-slate-500">Bugün için onaylanmış hisse sinyali bulunmamaktadır.</td></tr>`;
    return;
  }

  longs.forEach((item, idx) => {
    const tr = document.createElement('tr');
    tr.className = "border-b border-slate-800/60 hover:bg-slate-800/30 transition font-mono text-xs";

    const weight = item.recommended_weight || "%20.0";
    const stopLoss = item.dynamic_stop_loss || "-%5.0";
    const tier = item.liquidity_tier || "BIST 30";

    tr.innerHTML = `
      <td class="py-3 px-3 font-bold text-white flex items-center gap-2">
        <span class="w-5 h-5 rounded bg-slate-800 text-slate-300 flex items-center justify-center text-[10px] font-mono">${idx + 1}</span>
        <span class="text-emerald-400 font-bold">${item.ticker}</span>
      </td>
      <td class="py-3 px-3 text-cyan-300">${item.expected_excess_return}</td>
      <td class="py-3 px-3 text-slate-300">${item.confidence}</td>
      <td class="py-3 px-3 font-bold text-emerald-300">${weight}</td>
      <td class="py-3 px-3 text-rose-400 font-semibold">${stopLoss}</td>
      <td class="py-3 px-3 text-slate-400 text-[11px]">${tier}</td>
      <td class="py-3 px-3">
        <span class="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
          ${item.action}
        </span>
      </td>
    `;
    tableBody.appendChild(tr);
  });
}

// 4. Portfolio & Ledger Render
function renderPortfolio(ledger, signalsData) {
  const equityEl = document.getElementById('kpi-equity');
  const returnEl = document.getElementById('kpi-return');
  const alphaEl = document.getElementById('kpi-alpha');
  const benchmarkEl = document.getElementById('kpi-benchmark');
  const cashEl = document.getElementById('kpi-cash');
  const cbBadgeEl = document.getElementById('circuit-breaker-badge');
  const rollingDdEl = document.getElementById('kpi-drawdown');

  const equity = ledger.current_portfolio_value_try || 1000000;
  const totReturn = ledger.total_return_pct || 0;
  const alpha = ledger.net_alpha_pct || 0;
  const benchReturn = ledger.benchmark_cumulative_pct || 0;
  const cash = ledger.current_cash_try || equity;
  const rollingDd = ledger.rolling_drawdown_pct || 0;

  if (equityEl) equityEl.innerText = `${equity.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ₺`;
  if (returnEl) {
    returnEl.innerText = `${totReturn >= 0 ? '+' : ''}%${totReturn.toFixed(2)}`;
    returnEl.className = `text-lg font-bold font-mono ${totReturn >= 0 ? 'text-emerald-400' : 'text-rose-400'}`;
  }
  if (alphaEl) {
    alphaEl.innerText = `${alpha >= 0 ? '+' : ''}%${alpha.toFixed(2)}`;
    alphaEl.className = `text-lg font-bold font-mono ${alpha >= 0 ? 'text-emerald-400' : 'text-rose-400'}`;
  }
  if (benchmarkEl) benchmarkEl.innerText = `${benchReturn >= 0 ? '+' : ''}%${benchReturn.toFixed(2)}`;
  if (cashEl) cashEl.innerText = `${cash.toLocaleString('tr-TR', { minimumFractionDigits: 0, maximumFractionDigits: 0 })} ₺`;
  if (rollingDdEl) rollingDdEl.innerText = `-%${Math.abs(rollingDd).toFixed(2)}`;

  // Circuit breaker badge
  if (cbBadgeEl) {
    if (ledger.circuit_breaker_active) {
      cbBadgeEl.className = "px-3 py-1 rounded-full text-xs font-semibold badge-bear flex items-center gap-1.5";
      cbBadgeEl.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-500 live-dot"></span> DEVRE KESİCİ DEVREDE (-%5 Aşımı)`;
    } else {
      cbBadgeEl.className = "px-3 py-1 rounded-full text-xs font-semibold badge-bull flex items-center gap-1.5";
      cbBadgeEl.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400"></span> DEVRE KESİCİ: NORMAL`;
    }
  }

  // Render Charts safely
  try {
    renderEquityChart(ledger.daily_history || []);
  } catch (e) {
    console.warn("Equity chart error:", e);
  }
  
  try {
    renderAllocationChart(cash, equity, signalsData ? signalsData.top_longs : []);
  } catch (e) {
    console.warn("Allocation chart error:", e);
  }

  // Render Ledger History Table (last 15 rows)
  renderLedgerTable(ledger.daily_history || []);
}

// 5. KAP Sentiment Render
function renderSentiment(sentiment) {
  const container = document.getElementById('sentiment-list');
  const countEl = document.getElementById('sentiment-count');
  const vetoCountEl = document.getElementById('veto-count');

  if (countEl) countEl.innerText = `${sentiment.total_cached_news} Haber`;
  if (vetoCountEl) vetoCountEl.innerText = `${sentiment.vetoed_count} Veto`;

  if (!container) return;
  container.innerHTML = '';

  const items = sentiment.news_items || [];
  if (items.length === 0) {
    container.innerHTML = `<div class="text-slate-500 text-xs text-center py-4">Önbellekte haber bulunamadı.</div>`;
    return;
  }

  items.slice(0, 15).forEach(item => {
    const row = document.createElement('div');
    row.className = "p-2.5 rounded border border-slate-800/80 bg-slate-900/50 hover:border-slate-700 transition flex items-start justify-between gap-3 text-xs";

    const score = item.score || 0;
    const scorePct = Math.round((score + 1) * 50); // Map [-1, 1] to [0, 100]%
    const scoreColor = score > 0.15 ? 'text-emerald-400' : (score < -0.15 ? 'text-rose-400' : 'text-slate-400');
    
    let vetoBadge = '';
    if (item.is_vetoed) {
      vetoBadge = `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40">VETO: ${item.veto_reason || 'Risk'}</span>`;
    }

    row.innerHTML = `
      <div class="flex-1 min-w-0">
        <div class="flex items-center gap-2 mb-1">
          <span class="font-bold text-white font-mono bg-slate-800 px-1.5 py-0.5 rounded text-[10px]">${item.ticker}</span>
          <span class="text-slate-400 text-[10px]">${item.aligned_date || ''}</span>
          ${vetoBadge}
        </div>
        <div class="text-slate-300 text-xs truncate" title="${item.title}">${item.title}</div>
      </div>
      <div class="text-right flex flex-col items-end justify-center min-w-[70px]">
        <span class="font-mono text-xs font-bold ${scoreColor}">${score >= 0 ? '+' : ''}${score.toFixed(2)}</span>
        <div class="w-12 h-1 bg-slate-800 rounded-full mt-1 overflow-hidden">
          <div class="h-full ${score >= 0 ? 'bg-emerald-500' : 'bg-rose-500'}" style="width: ${scorePct}%"></div>
        </div>
      </div>
    `;
    container.appendChild(row);
  });
}

// 6. Chart.js Equity Curve
function renderEquityChart(history) {
  if (typeof Chart === 'undefined') return;
  const ctx = document.getElementById('equityChart');
  if (!ctx) return;

  const sliced = history.length > 50 ? history.slice(-50) : history;
  const labels = sliced.map(h => h.execution_date);
  const portfolioReturns = sliced.map(h => h.cumulative_return_pct);
  const benchmarkReturns = sliced.map(h => h.benchmark_cumulative_pct);

  if (equityChartInstance) {
    equityChartInstance.destroy();
  }

  equityChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'GRU Portföy (%)',
          data: portfolioReturns,
          borderColor: '#00f29d',
          backgroundColor: 'rgba(0, 242, 157, 0.08)',
          borderWidth: 2,
          fill: true,
          tension: 0.25,
          pointRadius: 0,
          pointHoverRadius: 5
        },
        {
          label: 'XU100 Benchmark (%)',
          data: benchmarkReturns,
          borderColor: '#94a3b8',
          borderWidth: 1.5,
          borderDash: [4, 4],
          fill: false,
          tension: 0.25,
          pointRadius: 0,
          pointHoverRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
        },
        tooltip: {
          backgroundColor: '#121824',
          borderColor: '#1e293b',
          borderWidth: 1,
          titleColor: '#f1f5f9',
          bodyColor: '#cbd5e1',
          padding: 10
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(30, 41, 59, 0.5)' },
          ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 8 }
        },
        y: {
          grid: { color: 'rgba(30, 41, 59, 0.5)' },
          ticks: {
            color: '#64748b',
            font: { size: 10 },
            callback: value => `%${value}`
          }
        }
      }
    }
  });
}

// 7. Chart.js Allocation Donut
function renderAllocationChart(cash, equity, topLongs) {
  if (typeof Chart === 'undefined') return;
  const ctx = document.getElementById('allocationChart');
  if (!ctx) return;

  const cashPct = Math.max(0, Math.min(100, Math.round((cash / equity) * 100)));
  const equityPct = 100 - cashPct;

  let labels = ['PPF / Repo Nakit', 'Aktif Hisse Senedi'];
  let data = [cashPct, equityPct];
  let bgColors = ['#ffb800', '#00f29d'];

  if (allocationChartInstance) {
    allocationChartInstance.destroy();
  }

  allocationChartInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: data,
        backgroundColor: bgColors,
        borderColor: '#121824',
        borderWidth: 3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#94a3b8', font: { size: 11 } }
        }
      },
      cutout: '70%'
    }
  });
}

// 8. Render Ledger Table
function renderLedgerTable(history) {
  const tableBody = document.getElementById('ledger-table-body');
  if (!tableBody) return;
  tableBody.innerHTML = '';

  const recent = [...history].reverse().slice(0, 10);
  recent.forEach(row => {
    const tr = document.createElement('tr');
    tr.className = "border-b border-slate-800/60 hover:bg-slate-800/30 transition font-mono text-xs";
    
    const pnl = row.net_daily_pnl_try || 0;
    const pnlColor = pnl >= 0 ? 'text-emerald-400' : 'text-rose-400';
    const retPct = row.daily_return_pct || 0;
    const alpha = row.alpha_daily_pct || 0;

    tr.innerHTML = `
      <td class="py-2.5 px-3 text-slate-300 font-bold">${row.execution_date}</td>
      <td class="py-2.5 px-3 text-slate-400">${row.action}</td>
      <td class="py-2.5 px-3 text-right text-slate-300">${(row.end_equity_try || 0).toLocaleString('tr-TR', { maximumFractionDigits: 0 })} ₺</td>
      <td class="py-2.5 px-3 text-right font-bold ${pnlColor}">${pnl >= 0 ? '+' : ''}${pnl.toLocaleString('tr-TR', { maximumFractionDigits: 0 })} ₺</td>
      <td class="py-2.5 px-3 text-right ${pnlColor}">${retPct >= 0 ? '+' : ''}%${retPct.toFixed(2)}</td>
      <td class="py-2.5 px-3 text-right ${alpha >= 0 ? 'text-emerald-400' : 'text-rose-400'}">${alpha >= 0 ? '+' : ''}%${alpha.toFixed(2)}</td>
      <td class="py-2.5 px-3 text-center text-slate-400">${row.trades_count || 0}</td>
    `;
    tableBody.appendChild(tr);
  });
}

// 9. Trigger Daily Execution
async function triggerExecution(isForce = false) {
  const btn = isForce ? document.getElementById('btn-force-execute') : document.getElementById('btn-execute-daily');
  const defaultText = isForce ? "Mükerrer İcra Yap (Zorla)" : "17:50 Kapanış Seansı Koştur";
  
  if (isForce) {
    const confirmed = confirm("DİKKAT: Bugün için zaten bir icra kaydı yapılmış olabilir.\n\nMükerrer kayıt oluşturarak 17:50 Kapanış Seansı emir icrasını zorla koşturmak istiyor musunuz?");
    if (!confirmed) return;
  }

  if (btn) {
    btn.disabled = true;
    btn.innerText = "İcra Ediliyor...";
  }

  showToast(isForce ? "Mükerrer icra motoru zorla çalıştırılıyor..." : "17:50 Kapanış seansı icra motoru çalıştırılıyor...", "info");

  try {
    const res = await fetch('/api/v1/execution/run-daily', {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ mode: 'close_auction', force_cash: false, rebalance_step: 5, force_execution: isForce })
    });
    const data = await res.json();
    
    if (res.ok) {
      showToast(isForce ? "Mükerrer icra başarıyla tamamlandı! Defter güncellendi." : "Kapanış seansı icrası başarıyla tamamlandı! Defter güncellendi.", "success");
      await loadDashboardData(true);
    } else if (res.status === 409) {
      // 409 Conflict: Kullanıcıya mükerrer çalıştırma seçeneği sun
      const userWantsForce = confirm(
        `[MÜKERRER İCRA ENGELİ]\n\n${data.detail || 'Bugün için icra zaten gerçekleştirildi.'}\n\nYine de mükerrer kayıt oluşturarak zorla icra yapmak istiyor musunuz?`
      );
      if (userWantsForce) {
        await triggerExecution(true);
      } else {
        showToast("Mükerrer icra işlemi iptal edildi.", "info");
      }
    } else {
      showToast(`İcra hatası: ${data.detail || 'Bilinmeyen hata'}`, "error");
    }
  } catch (err) {
    showToast("İcra servisine ulaşılamadı.", "error");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = defaultText;
    }
  }
}

// Initial setup
document.addEventListener('DOMContentLoaded', () => {
  loadDashboardData();

  const btnRefresh = document.getElementById('btn-refresh');
  if (btnRefresh) {
    btnRefresh.addEventListener('click', manualRefreshData);
  }

  const btnRecalc = document.getElementById('btn-recalc-signals');
  if (btnRecalc) {
    btnRecalc.addEventListener('click', recalculateSignals);
  }

  const btnExec = document.getElementById('btn-execute-daily');
  if (btnExec) {
    btnExec.addEventListener('click', () => triggerExecution(false));
  }

  const btnForceExec = document.getElementById('btn-force-execute');
  if (btnForceExec) {
    btnForceExec.addEventListener('click', () => triggerExecution(true));
  }

  // Auto-refresh every 60 seconds
  setInterval(() => loadDashboardData(false), 60000);
});
