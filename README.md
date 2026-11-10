# DDS001 AT4 | 旅遊需求預測專家融合架構簡化復現

**作者：黃達｜澳門城市大學 數據科學學院｜DDS001《數據科學前沿分析研究》｜2026-11-10**

本專案為 AT4 期末整合報告的 **GitHub-ready repository**。`paper.pdf` 是 10 頁論文；`src/` 是原先 2026-10-28 AT3 實際使用的完整資料準備、訓練與報告生成程式；`data/`、`results/`、`logs/` 保存該次試驗的輸入、輸出及稽核證據。

> **研究邊界：** 是 SG-MSF 「特徵族專家 + context gate」的**簡化架構復現**，不是 Zou et al. (2026) 六個獨立訊號來源的完整復現。所用酒店與其他住宿資料均來自同一個 Eurostat 表，不能宣稱六源融合的性能。

## 主要結果（2026-10-28 AT3 原始實驗）

27 國 MASE 中位數（越低越好）：

| 模型 | H=1 | H=3 | H=6 |
|---|---:|---:|---:|
| Seasonal naive | 0.433 | 0.433 | 0.433 |
| Equal-weight experts | **0.396** | 0.412 | 0.624 |
| Early-fusion LightGBM | 0.406 | 0.443 | 0.514 |
| Context-gated fusion | 0.439 | **0.398** | 0.658 |

**不可忽略的對照：** WAPE 由 seasonal naive 在 1／3／6 月預測期限均勝出。上述數值是已有 AT3 的結果，不是生成此 AT4 時重新訓練得到的值；原文作者的 SG-MSF 數字另存 `results/paper_reported_reference_metrics.csv`，請勿與本實驗數字直接做「改善幅度」比較。

## 目錄

```text
./
├── paper.pdf                       # AT4 10 頁正式論文
├── README.md
├── requirements.txt
├── src/
│   ├── prepare_data.py            # 從原始 Eurostat Excel 擷取資料
│   ├── run_experiment.py          # 特徵、LightGBM、門控、指標、圖表
│   ├── build_report.py            # AT3 舊報告生成器（非 AT4 論文）
│   └── verify_artifacts.py        # 無須重新訓練的離線結果稽核
├── data/raw/                      # Eurostat 2026-10-23 已下載工作簿
├── data/processed/                # 長格式 panel 與資料品質／來源說明
├── results/                       # 預測值、指標、門控權重、實驗記錄
│   └── figures/                   # 3 張 AT3 實驗圖
├── figures/                        # 原始腳本輸出位置
├── logs/                           # 門控訓練／重訓稽核／超參數
├── reproducibility.md
└── presentation_outline.md
```

## 執行環境與步驟

建議 Python **3.10+**，需有 CPU 與足夠記憶體；torch 與 LightGBM 在部分系統須依官方安裝說明先安裝。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**A. 快速核對已有證據**（不需重訓、通常數秒）：

```bash
python src/verify_artifacts.py
```

**B. 從本地原始工作簿重建乾淨資料：**

```bash
python src/prepare_data.py
```

**C. 重新完整訓練／繪圖**（CPU 耗時可能較長，會覆蓋 `results/`、`logs/` 中同名檔案；請先備份）：

```bash
python src/run_experiment.py
```

**D. 舊 AT3 報告生成器（選做；非 10 頁 AT4 論文）：**

```bash
python src/build_report.py
```

要從原始 CSV 重算全部特徵，先刪除 `results/feature_panel.csv` 快取再執行 C；否則程式會讀取既有面板，加速重訓。

## 資料定義與時間切分

- 目標：EU27、非居民 `FOR`、數量 `NR`、NACE `I551-I553` 的月住宿夜數；2015-01 至 2025-12。
- 原始輸入：`data/raw/tour_occ_nim__custom_23540219_spreadsheet.xlsx`；詳見 `data/processed/data_provenance.txt`。
- 3,564 理論觀測，69 缺失；缺失不設成零。
- H=1/3/6；源特徵使用截至預測 cutoff 的歷史資料，程式採兩個月發布延遲假設。
- 專家每年擴展窗口重訓；門控使用 2022 擬合、2023 驗證／早停，2024–2025 測試。
- 結果同時報國家級 MASE 中位數與住宿夜數加權 WAPE，不僅報最佳模型。

## 著作權、來源及透明度

- Eurostat `tour_occ_nim`，DOI: <https://doi.org/10.2908/tour_occ_nim>。本資料附來源資訊，請遵守 Eurostat 再利用與標註條款。
- 論文方法出自 Zou et al. (2026)，DOI: <https://doi.org/10.1016/j.sasc.2026.200665>。
- 本倉庫**不**再分發原出版商的 article PDF；請透過 DOI 查閱原文。
- 請閱讀 `reproducibility.md`，其中明確區分既有實測、可在本機核對的數據，以及尚未完成的測試。

## 上傳 GitHub

解壓此 ZIP，將資料夾內容放到新 repository 根目錄，`git add . && git commit -m "DDS001 AT4 reproducible report" && git push`。請勿把原文 PDF、個人私密資料或尚未允許公開的內容推送到公開倉庫。
